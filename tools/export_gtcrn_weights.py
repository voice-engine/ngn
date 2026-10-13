#!/usr/bin/env python3
"""从 GTCRN checkpoint 导出 C 权重 → ngn/algo/gtcrn_weights.{c,h}。

导出时的等价变换（保持数学不变，简化 C 实现）：
- BatchNorm 推理态折叠进前层：W' = W·s,  b' = s·b + β − s·μ,  s = γ/√(σ²+1e-5)
- 解码器 GT 块 depthwise ConvTranspose 按 stream/modules/convert.py 规则做
  T/F 双轴 flip（形状相同无需 permute），之后与普通 depthwise 相关同一代码路径
- GT 块 1×1 ConvTranspose（in,out,1,1）折叠后转置为 (out,in) 的 GEMV 布局
- 频域 (1,5,s2) ConvTranspose（ConvBlock）保持原 scatter 布局 (in, out/g, 1, 5)
- GRU 合并偏置 b = b_ih + b_rh；gate 顺序 torch 默认 [r, z, n]

用法: export_gtcrn_weights.py [--ckpt model.tar] [--gtcrn-repo DIR] [--out-dir DIR]
默认 ckpt 为官方仓库 stream/onnx_models/model_trained_on_dns3.tar（与官方 ONNX 配套）。
"""
import argparse
import os
import sys

import numpy as np
import torch

BN_EPS = 1e-5


def fold_bn_axis(w, b, bn, axis):
    """沿 out 通道轴折叠推理态 BN：W'=W·s, b'=s·b+β−s·μ, s=γ/√(σ²+ε)。"""
    g = bn[".weight"].numpy()
    beta = bn[".bias"].numpy()
    mu = bn[".running_mean"].numpy()
    rv = bn[".running_var"].numpy()
    s = g / np.sqrt(rv + BN_EPS)
    sh = [1] * w.ndim
    sh[axis] = -1
    w2 = w * s.reshape(sh)
    b2 = s * b + beta - s * mu
    return w2.astype(np.float32), b2.astype(np.float32)


def fold_bn_convt_grouped(w, b, bn, groups):
    """分组 ConvTranspose 折叠：weight (in, out/g, 1, k)，out 全通道 BN。

    w[j][oc] 连接 in j（组 g=j//g_in）→ out 通道 g·(out/g)+oc，
    尺度按真实 out 通道取。
    """
    g = bn[".weight"].numpy()
    beta = bn[".bias"].numpy()
    mu = bn[".running_mean"].numpy()
    rv = bn[".running_var"].numpy()
    s = g / np.sqrt(rv + BN_EPS)
    og = w.shape[1]                       # out per group
    s_map = np.empty_like(w)
    for j in range(w.shape[0]):
        s_map[j] = s[(j // og) * og:(j // og + 1) * og].reshape(1, og, 1, 1)
    w2 = (w * s_map).astype(np.float32)
    b2 = (s * b + beta - s * mu).astype(np.float32)
    return w2, b2


def gru_params(sd, prefix, suffix=""):
    """b_ih 与 b_hh 必须分开：n 门为 tanh(gi_n + r*(gh_n))，合并会改变语义。"""
    wi = sd[prefix + ".weight_ih_l0" + suffix].numpy()
    wh = sd[prefix + ".weight_hh_l0" + suffix].numpy()
    bi = sd[prefix + ".bias_ih_l0" + suffix].numpy()
    bh = sd[prefix + ".bias_hh_l0" + suffix].numpy()
    return wi, wh, bi, bh


def gru_params_bi(sd, prefix):
    return {tag: gru_params(sd, prefix, suf)
            for tag, suf in (("f", ""), ("b", "_reverse"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser(
        "~/lab/nanopi-neo-air/gtcrn/stream/onnx_models/model_trained_on_dns3.tar"))
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(__file__), "..", "algo"))
    args = ap.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu")["model"]
    sd = {k: v for k, v in sd.items() if not k.endswith("num_batches_tracked")}

    arrs = {}  # name -> np.float32 1-D

    def put(name, a):
        a = np.ascontiguousarray(a, dtype=np.float32).ravel()
        arrs[name] = a

    def get(k):
        return sd[k].numpy()

    def bn_of(k):
        return {n: sd[k + n] for n in (".weight", ".bias", ".running_mean", ".running_var")}

    # ---- ERB 固定矩阵 ----
    put("erb_fc", get("erb.erb_fc.weight"))      # (64,192)
    put("ierb_fc", get("erb.ierb_fc.weight"))    # (192,64)

    # ---- encoder 频域 conv ----
    for i in range(2):
        p = f"encoder.en_convs.{i}"
        w = get(p + ".conv.weight")              # (out, in/g, 1, 5)
        b = get(p + ".conv.bias")
        w, b = fold_bn_axis(w, b, bn_of(p + ".bn"), axis=0)
        put(f"en{i}_w", w)
        put(f"en{i}_b", b)
        put(f"en{i}_act", get(p + ".act.weight"))

    # ---- GT 块（编码 3 + 解码 3） ----
    for side, blocks in (("e", [2, 3, 4]), ("d", [0, 1, 2])):
        for j, i in enumerate(blocks):
            p = f"decoder.de_convs.{i}" if side == "d" else f"encoder.en_convs.{i}"
            pc_axis = 0 if side == "e" else 1        # 编码 conv2d / 解码 convT
            pc1 = get(p + ".point_conv1.weight")     # (24,16,1,1) convT 于解码器
            pc1b = get(p + ".point_conv1.bias")
            pc1, pc1b = fold_bn_axis(pc1, pc1b, bn_of(p + ".point_bn1"), axis=pc_axis)
            # 编码器 conv2d 权重本就是 (out,in)；解码器 convT (in,out) 需转置
            pc1 = pc1[:, :, 0, 0] if side == "e" else pc1[:, :, 0, 0].T
            put(f"gt{side}{j}_pc1_w", pc1)                  # 统一 (16,24)
            put(f"gt{side}{j}_pc1_b", pc1b)
            put(f"gt{side}{j}_pc1_act", get(p + ".point_act.weight"))

            dw = get(p + ".depth_conv.weight")        # (16,1,3,3)
            dwb = get(p + ".depth_conv.bias")
            dw, dwb = fold_bn_axis(dw, dwb, bn_of(p + ".depth_bn"), axis=0)
            if side == "d":
                dw = dw[:, :, ::-1, ::-1]             # stream convert: T/F 双轴 flip
            put(f"gt{side}{j}_dw_w", dw)
            put(f"gt{side}{j}_dw_b", dwb)
            put(f"gt{side}{j}_dw_act", get(p + ".depth_act.weight"))

            pc2 = get(p + ".point_conv2.weight")      # (16,8,1,1) convT 于解码器
            pc2b = get(p + ".point_conv2.bias")
            pc2, pc2b = fold_bn_axis(pc2, pc2b, bn_of(p + ".point_bn2"), axis=pc_axis)
            pc2 = pc2[:, :, 0, 0] if side == "e" else pc2[:, :, 0, 0].T
            put(f"gt{side}{j}_pc2_w", pc2)                  # 统一 (8,16)
            put(f"gt{side}{j}_pc2_b", pc2b)

            wi, wh, bi, bh = gru_params(sd, p + ".tra.att_gru")
            put(f"gt{side}{j}_tra_wih", wi)           # (48,8)
            put(f"gt{side}{j}_tra_whh", wh)           # (48,16)
            put(f"gt{side}{j}_tra_bih", bi)
            put(f"gt{side}{j}_tra_bhh", bh)
            put(f"gt{side}{j}_tra_fc_w", get(p + ".tra.att_fc.weight"))  # (8,16)
            put(f"gt{side}{j}_tra_fc_b", get(p + ".tra.att_fc.bias"))

    # ---- DPGRNN ×2 ----
    for j in (1, 2):
        p = f"dpgrnn{j}"
        for half in (1, 2):
            g = gru_params_bi(sd, f"{p}.intra_rnn.rnn{half}")
            for tag in ("f", "b"):
                put(f"dp{j}_intra{half}{tag}_wih", g[tag][0])  # (12,8)
                put(f"dp{j}_intra{half}{tag}_whh", g[tag][1])  # (12,4)
                put(f"dp{j}_intra{half}{tag}_bih", g[tag][2])
                put(f"dp{j}_intra{half}{tag}_bhh", g[tag][3])
            wi, wh, bi, bh = gru_params(sd, f"{p}.inter_rnn.rnn{half}")
            put(f"dp{j}_inter{half}_wih", wi)                  # (24,8)
            put(f"dp{j}_inter{half}_whh", wh)                  # (24,8)
            put(f"dp{j}_inter{half}_bih", bi)
            put(f"dp{j}_inter{half}_bhh", bh)
        for tag in ("intra", "inter"):
            put(f"dp{j}_{tag}_fc_w", get(f"{p}.{tag}_fc.weight"))  # (16,16)
            put(f"dp{j}_{tag}_fc_b", get(f"{p}.{tag}_fc.bias"))
            put(f"dp{j}_{tag}_ln_w", get(f"{p}.{tag}_ln.weight"))  # (528,)
            put(f"dp{j}_{tag}_ln_b", get(f"{p}.{tag}_ln.bias"))

    # ---- decoder 频域 deconv（保持 ConvTranspose scatter 布局） ----
    for i in (3, 4):
        p = f"decoder.de_convs.{i}"
        w = get(p + ".conv.weight")                  # (in, out/g, 1, 5)
        b = get(p + ".conv.bias")
        if i == 3:                                   # g=2 需按组映射 out 尺度
            w, b = fold_bn_convt_grouped(w, b, bn_of(p + ".bn"), groups=2)
        else:
            w, b = fold_bn_axis(w, b, bn_of(p + ".bn"), axis=1)
        put(f"de{i}_w", w)
        put(f"de{i}_b", b)
        if i == 3:
            put("de3_act", get(p + ".act.weight"))   # de4 is_last → tanh 无权重

    # ---- 输出 ----
    total = sum(a.size for a in arrs.values())
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "gtcrn_weights.h"), "w") as fh:
        fh.write("/* 由 tools/export_gtcrn_weights.py 生成，勿手改。\n"
                 f" * 源: {os.path.basename(args.ckpt)}  共 {total} 个 fp32 权重 */\n"
                 "#ifndef GTCRN_WEIGHTS_H\n#define GTCRN_WEIGHTS_H\n")
        for name, a in arrs.items():
            fh.write(f"extern const float gw_{name}[{a.size}];\n")
        fh.write("#endif\n")

    with open(os.path.join(args.out_dir, "gtcrn_weights.c"), "w") as fc:
        fc.write("/* 由 tools/export_gtcrn_weights.py 生成，勿手改 */\n"
                 '#include "gtcrn_weights.h"\n\n')
        for name, a in arrs.items():
            fc.write(f"const float gw_{name}[{a.size}] = {{\n")
            for i in range(0, a.size, 10):
                fc.write(" ".join("%.9g," % v for v in a[i:i + 10]) + "\n")
            fc.write("};\n\n")

    np.savez(os.path.join(args.out_dir, "gtcrn_weights.npz"), **arrs)
    print(f"exported {len(arrs)} arrays, {total} floats "
          f"({total * 4 / 1024:.0f} KB) → {args.out_dir}/gtcrn_weights.[ch]/npz")


if __name__ == "__main__":
    main()
