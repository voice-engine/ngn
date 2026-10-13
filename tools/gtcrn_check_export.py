#!/usr/bin/env python3
"""用导出的权重（gtcrn_weights.npz）按「C 实现计划语义」逐帧前向，
对照官方 StreamGTCRN torch 输出验证导出变换（BN 折叠/flip/GRU/ConvT scatter）。

DEBUG=1 时对第 0 帧逐级对拍，定位第一处发散。
"""
import os
import sys

import numpy as np
import torch

GT = os.path.expanduser("~/lab/nanopi-neo-air/gtcrn")
NGN_ALGO = os.path.join(os.path.dirname(__file__), "..", "algo")


def prelu(x, w):
    return np.where(x > 0, x, w * x)


def gru_cell(x, h, wih, whh, bi, bh):
    gi = x @ wih.T + bi
    gh = h @ whh.T + bh
    H = h.shape[-1]
    r = 1 / (1 + np.exp(-(gi[:, :H] + gh[:, :H])))
    z = 1 / (1 + np.exp(-(gi[:, H:2 * H] + gh[:, H:2 * H])))
    n = np.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
    return (1 - z) * n + z * h


def layernorm(x, w, b):  # x (33,16) 整块归一，逐位置仿射
    mu = x.mean()
    var = x.var()
    return (x - mu) / np.sqrt(var + 1e-8) * w.reshape(33, 16) + b.reshape(33, 16)


def reshape2(gw):
    gw = dict(gw)
    gw["erb_fc"] = gw["erb_fc"].reshape(64, 192)
    gw["ierb_fc"] = gw["ierb_fc"].reshape(192, 64)
    for k in list(gw):
        if k.endswith(("pc1_w", "pc2_w")):
            gw[k] = gw[k].reshape(16 if k.endswith("pc1_w") else 8, -1)
        elif k.endswith("_tra_fc_w"):
            gw[k] = gw[k].reshape(8, 16)
        elif k.endswith("_fc_w"):
            gw[k] = gw[k].reshape(16, 16)
        elif k.endswith("_wih"):
            gw[k] = gw[k].reshape(-1, 8)
        elif k.endswith("_whh"):
            gw[k] = gw[k].reshape(-1, 4 if "intra" in k else (16 if "_tra_" in k else 8))
    return gw


class Ref:
    """与计划中的 C 实现逐算子一致。stages 非空时记录中间量。"""

    def __init__(self, gw):
        self.g = reshape2(gw)

    def sfe(self, x):  # (C,F) → (C*3,F)：通道 c*3+j = x[c][f+j-1]
        C, F = x.shape
        p = np.pad(x, ((0, 0), (1, 1)))
        return np.stack([p[:, j:j + F] for j in range(3)], axis=1).reshape(C * 3, F)

    def gt_block(self, x, side, j, dil, conv_cache, tra_h, st=None, tag=""):
        g = self.g
        x1, x2 = x[:8], x[8:]
        s = self.sfe(x1)                                   # (24,33)
        h = s.T @ g[f"gt{side}{j}_pc1_w"].T + g[f"gt{side}{j}_pc1_b"]
        h = prelu(h.T, g[f"gt{side}{j}_pc1_act"][0])
        if st is not None:
            st[f"{tag}pc1"] = h
        win = np.concatenate([conv_cache, h[None]], axis=0)  # (2d+1,16,33)
        w = g[f"gt{side}{j}_dw_w"].reshape(16, 3, 3)
        hb = g[f"gt{side}{j}_dw_b"]
        hp = np.pad(win, ((0, 0), (0, 0), (1, 1)))
        out = np.tile(hb[:, None], (1, 33))
        for c in range(16):
            for kt in range(3):
                fr = win.shape[0] - 1 - (2 - kt) * dil      # 抽头 t-(2-kt)d
                for kf in range(3):
                    out[c] += w[c, kt, kf] * hp[fr, c, kf:kf + 33]
        out = prelu(out, g[f"gt{side}{j}_dw_act"][0])
        if st is not None:
            st[f"{tag}dw"] = out
        out = (g[f"gt{side}{j}_pc2_w"] @ out).T + g[f"gt{side}{j}_pc2_b"]
        out = out.T                                          # (8,33)
        if st is not None:
            st[f"{tag}pc2"] = out
        zt = (out ** 2).mean(axis=1)
        hh = gru_cell(zt[None], tra_h, g[f"gt{side}{j}_tra_wih"],
                      g[f"gt{side}{j}_tra_whh"], g[f"gt{side}{j}_tra_bih"], g[f"gt{side}{j}_tra_bhh"])
        at = 1 / (1 + np.exp(-(hh[0] @ g[f"gt{side}{j}_tra_fc_w"].T
                               + g[f"gt{side}{j}_tra_fc_b"])))
        out = out * at[:, None]
        if st is not None:
            st[f"{tag}tra"] = out
        res = np.empty((16, 33), np.float32)
        res[0::2] = out
        res[1::2] = x2
        return res, win[1:], hh

    def dpgrnn(self, x, j, inter_h, st=None, tag=""):
        g = self.g
        X = x.T                                              # (33,16)
        y = np.zeros((33, 16), np.float32)
        for half in (0, 1):
            xs = X[:, half * 8:half * 8 + 8]
            hf = np.zeros((1, 4), np.float32)
            hb = np.zeros((1, 4), np.float32)
            outf, outb = np.zeros((33, 4), np.float32), np.zeros((33, 4), np.float32)
            for f in range(33):
                hf = gru_cell(xs[f:f + 1], hf, g[f"dp{j}_intra{half+1}f_wih"],
                              g[f"dp{j}_intra{half+1}f_whh"], g[f"dp{j}_intra{half+1}f_bih"], g[f"dp{j}_intra{half+1}f_bhh"])
                outf[f] = hf[0]
            for f in range(32, -1, -1):
                hb = gru_cell(xs[f:f + 1], hb, g[f"dp{j}_intra{half+1}b_wih"],
                              g[f"dp{j}_intra{half+1}b_whh"], g[f"dp{j}_intra{half+1}b_bih"], g[f"dp{j}_intra{half+1}b_bhh"])
                outb[f] = hb[0]
            y[:, half * 8:half * 8 + 8] = np.concatenate([outf, outb], axis=1)
        if st is not None:
            st[f"{tag}intra_rnn"] = y
        y = y @ g[f"dp{j}_intra_fc_w"].T + g[f"dp{j}_intra_fc_b"]
        y = layernorm(y, g[f"dp{j}_intra_ln_w"], g[f"dp{j}_intra_ln_b"])
        intra_out = X + y
        if st is not None:
            st[f"{tag}intra"] = intra_out
        z = np.zeros((33, 16), np.float32)
        for f in range(33):
            for half in (0, 1):
                hprev = inter_h[f, half * 8:half * 8 + 8][None]
                hn = gru_cell(intra_out[f, half * 8:half * 8 + 8][None], hprev,
                              g[f"dp{j}_inter{half+1}_wih"], g[f"dp{j}_inter{half+1}_whh"],
                              g[f"dp{j}_inter{half+1}_bih"], g[f"dp{j}_inter{half+1}_bhh"])
                z[f, half * 8:half * 8 + 8] = hn[0]
                inter_h[f, half * 8:half * 8 + 8] = hn[0]
        if st is not None:
            st[f"{tag}inter_rnn"] = z
        z = z @ g[f"dp{j}_inter_fc_w"].T + g[f"dp{j}_inter_fc_b"]
        z = layernorm(z, g[f"dp{j}_inter_ln_w"], g[f"dp{j}_inter_ln_b"])
        return (intra_out + z).T, inter_h

    def frame(self, spec257, caches, st=None):
        g = self.g
        feat = np.stack([np.sqrt(spec257[:, 0] ** 2 + spec257[:, 1] ** 2 + 1e-12),
                         spec257[:, 0], spec257[:, 1]])
        low, high = feat[:, :65], feat[:, 65:]
        bm = np.concatenate([low, high @ g["erb_fc"].T], axis=1)
        if st is not None:
            st["feat"] = feat
            st["bm"] = bm
        s9 = self.sfe(bm)
        if st is not None:
            st["sfe9"] = s9
        # en0: (1,5) s2 p2，129→65
        w = g["en0_w"].reshape(16, 9, 5)
        hp = np.pad(s9, ((0, 0), (2, 2)))
        a65 = np.tile(g["en0_b"][:, None], (1, 65))
        for o in range(16):
            for ci in range(9):
                for k in range(5):
                    a65[o] += w[o, ci, k] * hp[ci, k:k + 129:2]
        a65 = prelu(a65, g["en0_act"][0])
        if st is not None:
            st["en0"] = a65
        # en1: g2 (1,5) s2 p2，65→33
        w = g["en1_w"].reshape(16, 8, 5)
        hp = np.pad(a65, ((0, 0), (2, 2)))
        a33 = np.tile(g["en1_b"][:, None], (1, 33))
        for o in range(16):
            for ci in range(8):
                for k in range(5):
                    a33[o] += w[o, ci, k] * hp[(o // 8) * 8 + ci, k:k + 66:2]
        a33 = prelu(a33, g["en1_act"][0])
        if st is not None:
            st["en1"] = a33
        skips = [a65.copy(), a33.copy()]
        for j, dil in enumerate((1, 2, 5)):
            a33, caches["ec"][j], caches["et"][j] = self.gt_block(
                a33, "e", j, dil, caches["ec"][j], caches["et"][j], st, f"gte{j}.")
            skips.append(a33.copy())
            if st is not None:
                st[f"gte{j}"] = a33
        a33, caches["i1"] = self.dpgrnn(a33, 1, caches["i1"], st, "dp1.")
        if st is not None:
            st["dp1"] = a33
        a33, caches["i2"] = self.dpgrnn(a33, 2, caches["i2"], st, "dp2.")
        if st is not None:
            st["dp2"] = a33
        for j, dil in enumerate((5, 2, 1)):
            x = a33 + skips[4 - j]
            if st is not None:
                st[f"gtd{j}_in"] = x
            x, caches["dc"][j], caches["dt"][j] = self.gt_block(
                x, "d", j, dil, caches["dc"][j], caches["dt"][j], st, f"gtd{j}.")
            a33 = x
            if st is not None:
                st[f"gtd{j}"] = a33
        # de3: ConvT g2 (1,5) s2 p2，33→65（先加 skip en1）
        a33 = a33 + skips[1]
        if st is not None:
            st["de3_in"] = a33
        w = g["de3_w"].reshape(16, 8, 5)
        a65 = np.tile(g["de3_b"][:, None], (1, 65))
        for ci in range(16):
            for oc in range(8):
                for jin in range(33):
                    for k in range(5):
                        o = 2 * jin - 2 + k
                        if 0 <= o < 65:
                            a65[(ci // 8) * 8 + oc, o] += w[ci, oc, k] * a33[ci, jin]
        a65 = prelu(a65, g["de3_act"][0])
        if st is not None:
            st["de3"] = a65
        # de4: ConvT (1,5) s2 p2，65→129 + tanh（先加 skip en0）
        a65 = a65 + skips[0]
        w = g["de4_w"].reshape(16, 2, 5)
        m129 = np.tile(g["de4_b"][:, None], (1, 129))
        for ci in range(16):
            for oc in range(2):
                for jin in range(65):
                    for k in range(5):
                        o = 2 * jin - 2 + k
                        if 0 <= o < 129:
                            m129[oc, o] += w[ci, oc, k] * a65[ci, jin]
        m129 = np.tanh(m129)
        if st is not None:
            st["de4"] = m129
        m = np.concatenate([m129[:, :65], m129[:, 65:] @ g["ierb_fc"].T], axis=1)
        sr_, si_ = spec257[:, 0], spec257[:, 1]
        mr, mi = m[0], m[1]
        enh = np.stack([sr_ * mr - si_ * mi, si_ * mr + sr_ * mi], axis=1)
        if st is not None:
            st["m"] = m
            st["enh"] = enh
        return enh

    @staticmethod
    def fresh_caches():
        z = lambda *s: np.zeros(s, np.float32)
        return {"ec": [z(2, 16, 33), z(4, 16, 33), z(10, 16, 33)],
                "dc": [z(10, 16, 33), z(4, 16, 33), z(2, 16, 33)],
                "et": [z(1, 16)] * 3, "dt": [z(1, 16)] * 3,
                "i1": z(33, 16), "i2": z(33, 16)}


def torch_stages(tm, spec257, caches_t):
    """torch 流式模型逐模块前向，返回 (各级中间量, 更新后的 caches)。"""
    st = {}
    cc, tc, ic = caches_t
    with torch.no_grad():
        spec = torch.from_numpy(spec257)[None, :, None, :]
        r = spec[..., 0].permute(0, 2, 1)
        i = spec[..., 1].permute(0, 2, 1)
        mag = torch.sqrt(r ** 2 + i ** 2 + 1e-12)
        feat = torch.stack([mag, r, i], dim=1)
        st["feat"] = feat[0, :, 0, :].numpy()
        bm = tm.erb.bm(feat)
        st["bm"] = bm[0, :, 0, :].numpy()
        s9 = tm.sfe(bm)
        st["sfe9"] = s9[0, :, 0, :].numpy()
        x = s9
        en_outs = []
        x = tm.encoder.en_convs[0](x)
        st["en0"] = x[0, :, 0, :].numpy()
        en_outs.append(x)
        x = tm.encoder.en_convs[1](x)
        st["en1"] = x[0, :, 0, :].numpy()
        en_outs.append(x)
        for k, (lo, hi) in enumerate(((0, 2), (2, 6), (6, 16))):
            g = tm.encoder.en_convs[2 + k]
            x1, x2 = x[:, :8], x[:, 8:]
            xs = g.sfe(x1)
            h1 = g.point_act(g.point_bn1(g.point_conv1(xs)))
            st[f"gte{k}.pc1"] = h1[0, :, 0, :].numpy()
            h1, cc[0][:, :, lo:hi, :] = g.depth_conv(h1, cc[0][:, :, lo:hi, :])
            h1 = g.depth_act(g.depth_bn(h1))
            st[f"gte{k}.dw"] = h1[0, :, 0, :].numpy()
            h1 = g.point_bn2(g.point_conv2(h1))
            st[f"gte{k}.pc2"] = h1[0, :, 0, :].numpy()
            h1, tc[0][k] = g.tra(h1, tc[0][k])
            st[f"gte{k}.tra"] = h1[0, :, 0, :].numpy()
            xs2 = torch.stack([h1, x2], dim=1).transpose(1, 2).reshape(
                x.shape[0], -1, x.shape[2], x.shape[3])
            x = xs2
            st[f"gte{k}"] = x[0, :, 0, :].numpy()
            en_outs.append(x)
        for j, d in ((1, tm.dpgrnn1), (2, tm.dpgrnn2)):
            xt = x.permute(0, 2, 3, 1)                       # (B,T,F,C)
            ia = d.intra_rnn(xt.reshape(-1, 33, 16))[0]
            st[f"dp{j}.intra_rnn"] = ia[0].numpy()
            ia = d.intra_ln(d.intra_fc(ia)).reshape(1, 1, 33, 16)
            intra_out = torch.add(xt, ia)
            st[f"dp{j}.intra"] = intra_out[0, 0].numpy()
            ie = intra_out.permute(0, 2, 1, 3).reshape(-1, 1, 16)
            ie, ic[j - 1] = d.inter_rnn(ie, ic[j - 1])
            st[f"dp{j}.inter_rnn"] = ie.reshape(33, 16).numpy()
            ie = d.inter_fc(ie)
            ie = d.inter_ln(ie.reshape(1, 33, 1, 16).permute(0, 2, 1, 3))
            x = torch.add(intra_out, ie).permute(0, 3, 1, 2)
            st[f"dp{j}"] = x[0, :, 0, :].numpy()
        for j, (lo, hi) in enumerate(((6, 16), (2, 6), (0, 2))):
            xin = x + en_outs[4 - j]
            st[f"gtd{j}_in"] = xin[0, :, 0, :].numpy()
            out = tm.decoder.de_convs[j](xin, cc[1][:, :, lo:hi, :], tc[1][j])
            x, cc[1][:, :, lo:hi, :], tc[1][j] = out[0], out[1], out[2]
            st[f"gtd{j}"] = x[0, :, 0, :].numpy()
        x = tm.decoder.de_convs[3](x + en_outs[1])
        st["de3"] = x[0, :, 0, :].numpy()
        x = tm.decoder.de_convs[4](x + en_outs[0])
        st["de4"] = x[0, :, 0, :].numpy()
        m = tm.erb.bs(x)
        st["m"] = m[0, :, 0, :].numpy()
        spec_ref = spec.permute(0, 3, 2, 1)
        enh = tm.mask(m, spec_ref)
        st["enh"] = enh[0, :, 0, :].T.numpy()
    return st, (cc, tc, ic)


def fresh_torch_caches():
    return [torch.zeros(2, 1, 16, 16, 33), torch.zeros(2, 3, 1, 1, 16),
            torch.zeros(2, 1, 33, 16)]


def main():
    sys.path.insert(0, os.path.join(GT, "stream"))
    import soundfile as sf
    import gtcrn
    from gtcrn_stream import StreamGTCRN
    from modules.convert import convert_to_stream

    gw = dict(np.load(os.path.join(NGN_ALGO, "gtcrn_weights.npz")))
    offline = gtcrn.GTCRN().eval()
    ckpt = torch.load(os.path.join(GT, "stream/onnx_models/model_trained_on_dns3.tar"),
                      map_location="cpu")
    offline.load_state_dict(ckpt["model"])
    tmodel = StreamGTCRN().eval()
    convert_to_stream(tmodel, offline)

    mix, fs = sf.read(os.path.join(GT, "test_wavs/mix.wav"), dtype="float32")
    n = (len(mix) + 255) // 256 * 256
    mix = np.pad(mix, (0, n - len(mix)))
    win = torch.hann_window(512).pow(0.5)
    spec = torch.stft(torch.from_numpy(mix), 512, 256, 512, win, return_complex=True)
    X = torch.view_as_real(spec).numpy()

    ref = Ref(gw)
    caches = Ref.fresh_caches()
    tcaches = fresh_torch_caches()
    debug = os.environ.get("DEBUG") == "1"
    NF = int(os.environ.get("FRAMES", "12"))
    worst = 0.0
    for t in range(NF):
        st = {} if debug else None
        my = ref.frame(X[:, t, :], caches, st)
        if debug:
            tst, tcaches = torch_stages(tmodel, X[:, t, :], tcaches)
            for k in tst:
                a, b = tst[k], st.get(k)
                if b is None:
                    print(f"  [缺] {k}")
                    continue
                if a.shape != b.shape:
                    print(f"  {k}: shape {a.shape} vs {b.shape}")
                    continue
                d = np.abs(a - b).max()
                flag = " <-- 发散" if d > 1e-3 and np.abs(a).max() > 1e-6 else ""
                print(f"  {k:14s} max|Δ| {d:.2e}{flag}")
            break
        with torch.no_grad():
            ty, tcaches[0], tcaches[1], tcaches[2] = tmodel(
                torch.from_numpy(X[None, :, t:t + 1, :]), *tcaches)
        d = np.abs(ty[0, :, 0, :].numpy() - my).max()
        rel = d / (np.abs(ty[0, :, 0, :].numpy()).max() + 1e-12)
        worst = max(worst, rel)
        print(f"frame {t}: max|Δ| {d:.2e}  rel {rel:.2e}")
    if not debug:
        print("WORST rel:", worst)
        assert worst < 1e-4, "导出变换校验未过"


if __name__ == "__main__":
    main()
