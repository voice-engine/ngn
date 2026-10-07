#!/usr/bin/env python3
"""analyze_geometry.py — 麦克风位置标定与 DOA 准确度（v2，2026-10-06 重写）

上一版三个教训：
  ① GCC 带宽 6800Hz 超过 ULA d=40mm 空间混叠上限 c/2d≈4.29kHz（端射附近更低）
    → τ 被混叠污染、逐对符号不一致 → 全部限带 ≤3400Hz；
  ② 能量突发检测混入环境假段 → 源素材↔录音的突发配对错位 → 时延全垃圾
    → 改用 1.6s 周期链提取真突发（素材结构 1s on/0.6s off）；
  ③ 逐突发独立大窗(max_lag±2s)互相关 → 假峰 → 先整列车相关定 T0，
    再 ±40 样本小窗逐道时延（物理上限 |τ|=3d/c=5.60 样本）。

方法（用户指定）：
  1) 麦间 GCC-PHAT TDOA（L/R 双源独立）→ ULA 排列拟合（残差最小前 3）
  2) 源信号↔麦采集互相关 → 每道传播时延差分 Δd_i（与麦间 TDOA 互验；
     L/R 两源差分模式相反 → 解镜像简并）
  3) DOA：ULA 导向搜索（fmax=3k 限带）逐突发 + L/R/both 材料；重复性=std

自检：合成 θ=30° 4ch（cycles/sample 单位分数延迟）全链路验证。
  注：analyze_acoustic.py 旧 SYN_CHECK 的 delay() 用 Hz×样本=2.5 秒，是坏的；
  本文件用 cycles/sample，实证 gcc_phat(a,b)=t_a−t_b 子样本精度 ±0.1。
"""
import itertools
import sys
import wave

import numpy as np
from scipy.signal import resample_poly

D = "/tmp/acoustic/mac"
MAT = "/tmp/acoustic"
FS = 16000
C = 343.0
SPACING = 0.04
GCC_FMIN, GCC_FMAX = 250.0, 3400.0   # ≤ c/2d=4.29k 混叠安全，且避开 Mac 喇叭低频滚落
OUT = []
BURST_ON, BURST_GAP = 1.0, 0.6       # 素材：1s 突发 + 0.6s 间隙，周期 1.6s


def log(msg=""):
    print(msg)
    OUT.append(msg)


def read_wav(path):
    w = wave.open(path)
    n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
    x = np.frombuffer(w.readframes(n), dtype=np.int16)
    w.close()
    x = x.astype(np.float64) / 32768.0
    x = x.reshape(-1, ch).T if ch > 1 else x.reshape(1, -1)
    return (x, sr) if sr != FS else (x, FS)


def read_wav16(path):
    x, sr = read_wav(path)
    if sr != FS:
        x = resample_poly(x, 1, sr // FS, axis=1)
    return x


def gcc_phat(xa, xb, fmin=GCC_FMIN, fmax=GCC_FMAX, max_lag=80):
    """PHAT 白化互相关 + 抛物线内插。返回 (lag_samples, peak)。约定 lag=t_a−t_b。"""
    n = 1
    while n < len(xa) * 2:
        n <<= 1
    A, B = np.fft.rfft(xa, n), np.fft.rfft(xb, n)
    X = A * np.conj(B)
    mag = np.abs(X)
    mag[mag < 1e-12] = 1e-12
    f = np.fft.rfftfreq(n, 1.0 / FS)
    band = (f >= fmin) & (f <= fmax)
    Xw = np.zeros_like(X)
    Xw[band] = X[band] / mag[band]
    r = np.fft.irfft(Xw, n)
    r = np.concatenate([r[-max_lag:], r[: max_lag + 1]])
    lags = np.arange(-max_lag, max_lag + 1)
    k = int(np.argmax(np.abs(r)))
    if k == 0 or k == len(r) - 1:
        return float(lags[k]), 0.0
    y0, y1, y2 = r[k - 1], r[k], r[k + 1]
    den = y0 - 2 * y1 + y2
    frac = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
    return float(lags[k] + frac), float(abs(y1))


def env_bins(x, bin_s=0.05):
    win = int(FS * bin_s)
    m = x.shape[1] // win * win
    return (20 * np.log10(np.sqrt((x[:, :m].reshape(x.shape[0], -1, win) ** 2)
                                  .mean(axis=(0, 2))) + 1e-12)), bin_s


def burst_chain(x4):
    """从 4ch 能量包络提取 1.6s 周期突发链。返回 (onsets_list, floor_db)。
    只保留相邻 onset 差 ∈ 1.6±0.25s 的最长链 → 环境假段全剔除。"""
    env, bs = env_bins(x4)
    floor = np.percentile(env, 25)
    on = env > floor + 6
    # 粗 onset：on 状态的上升沿（0.05s 粒度）
    edges = []
    prev = False
    for i, v in enumerate(on):
        if v and not prev:
            edges.append(i * bs)
        prev = v
    if not edges:
        return [], floor
    period = BURST_ON + BURST_GAP
    # 最长 1.6s 周期链（动态规划式贪心）
    best_chain, best_len = [], 0
    for s in range(len(edges)):
        chain, last = [edges[s]], edges[s]
        for e in edges[s + 1:]:
            if period - 0.25 <= e - last <= period + 0.25:
                chain.append(e)
                last = e
        if len(chain) > best_len:
            best_chain, best_len = chain, len(chain)
    return best_chain, floor


def fit_ula(T):
    """T: 4x4 样本 TDOA（τ_ij=t_i−t_j）。返回按残差排序 (perm, sinθ, resid_samples)。"""
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    results = []
    for perm in itertools.permutations(range(4)):
        pos = {ch: SPACING * k for ch, k in zip(perm, range(4))}
        num = den = 0.0
        for i, j in pairs:
            dx = pos[i] - pos[j]
            num += T[i, j] * dx
            den += dx * dx
        slope = num / den
        resid = [T[i, j] - (pos[i] - pos[j]) * slope for i, j in pairs]
        results.append((perm, slope * C / FS,
                        float(np.sqrt(np.mean(np.square(resid))))))
    results.sort(key=lambda r: r[2])
    return results


def ula_doa_search(x4, pos_map, fmax=3000.0):
    N = min(x4.shape[1], int(6.4 * FS))
    B = 8192
    f = np.fft.rfftfreq(B, 1.0 / FS)
    band = (f >= 150) & (f <= fmax)
    chs = sorted(pos_map)
    pos = np.array([pos_map[c] for c in chs])
    nb = N // B
    specs = []
    for c in chs:
        S = 0
        for i in range(nb):
            S = S + np.fft.rfft(x4[c][i * B:(i + 1) * B] * np.hanning(B))
        specs.append(S / max(nb, 1))
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    thetas = np.arange(-90, 90.01, 0.5)
    scores = np.zeros(len(thetas))
    for ti, th in enumerate(thetas):
        s = np.sin(np.radians(th))
        for i, j in pairs:
            X = specs[i] * np.conj(specs[j])
            mag = np.abs(X)
            w = np.zeros_like(mag)
            w[band] = 1.0 / (mag[band] + 1e-12)
            lag = (pos[i] - pos[j]) * s / C
            scores[ti] += np.real(np.sum(w * X * np.exp(2j * np.pi * f * lag))) / band.sum()
    k = int(np.argmax(scores))
    dth = 0.0
    if 0 < k < len(thetas) - 1:
        y0, y1, y2 = scores[k - 1], scores[k], scores[k + 1]
        dth = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * 0.5
    return thetas[k] + dth


def synth_check():
    rng = np.random.default_rng(11)
    n = int(2.0 * FS)
    src = rng.standard_normal(n)
    f = np.fft.rfftfreq(n)  # cycles/sample
    x4 = []
    for ch in range(4):
        lag = ch * SPACING * np.sin(np.radians(30.0)) / C * FS
        x4.append(np.fft.irfft(np.fft.rfft(src) * np.exp(-2j * np.pi * f * lag), n))
    x4 = np.array(x4)
    T = np.zeros((4, 4))
    for i in range(4):
        for j in range(i + 1, 4):
            lag, pk = gcc_phat(x4[i], x4[j])
            assert pk > 0.05, f"SYN gcc 峰值过低 {pk}"
            T[i, j], T[j, i] = lag, -lag
    perm, s, r = fit_ula(T)[0]
    assert list(perm) in ([0, 1, 2, 3], [3, 2, 1, 0]), f"SYN 排列错 {perm}"
    assert abs(abs(s) - 0.5) < 0.05, f"SYN sinθ 偏差 {s:+.3f} (期望 ±0.500)"
    assert r < 0.15, f"SYN 残差 {r:.3f}"
    pos_map = {ch: SPACING * k for ch, k in zip(perm, range(4))}
    th = ula_doa_search(x4, pos_map)
    assert abs(abs(th) - 30.0) < 3.0, f"SYN 导向搜索 θ={th:+.1f}° (期望 ±30°)"
    dd = [gcc_phat(x4[ch], src, max_lag=200)[0] for ch in range(4)]
    assert dd[3] > dd[0] and abs(dd[3] - dd[0] - 3 * 0.5 * SPACING / C * FS) < 0.5, \
        f"SYN 源↔麦差分错 {[round(v, 2) for v in dd]}"
    return True


def analyze_source(tag, mat_wav):
    log("\n" + "=" * 74)
    log(f"源 {tag}（素材 {mat_wav}，平滑白噪突发 1s/0.6s ×6）")
    log("=" * 74)
    x4 = read_wav16(f"{D}/raw_{tag}.wav")
    onsets, floor = burst_chain(x4)
    log(f"底噪 {floor:.1f} dBFS；1.6s 周期突发链 onset: "
        + " ".join(f"{t:.2f}" for t in onsets))
    if len(onsets) < 4:
        log("!! 突发链 <4，该源无结论")
        return None

    # ① 麦间 TDOA（逐突发，取突发中段 0.7s）
    mats = []
    for a in onsets:
        seg = x4[:, int((a + 0.15) * FS): int((a + 0.85) * FS)]
        T = np.zeros((4, 4))
        ok = True
        for i in range(4):
            for j in range(i + 1, 4):
                lag, pk = gcc_phat(seg[i], seg[j])
                if pk < 0.05:
                    ok = False
                T[i, j], T[j, i] = lag, -lag
        if ok:
            mats.append(T)
    if not mats:
        log("!! 无有效 TDOA 突发")
        return None
    T = np.mean(mats, axis=0)
    std = np.std(mats, axis=0)
    log(f"麦间 TDOA τ_ij（样本@16k，τ_ij=t_i−t_j；|τ|物理上限=3d/c=5.60；{len(mats)} 突发均值）：")
    log("       ch0     ch1     ch2     ch3")
    for i in range(4):
        log(f"ch{i}  " + "".join(f"{T[i, j]:+7.2f}" for j in range(4)))
    log(f"burst 间 τ std 最大 {std.max():.2f} 样本")

    # ② 源↔麦互相关：整列车定 T0（ch0），再逐道 ±40 样本小窗
    ref_ch = 0 if "L" in mat_wav else 1                 # calib2_L→左声道，calib2_R→右声道
    ref = read_wav16(f"{MAT}/{mat_wav}")[ref_ch]        # 1D 素材信号
    t0_full, pk0 = gcc_phat(x4[0], ref, max_lag=int(2.5 * FS))
    log(f"\n整列车互相关定 T0：lag={t0_full:.1f} 样本（{t0_full/FS*1000:.0f}ms，peak={pk0:.2f}）")
    per_burst = []
    for bi, a in enumerate(onsets):
        rs = int((bi * (BURST_ON + BURST_GAP) + 0.15) * FS)
        re_ = int((bi * (BURST_ON + BURST_GAP) + 0.85) * FS)
        row = []
        for ch in range(4):
            ms = int(t0_full + rs - 40)          # 麦窗以 T0 对齐素材突发，前后 ±40
            mseg = x4[ch, ms: ms + (re_ - rs) + 80]
            if mseg.shape[0] < (re_ - rs):
                row.append(np.nan)
                continue
            lag, pk = gcc_phat(mseg, ref[rs:re_], max_lag=80)
            row.append(lag + (t0_full - 40) if pk > 0.05 else np.nan)
            # lag 已相对 mseg 起点；d_i = T0 + τ_i（τ_i 为麦 i 相对 ch0 的额外时延）
        per_burst.append(row)
    per_burst = np.array(per_burst)
    dd_raw = np.nanmean(per_burst, axis=0)
    if np.isnan(dd_raw).any():
        log("  [warn] 部分道互相关峰值不足")
    dd = dd_raw - np.nanmin(dd_raw)
    log(f"源↔麦时延 d_i（相对最远道，样本；{np.sum(~np.isnan(per_burst[:, 0]))} 突发均值）：")
    log("  d_i:   " + " ".join(f"ch{i}:{dd_raw[i]:+7.2f}" for i in range(4)))
    log("  Δd_i:  " + " ".join(f"ch{i}:{dd[i]:+7.2f}" for i in range(4)))
    log("  折声程差(mm): " + " ".join(f"ch{i}:{dd[i]*1000*C/FS:+6.1f}" for i in range(4)))
    # 与麦间 TDOA 互验：Δd_i − Δd_j 应 ≈ τ_ij − τ_min... 直接对比逐对
    tau_from_ref = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            tau_from_ref[i, j] = dd_raw[i] - dd_raw[j]
    agree = np.abs(tau_from_ref - T)[np.triu_indices(4, 1)]
    log(f"  与麦间 TDOA 逐对一致性: RMS 差 {np.sqrt((agree**2).mean()):.2f} 样本"
        f"（≤0.5 为互验通过）")

    # ③ ULA 排列拟合
    res = fit_ula(T)
    log("\nULA(d=40mm) 排列拟合（残差最小前 3；排列=位置0..3 上的通道号）：")
    log("  排列            sinθ      θ(°)    残差RMS(样本)")
    for perm, s, r in res[:3]:
        th = np.degrees(np.arcsin(np.clip(s, -1, 1))) if abs(s) <= 1 else float("nan")
        log(f"  {list(perm)}   {s:+7.3f}  {th:+7.1f}  {r:8.3f}")
    perm, s, r = res[0]
    # 镜像解被：L/R 源差分梯度方向相反 → 两次分析后统一（main 里对比）
    pos = {ch: SPACING * k for ch, k in zip(perm, range(4))}
    chs = sorted(pos)
    a_ = np.array([pos[c] / SPACING for c in chs])
    a_ = a_ - a_.mean()
    b_ = (dd - np.nanmean(dd)) / (np.nanmax(dd) - np.nanmin(dd) + 1e-9)
    corr_dir = float(np.dot(a_, b_) / (np.linalg.norm(a_) * np.linalg.norm(b_) + 1e-12))
    log(f"\n拟合排列 {list(perm)} sinθ={s:+.3f}；差分梯度与该排列位置序相关 = {corr_dir:+.2f}")
    return dict(perm=perm, sin=s, resid=r, T=T, dd=dd_raw, corr_dir=corr_dir,
                onsets=onsets)


def main():
    assert synth_check()
    log("[SYN_CHECK] 通过：θ=30° 合成 → 排列/sinθ=±0.500±0.05/残差<0.15/导向±30°/互相关差分±0.5样本")
    gL = analyze_source("calib", "calib2_L.wav")
    gR = analyze_source("calib_R", "calib2_R.wav")

    # 统一物理朝向：L 源梯度为正的排列方向定义为位置 0→3（L 源在位置 0 一侧）
    if gL and gR:
        same_fit = gL["perm"] == gR["perm"] or gL["perm"] == tuple(gR["perm"][3 - k] for k in range(4))
        log("\n" + "=" * 74)
        log(f"双源一致性：排列拟合 {'一致(含镜像) ✔' if same_fit else '不一致 ✘'}"
            f"；梯度相关 L={gL['corr_dir']:+.2f} R={gR['corr_dir']:+.2f}"
            f"（符号应相反）")
        log("=" * 74)
        base = gL["perm"] if gL["corr_dir"] >= 0 else tuple(gL["perm"][3 - k] for k in range(4))
        sign = 1.0 if gL["corr_dir"] >= 0 else -1.0
        log(f"统一朝向（L 源在位置0侧）：位置k←通道ch { {k: base[k] for k in range(4)} }")
        log(f"  L 源 sinθ={sign*gL['sin']:+.3f}（{np.degrees(np.arcsin(np.clip(sign*gL['sin'],-1,1))):+.1f}°）"
            f"  R 源 sinθ={sign*gR['sin']:+.3f}（{np.degrees(np.arcsin(np.clip(sign*gR['sin'],-1,1))):+.1f}°）")

        pos_map = {ch: SPACING * k for ch, k in zip(base, range(4))}
        log("\nDOA 准确度（导向搜索 fmax=3k，统一朝向映射）：")
        for tag in ("calib", "calib_R"):
            x4 = read_wav16(f"{D}/raw_{tag}.wav")
            onsets, _ = burst_chain(x4)
            ths = []
            for a in onsets:
                seg = x4[:, int((a + 0.15) * FS): int((a + 0.85) * FS)]
                if seg.shape[1] > FS // 2:
                    ths.append(ula_doa_search(seg, pos_map))
            ths = np.array(ths)
            log(f"  {tag:10s} 逐突发: 中位 {np.median(ths):+6.1f}° std {ths.std():4.1f}° "
                f"范围 [{ths.min():+.1f},{ths.max():+.1f}] ({len(ths)}段)")
        for tag in ("doa_Lonly", "doa_Ronly", "doa_both"):
            try:
                x4 = read_wav16(f"{D}/raw_{tag}.wav")
            except FileNotFoundError:
                continue
            env, bs = env_bins(x4, 0.25)
            k = max(2, int(6.0 / bs))
            sums = np.convolve(env, np.ones(k) / k, "valid")
            t0 = int(np.argmax(sums)) * bs
            seg = x4[:, int((t0 + 0.2) * FS): int((t0 + 6.0) * FS)]
            th = ula_doa_search(seg, pos_map)
            lv = 20 * np.log10(np.sqrt((seg ** 2).mean()) + 1e-12)
            log(f"  {tag:10s} 窗[{t0:.1f}+6s] 电平{lv:.1f}dBFS  θ={th:+6.1f}°")

    with open(f"{D}/geometry.txt", "w") as fp:
        fp.write("\n".join(OUT) + "\n")
    print(f"\n[written] {D}/geometry.txt")


if __name__ == "__main__":
    sys.exit(main())
