#!/usr/bin/env python3
"""analyze_acoustic.py — 首批真实声学测试分析（Mac 侧，离线）

输入：/tmp/acoustic/mac/ 下 4ch16k 原始直采（mic4-record16.sh，引擎关）、引擎
处理后 mono16k（voice_mic_shm）、engineB.log（[stats] 行）、marks.txt（各引擎轮
起始行号）。分段全部用「最优连续窗 + 窗外中位底噪」，不依赖录制调度假设。

  T1 几何/麦序：raw_calib 白噪突发 GCC-PHAT TDOA → 24 排列×符号拟合 ULA(d=40mm)
      → 夹角θ + ch→位置映射（模镜像简并）+ 置信度
  T2 真实 AEC：raw_echo_*（AEC-off 直采）回声电平 vs 底噪 → 有效性；
      proc_echo_*（引擎管线）残留 → ERLE（vs 最佳单麦 / 4ch 均值）；ref 通路 stats
  T3 DOA：raw_doa_* ULA GCC-PHAT 导向搜索（映射用 T1 标定序）+ 引擎 doa_latest 对照
  T4 波束增益：raw_beam（左人声 [3,8)s + 右带限白噪全时）
      最佳单麦 vs DSB vs MVDR（STFT-SMI 对角加载）「语音段-纯噪段」SNR

GCC 符号自检：合成已知分数延迟（SYN_CHECK）。
"""
import itertools
import os
import re
import sys
import wave

import numpy as np

D = "/tmp/acoustic/mac"
FS = 16000
C = 343.0
SPACING = 0.04
OUT = []


def log(msg=""):
    print(msg)
    OUT.append(msg)


def read_wav(path):
    w = wave.open(path)
    n, ch = w.getnframes(), w.getnchannels()
    x = np.frombuffer(w.readframes(n), dtype=np.int16)
    w.close()
    x = x.astype(np.float64) / 32768.0
    return (x.reshape(-1, ch).T if ch > 1 else x.reshape(1, -1))


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-12)


def env_bins(x, bin_s=0.5):
    win = int(FS * bin_s)
    m = x.shape[1] // win * win
    return 20 * np.log10(np.sqrt((x[:, :m].reshape(x.shape[0], -1, win) ** 2)
                                 .mean(axis=(0, 2))) + 1e-12), win


def best_window(x, stim_s, guard_bins=2):
    """返回 (t0, t1, floor_db)：时长 stim_s 的最优连续窗，底噪=窗外中位。"""
    env, win = env_bins(x)
    k = max(2, int(round(stim_s / (win / FS))))
    sums = np.convolve(env, np.ones(k) / k, mode="valid")
    i0 = int(np.argmax(sums))
    mask = np.ones(len(env), bool)
    mask[max(0, i0 - guard_bins): i0 + k + guard_bins] = False
    floor = float(np.median(env[mask]))
    return i0 * win / FS, (i0 + k) * win / FS, floor


# ---------------------------------------------------------------- GCC-PHAT --
def gcc_phat(xa, xb, fmax=3400.0, fmin=200.0):
    n = 1
    while n < len(xa) * 2:
        n <<= 1
    A = np.fft.rfft(xa, n)
    B = np.fft.rfft(xb, n)
    X = A * np.conj(B)
    mag = np.abs(X)
    mag[mag < 1e-12] = 1e-12
    f = np.fft.rfftfreq(n, 1.0 / FS)
    band = (f >= fmin) & (f <= fmax)
    Xw = np.zeros_like(X)
    Xw[band] = X[band] / mag[band]
    r = np.fft.irfft(Xw, n)
    r = np.concatenate([r[-64:], r[:64]])
    lags = np.arange(-64, 64)
    k = int(np.argmax(np.abs(r)))
    y0, y1, y2 = r[k - 1], r[k], r[k + 1]
    denom = y0 - 2 * y1 + y2
    frac = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    lag = lags[k] + frac
    peak = abs(r[k]) / (len(r) / 2)
    return lag / FS, peak


def synth_check():
    rng = np.random.default_rng(0)
    s = rng.standard_normal(FS * 2)

    def delay(x, lag_samples):
        f = np.fft.rfftfreq(len(x), 1.0)
        return np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * f * lag_samples), len(x))

    a = delay(s, 2.5)
    b = delay(s, 0.0)
    t, _ = gcc_phat(a[:FS], b[:FS])
    assert abs(t - 2.5 / FS) < 0.2 / FS, f"GCC 符号自检失败: {t*FS:.2f} 样本"
    return True


# ------------------------------------------------------- T1 排列拟合 --------
def fit_ula_permutation(T):
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
        results.append((perm, slope * C, float(np.sqrt(np.mean(np.square(resid))))))
    results.sort(key=lambda r: r[2])
    return results


def t1_geometry():
    log("\n" + "=" * 74)
    log("T1 几何与麦序标定（Mac 左 speaker 白噪突发 0.5s×6，raw_calib.wav 直采）")
    log("=" * 74)
    x4 = read_wav(f"{D}/raw_calib.wav")
    t0, t1, floor = best_window(x4, 4.0)
    log(f"最强 4s 窗 [{t0:.2f},{t1:.2f}]s，窗外中位底噪 {floor:.1f} dBFS")
    env, win = env_bins(x4)
    on = env > floor + 8
    segs = []
    st = None
    for i, v in enumerate(on):
        if v and st is None:
            st = i
        elif not v and st is not None:
            if (i - st) * (win / FS) >= 0.35:
                segs.append((st * win / FS, i * win / FS))
            st = None
    if st is not None and (len(on) - st) * (win / FS) >= 0.35:
        segs.append((st * win / FS, len(on) * win / FS))
    # 合并贴近段（同一个 0.5s burst 的抖动）
    merged = []
    for a, b in segs:
        if merged and a - merged[-1][1] < 0.15:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    segs = merged
    log(f"能量突发段（≥0.35s）: " + " ".join(f"[{a:.2f},{b:.2f}]" for a, b in segs))
    if len(segs) < 3:
        log("!! 突发段不足 3 个，T1 无法下结论")
        return None, None, None

    per_burst_T = []
    for a, b in segs:
        seg = x4[:, int((a + 0.06) * FS): int((b - 0.06) * FS)]
        T = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                if i < j:
                    tau, _ = gcc_phat(seg[i], seg[j])
                    T[i, j], T[j, i] = tau, -tau
        per_burst_T.append(T)
    Tall = np.mean(per_burst_T, axis=0)
    jit = np.std(per_burst_T, axis=0)
    log(f"\n平均 TDOA 矩阵 τ_ij（样本@16k，τ_ij=t_i−t_j，物理上限 |τ|=3d/c=5.60 样本）:")
    log("       ch0     ch1     ch2     ch3")
    for i in range(4):
        log(f"ch{i}  " + "".join(f"{Tall[i,j]*FS:+7.2f}" for j in range(4)))
    log(f"burst 间 τ 标准差最大 {jit.max()*FS:.2f} 样本（{len(per_burst_T)} 段）")

    res = fit_ula_permutation(Tall)
    log("\nULA(d=40mm) 排列拟合（残差最小前 4；排列=位置0..3 上的通道号）：")
    log("  排列            sinθ      θ(°)    残差RMS(样本)")
    for perm, s, r in res[:4]:
        th = np.degrees(np.arcsin(np.clip(s, -1, 1))) if abs(s) <= 1 else float("nan")
        log(f"  {list(perm)}   {s:+7.3f}  {th:+7.1f}  {r*FS:8.3f}")

    best_perm, best_s, best_r = res[0]
    rev = tuple(best_perm[3 - k] for k in range(4))
    rev_r = [e for e in res if e[0] == rev][0][2]
    theta = np.degrees(np.arcsin(np.clip(best_s, -1, 1))) if abs(best_s) <= 1 else None
    mapping = {k: best_perm[k] for k in range(4)}
    log(f"\n结论：位置k ← 通道ch 映射 {mapping}")
    log(f"  镜像排列 {list(rev)} 残差 {rev_r*FS:.3f}（简并：两端谁为位置 0 不可分）")
    if theta is not None:
        log(f"  阵轴与源方向夹角 θ = {theta:+.1f}°（broadside=0°，端射=±90°，前后模糊）")
        log(f"  ⇒ |sinθ|={abs(best_s):.3f}，源{'沿阵轴（端射附近）' if abs(best_s)>0.8 else '斜向' if abs(best_s)>0.3 else '近 broadside'}")
    pos = {ch: SPACING * k for ch, k in zip(best_perm, range(4))}
    log("\n逐对独立 sinθ=τ·c/(Δx)（最佳排列；|sinθ|>1 即不可行=数据差）：")
    for i in range(4):
        for j in range(i + 1, 4):
            dx = (pos[i] - pos[j]) / SPACING
            s_ij = Tall[i, j] * C / (dx * SPACING)
            flag = "" if abs(s_ij) <= 1.05 else "  ←|sinθ|>1"
            log(f"  ch{i}-ch{j}: Δ{dx:+.0f}d τ={Tall[i,j]*FS:+6.2f}样本 sinθ={s_ij:+7.3f}{flag}")
    return best_perm, theta, res


# ------------------------------------------------------------- T2 AEC ------
def parse_stats(path):
    rows = []
    pat = re.compile(
        r"\[stats\] t=([\d.]+)s captured_samples=(\d+) aec_blocks=(\d+) fifo_written=(\d+) "
        r"fifo_dropped=(\d+) fifo_read=(\d+) ref_samples=(\d+) doa_latest=([-\d.]+) rtf=([\d.]+) "
        r"tag=\[([^\]]+)\]% overrun=(\d+) ring_ovf=(\d+) ref_ovf=(\d+) ref_padded=(\d+) "
        r"ref_discarded=(\d+)")
    for line in open(path):
        m = pat.search(line)
        if m:
            rows.append(dict(t=float(m.group(1)), cap=int(m.group(2)), aec=int(m.group(3)),
                             fw=int(m.group(4)), fd=int(m.group(5)), fr=int(m.group(6)),
                             ref=int(m.group(7)), doa=float(m.group(8)), rtf=float(m.group(9)),
                             tags=m.group(10), ovf=int(m.group(11)), rovf=int(m.group(12)),
                             refovf=int(m.group(13)), refpad=int(m.group(14)),
                             refdis=int(m.group(15))))
    return rows


def t2_aec():
    log("\n" + "=" * 74)
    log("T2 真实声学 AEC（自播 -26dBFS×8s；off=引擎关直采 / on=引擎管线 AEC）")
    log("=" * 74)
    amb_raw = read_wav(f"{D}/raw_ambient.wav")
    _, _, floor_amb = best_window(amb_raw, 0.5)
    fr = [rms_db(c) for c in amb_raw]
    log(f"原始底噪(raw_ambient 10s): ch 电平 " + " ".join(f"{v:.1f}" for v in fr)
        + f" dBFS；中位 {floor_amb:.1f}")
    amb_p = read_wav(f"{D}/proc_ambient.wav")
    p_floor = rms_db(amb_p[0])
    log(f"管线底噪(proc_ambient 10s, BF+AEC 输出): {p_floor:.1f} dBFS")

    for tag in ("white", "speech"):
        raw = read_wav(f"{D}/raw_echo_{tag}.wav")
        t0, t1, fl = best_window(raw, 6.0)
        seg = raw[:, int((t0 + 0.5) * FS): int((t1 - 0.5) * FS)]
        lv = [rms_db(c) for c in seg]
        snr = [l - floor_amb for l in lv]
        best_ch = int(np.argmax(lv))
        log(f"\n[{tag}] 直采回声窗 [{t0:.2f},{t1:.2f}]s（窗外底噪 {fl:.1f}）")
        log("  各麦回声电平: " + " ".join(f"ch{i}:{v:.1f}" for i, v in enumerate(lv))
            + f" dBFS；峰值 {20*np.log10(np.abs(seg).max()):.1f} dBFS（>-6 即削波）")
        log("  回声 vs 底噪:  " + " ".join(f"ch{i}:{v:+.1f}dB" for i, v in enumerate(snr))
            + f"；最佳 {max(snr):+.1f}dB（验收门槛 ≥10dB）"
            + ("  ✔ 有效" if max(snr) >= 10 else "  ✘ 不足"))

        proc = read_wav(f"{D}/proc_echo_{tag}.wav")
        tp0, tp1, pfl = best_window(proc, 4.0)
        segp = proc[0, int((tp0 + 0.5) * FS): int((tp1 - 0.5) * FS)]
        resid = rms_db(segp)
        log(f"  AEC 后残留窗 [{tp0:.2f},{tp1:.2f}]s: {resid:.1f} dBFS"
            f"（管线底噪 {p_floor:.1f}，残留{'高于' if resid > p_floor else '不高于'}底噪"
            f" {abs(resid-p_floor):.1f}dB）")
        log(f"  ERLE(vs 最佳单麦 ch{best_ch}) = {lv[best_ch]:.1f} − ({resid:.1f})"
            f" = {lv[best_ch]-resid:.1f} dB")
        log(f"  ERLE(vs 4ch 均值)      = {np.mean(lv):.1f} − ({resid:.1f})"
            f" = {np.mean(lv)-resid:.1f} dB")

    # 引擎 ref 通路一致性
    log("\n引擎 [stats] ref 通路（ref_samples 按 16k 计；播放 8s 应 ≈128000）：")
    rows = parse_stats(f"{D}/engineB.log")
    marks = {}
    mp = f"{D}/marks.txt"
    if os.path.exists(mp):
        for ln in open(mp):
            k, v = ln.split()
            marks[k] = float(v)
    for name in ("proc_echo_white", "proc_echo_speech", "proc_doa_both",
                 "proc_doa_Lonly", "proc_doa_Ronly"):
        if name not in marks:
            continue
        a = marks[name]
        seg = [r for r in rows if a <= r["t"] <= a + 20]
        if not seg:
            continue
        dref = seg[-1]["ref"] - seg[0]["ref"]
        dcap = seg[-1]["cap"] - seg[0]["cap"]
        daec = seg[-1]["aec"] - seg[0]["aec"]
        doas = [r["doa"] for r in seg]
        log(f"  {name:18s} Δt={seg[-1]['t']-seg[0]['t']:4.0f}s Δref={dref:7d}"
            f"({dref/FS:5.2f}s@16k) Δcap={dcap/FS:5.1f}s Δaec={daec:5d}blk"
            f" ovf={seg[-1]['ovf']-seg[0]['ovf']}/{seg[-1]['rovf']-seg[0]['rovf']}"
            f" doa_latest中位={np.median(doas):+6.1f}°")


# ------------------------------------------------------------- T3 DOA ------
def ula_doa_search(x4, pos_map, fmax=1400.0):
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


def t3_doa(perm):
    log("\n" + "=" * 74)
    log("T3 双源 DOA（离线 4ch，ULA GCC-PHAT 导向搜索，fmax=1.4k=引擎限带，映射用 T1 序）")
    log("=" * 74)
    pos_map = {ch: SPACING * k for ch, k in zip(perm, range(4))}
    log(f"位置k←通道ch: { {k: perm[k] for k in range(4)} }（模镜像）")
    res = {}
    for tag in ("Lonly", "Ronly", "both"):
        x4 = read_wav(f"{D}/raw_doa_{tag}.wav")
        t0, t1, fl = best_window(x4, 5.0)
        seg = x4[:, int((t0 + 0.3) * FS): int(t0 + 5.0 * FS)]
        th = ula_doa_search(seg, pos_map)
        lv = rms_db(seg.mean(axis=0))
        log(f"  doa_{tag:6s} 窗[{t0:.2f}+5s] 底噪{fl:.1f} 段电平{lv:.1f}  离线DOA θ={th:+6.1f}°")
        res[tag] = th
    dLR = res["Lonly"] - res["Ronly"]
    log(f"  左源 vs 右源 DOA 差 = {dLR:+.1f}°（几何预期：同方向不同距离 → 差应很小）")
    return res


# ------------------------------------------------------------- T4 波束 -----
def stft(x, nfft=512, hop=128):
    win = np.hanning(nfft)
    n = (len(x) - nfft) // hop + 1
    out = np.empty((nfft // 2 + 1, n), dtype=np.complex128)
    for i in range(n):
        out[:, i] = np.fft.rfft(x[i * hop: i * hop + nfft] * win)
    return out


def istft(X, nfft=512, hop=128, length=None):
    win = np.hanning(nfft)
    n = X.shape[1]
    y = np.zeros(nfft + (n - 1) * hop)
    wsum = np.zeros_like(y)
    for i in range(n):
        y[i * hop: i * hop + nfft] += np.fft.irfft(X[:, i], nfft) * win
        wsum[i * hop: i * hop + nfft] += win**2
    y = y / np.maximum(wsum, 1e-9)
    return y[:length] if length else y


def beam_t4(perm):
    log("\n" + "=" * 74)
    log("T4 波束增益（raw_beam：左 speaker 人声[3,8)s + 右 speaker 带限白噪全 10s）")
    log("=" * 74)
    x4 = read_wav(f"{D}/raw_beam.wav")
    t0, t1, fl = best_window(x4, 8.0)
    log(f"播放窗估计始于 {t0:.2f}s（底噪 {fl:.1f}）")
    na, nb = int((t0 + 0.2) * FS), int((t0 + 2.8) * FS)      # 纯噪段1
    sa, sb = int((t0 + 3.2) * FS), int((t0 + 7.8) * FS)      # 语音+噪
    n2a, n2b = int((t0 + 8.2) * FS), int((t0 + 9.8) * FS)    # 纯噪段2
    pos_map = {ch: SPACING * k for ch, k in zip(perm, range(4))}
    th_sp = ula_doa_search(x4[:, sa:sb], pos_map)
    th_ns = ula_doa_search(x4[:, na:nb], pos_map)
    log(f"  语音段主导 DOA θ_s={th_sp:+.1f}°，噪声段主导 DOA θ_n={th_ns:+.1f}°（DSB/MVDR 导向 θ_s）")

    def seg_power(y):
        pn = (np.mean(y[na:nb] ** 2) + np.mean(y[n2a:n2b] ** 2)) / 2
        ps = np.mean(y[sa:sb] ** 2)
        snr = 10 * np.log10(max(ps - pn, 1e-12) / max(pn, 1e-12))
        return 10 * np.log10(ps + 1e-12), 10 * np.log10(pn + 1e-12), snr

    per_ch = [seg_power(x4[c]) for c in range(4)]
    best = int(np.argmax([s for _, _, s in per_ch]))
    for c in range(4):
        log(f"  单麦 ch{c}: 语音段 {per_ch[c][0]:6.1f}  噪声段 {per_ch[c][1]:6.1f}"
            f"  SNR {per_ch[c][2]:+5.1f} dB")
    log(f"  ⇒ 最佳单麦 ch{best}，SNR {per_ch[best][2]:+.1f} dB")

    pos = np.zeros(4)
    for ch, k in zip(perm, range(4)):
        pos[ch] = SPACING * k
    nfft, hop = 512, 128
    Xs = [stft(x4[c]) for c in range(4)]
    f = np.fft.rfftfreq(nfft, 1.0 / FS)
    A = np.exp(-2j * np.pi * f[:, None] * (pos[None, :] * np.sin(np.radians(th_sp)) / C))
    length = x4.shape[1]

    Y = np.zeros_like(Xs[0])
    for b in range(Xs[0].shape[0]):
        Y[b] = sum(np.conj(A[b, m]) / 4 * Xs[m][b] for m in range(4))
    ps, pn, snr_dsb = seg_power(istft(Y, nfft, hop, length))
    log(f"  DSB : 语音段 {ps:6.1f}  噪声段 {pn:6.1f}  SNR {snr_dsb:+5.1f} dB"
        f"  → 阵列增益 {snr_dsb - per_ch[best][2]:+.1f} dB")

    Y = np.zeros_like(Xs[0])
    nb_frames = Xs[0].shape[1]
    for b in range(Xs[0].shape[0]):
        Xb = np.array([Xs[m][b] for m in range(4)])
        R = (Xb @ Xb.conj().T) / nb_frames
        R += 1e-2 * np.trace(R).real / 4 * np.eye(4)
        a = A[b]
        Ria = np.linalg.solve(R, a)
        w = Ria / (a.conj() @ Ria)
        Y[b] = w.conj() @ Xb
    ps, pn, snr_mv = seg_power(istft(Y, nfft, hop, length))
    log(f"  MVDR: 语音段 {ps:6.1f}  噪声段 {pn:6.1f}  SNR {snr_mv:+5.1f} dB"
        f"  → 阵列增益 {snr_mv - per_ch[best][2]:+.1f} dB（SMI 全段协方差, 负载 10⁻²）")
    return dict(best=best, snr_single=per_ch[best][2], snr_dsb=snr_dsb,
                snr_mv=snr_mv, th_sp=th_sp, th_ns=th_ns)


def main():
    assert synth_check()
    log("[SYN_CHECK] GCC-PHAT 符号自检通过（合成 +2.5 样本延迟正确恢复）")
    perm, theta, res = t1_geometry()
    if perm is not None:
        t2 = t2_aec()
        t3 = t3_doa(perm)
        t4 = beam_t4(perm)
    with open(f"{D}/analysis.txt", "w") as fp:
        fp.write("\n".join(OUT) + "\n")
    print(f"\n[written] {D}/analysis.txt")


if __name__ == "__main__":
    sys.exit(main())
