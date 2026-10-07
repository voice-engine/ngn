#!/usr/bin/env python3
"""gen_calib2.py — 平滑过渡白噪突发标定素材（几何/DOA 用，2026-10-06）

结构：1.0s 突发 + 0.6s 间隙 ×6（周期 1.6s，analyze_geometry.py 的突发链
提取与逐突发配对都以此周期为锚）。边沿 120ms 升余弦（cos²）——无咔哒
瞬态，包络检测与 GCC-PHAT 都干净。全带宽白噪（相关峰最锐）。

双源：calib2_L.wav（仅左声道）/ calib2_R.wav（仅右声道），独立种子。
RMS -26dBFS 与首批素材一致；Mac 播放音量 55% 校准点（麦上 SNR +12.6dB、
峰值 -17dBFS 无削波；65% 时 +23.6/-13.4，45% 时 +10.9/-23.0）。
"""
import os
import sys
import wave

import numpy as np

FS = 48000
OUT = "/tmp/acoustic"
RMS = 0.05
N_BURST, ON_S, GAP_S = 6, 1.0, 0.6
RAMP_MS = 120


def cos2ramp(x, ms=RAMP_MS):
    n = int(FS * ms / 1000)
    w = np.sin(np.linspace(0, np.pi / 2, n)) ** 2
    x = x.copy()
    x[:n] *= w
    x[-n:] *= w[::-1]
    return x


def burst_train(seed):
    r = np.random.default_rng(seed)
    parts = []
    for _ in range(N_BURST):
        seg = r.standard_normal(int(ON_S * FS))
        seg *= RMS / np.sqrt(np.mean(seg ** 2) + 1e-20)
        parts += [cos2ramp(seg), np.zeros(int(GAP_S * FS))]
    return np.concatenate(parts)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, ch, seed in (("calib2_L.wav", 0, 1000), ("calib2_R.wav", 1, 1001)):
        d = burst_train(seed)
        inter = np.zeros(len(d) * 2)
        inter[ch::2] = d
        w = wave.open(f"{OUT}/{name}", "wb")
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes((np.clip(inter, -0.98, 0.98) * 32767).astype(np.int16).tobytes())
        w.close()
        print(f"{name}: {len(d)/FS:.1f}s rms={20*np.log10(np.sqrt((d**2).mean())):+.1f}dBFS "
              f"peak={20*np.log10(np.abs(inter).max()):+.1f}dBFS")


if __name__ == "__main__":
    sys.exit(main())
