#!/usr/bin/env python3
# analyze_system.py — 步骤2 验证分析（Mac 侧，numpy/scipy）
# 用法: analyze_system.py <dir>   # dir 含 out_A.wav out_B.wav sim4.wav engine_A.log engine_B.log
#
# 校验项（对应任务验收 ①②③）：
#  ① out.wav 16k mono、时长≈20s、与 sim4 ch0 相关>0.3、语音 RMS vs 静音 >20dB
#  ② AEC 生效：A(灌 far48) vs B(灌静音) 在“仅回声”窗（sim [4,6]s 与 [0,1]s）的
#     输出残留能量差 ≥6dB
#  ③ stats: fifo_read≈48000×8、ref_samples≈16000×8、doa_latest∈[15,45]、rtf<1
import os, re, sys, wave
import numpy as np
from scipy.signal import fftconvolve

FS = 16000
SPK_WIN = [(1, 4), (6, 8), (10, 13), (15, 18)]      # 近端语音窗
QUIET_WIN = [(8.8, 10), (13.8, 15), (18.8, 20)]     # 纯静音窗（无语音无回声）
ECHO_WIN = [(4.0, 6.0), (0.2, 1.0)]                 # 仅回声窗（近端静音、回声在）
SIM_LEN = 20.0

def read_wav(path):
    with wave.open(path) as w:
        assert w.getsampwidth() == 2, f"{path}: not S16"
        ch, fs, n = w.getnchannels(), w.getframerate(), w.getnframes()
        d = np.frombuffer(w.readframes(n), dtype='<i2').astype(np.float64) / 32768.0
    if ch > 1:
        d = d.reshape(-1, ch)
    return d, fs, ch

def align_out_to_sim(out, ch0):
    ref = np.tile(ch0, 3)                            # sim 循环 3 周期
    c = fftconvolve(ref, out[::-1], mode='valid')    # c[j] = Σ ref[j+i]·out[i]
    j = int(np.argmax(c))
    seg = ref[j:j + len(out)]
    r = np.corrcoef(out, seg)[0, 1]
    return j, r, seg                                 # out[i] ↔ sim_time(j+i)

def win_rms(x, offset, a, b):
    """out 域窗口：sim_time∈[a,b) ⇔ 索引 i 满足 (offset+i) mod (20·FS) ∈ [a,b)"""
    idx = np.arange(len(x))
    sim_t = (offset + idx) % int(SIM_LEN * FS)
    m = (sim_t >= a * FS) & (sim_t < b * FS)
    return float(np.sqrt(np.mean(x[m] ** 2))) if m.sum() > FS // 2 else float('nan'), int(m.sum())

def rms_db(a, b):
    return 20 * np.log10(a / b) if a > 0 and b > 0 else float('-inf')

def parse_stats(path):
    rows = []
    for line in open(path):
        m = re.match(r'\[stats\] t=([\d.]+)s?\s+(.*)', line.strip())
        if not m:
            continue
        kv = dict(re.findall(r'(\w+)=(-?[\d.]+)', m.group(2)))
        rows.append((float(m.group(1)), kv))
    return rows

def main():
    d = sys.argv[1]
    ok = True
    def check(name, cond, detail):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")

    sim4, fs, nch = read_wav(os.path.join(d, 'sim4.wav'))
    ch0 = sim4[:, 0] if nch == 4 else sim4

    res = {}
    for tag in ('A', 'B'):
        out, fs_o, ch_o = read_wav(os.path.join(d, f'out_{tag}.wav'))
        dur = len(out) / fs_o
        print(f"== out_{tag}.wav: {ch_o}ch {fs_o}Hz {dur:.2f}s")
        if tag == 'A':
            check("① 16k mono S16", fs_o == 16000 and ch_o == 1, f"{fs_o}Hz {ch_o}ch S16")
            check("① 时长≈20s", abs(dur - 20) <= 0.5, f"{dur:.2f}s")
        off, r, seg = align_out_to_sim(out, ch0)
        res[tag] = (out, off)
        print(f"  对齐: out[0] ↔ sim_t={off/FS:.2f}s (loop 内 {(off % int(SIM_LEN*FS))/FS:.2f}s), 与 sim4 ch0 Pearson r={r:.3f}")
        if tag == 'A':
            check("① 与 sim4 ch0 相关>0.3", r > 0.3, f"r={r:.3f}")

    # ① 语音存在 RMS 对比静音（run A）
    outA, offA = res['A']
    sp = [win_rms(outA, offA, a, b)[0] for a, b in SPK_WIN]
    qt = [win_rms(outA, offA, a, b)[0] for a, b in QUIET_WIN]
    sp_rms = float(np.sqrt(np.mean([v ** 2 for v in sp if not np.isnan(v)])))
    qt_rms = float(np.nanmin(qt))
    check("① 语音 RMS vs 静音 >20dB", rms_db(sp_rms, qt_rms) > 20,
          f"speech={sp_rms:.5f} quiet={qt_rms:.5f} → {rms_db(sp_rms, qt_rms):.1f}dB")

    # ② AEC 生效：仅回声窗 B/A 能量差
    outB, offB = res['B']
    for a, b in ECHO_WIN:
        ea, na = win_rms(outA, offA, a, b)
        eb, nb = win_rms(outB, offB, a, b)
        db = rms_db(eb, ea)
        check(f"② 回声残留差 [{a},{b}]s ≥6dB", db >= 6,
              f"B(无ref)={eb:.5f} A(有ref)={ea:.5f} → {db:.1f}dB ({na} vs {nb} 样本)")

    # ③ stats
    for tag, feed in (('A', 'far48'), ('B', 'zero48')):
        rows = parse_stats(os.path.join(d, f'engine_{tag}.log'))
        last = rows[-1][1]
        rtf_max = max(float(kv['rtf']) for _, kv in rows)
        fifo_read, ref_s = float(last['fifo_read']), float(last['ref_samples'])
        doa = float(last['doa_latest'])
        print(f"== stats {tag}(灌{feed}): fifo_read={fifo_read:.0f} ref_samples={ref_s:.0f} "
              f"doa={doa:.1f} rtf_max={rtf_max:.3f} dropped={last.get('fifo_dropped')}")
        check(f"③[{tag}] fifo_read≈48000×8±10%", abs(fifo_read - 384000) <= 38400, f"{fifo_read:.0f}")
        check(f"③[{tag}] ref_samples≈16000×8±10%", abs(ref_s - 128000) <= 12800, f"{ref_s:.0f}")
        check(f"③[{tag}] doa_latest∈[15,45]", 15 <= doa <= 45, f"{doa:.1f}")
        check(f"③[{tag}] rtf<1", rtf_max < 1, f"max={rtf_max:.3f}")

    print("\nRESULT:", "ALL PASS" if ok else "HAS FAILURES")
    sys.exit(0 if ok else 1)

if __name__ == '__main__':
    main()
