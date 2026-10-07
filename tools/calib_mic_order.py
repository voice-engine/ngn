#!/usr/bin/env python3
# calib_mic_order.py — 线阵 4 麦「通道 → 物理位置」标定（LIVE_REPORT 遗留：
# 通道-物理位置映射未确证，通道重排曾使 DOA 7.9°→41.3°）
#
# 原理：声源沿线阵法向从一端走到另一端（或依次凑近每个麦说话/拍手）。
# 每个通道的能量包络峰值时刻 ≈ 声源最靠近该麦的时刻 → 按峰值时刻给通道
# 排序即得物理次序。全程能量法，不依赖相位/GCC-PHAT（窄带反射环境下更稳）。
#
# 输入：4ch 16bit 16k WAV，通道 N = 物理麦 N ——用**原始采集路径**录：
#     sudo ~/ac108/mic4-record16.sh 10 /tmp/calib.wav
#   （不要用 arecord -D voice_mic*：那是引擎处理后的单声道，无通道信息）
#
# 输出：ch→位置 推断映射 + 建议 --mic-map 文案；不确定度不足时拒绝给结论
#   （环境音冒烟即应走该分支），exit code：0=给出映射 2=不确定 1=用法错。
#
# 纯 stdlib（板上无 numpy）。窗口能量用 RMS，包络平滑用滑动均值。

import argparse
import math
import struct
import sys
import wave


def read_wav(path):
    w = wave.open(path, "rb")
    nch, sw, rate, nframes = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
    if sw != 2 or nch != 4:
        w.close()
        raise SystemExit(f"[calib] 需要 4ch/16bit WAV（得到 {nch}ch/{sw * 8}bit）: {path}")
    raw = w.readframes(nframes)
    w.close()
    samples = struct.unpack("<%dh" % (nframes * 4), raw)
    chans = [samples[c::4] for c in range(4)]
    return chans, rate


def envelope(ch, win, hop):
    """每窗口 RMS（线性域），返回 len = (len(ch)-win)//hop + 1 的列表"""
    env, acc = [], 0.0
    sq = [s * s for s in ch]
    # 前缀和降复杂度
    pre = [0] * (len(sq) + 1)
    for i, v in enumerate(sq):
        pre[i + 1] = pre[i] + v
    for start in range(0, len(sq) - win + 1, hop):
        s = pre[start + win] - pre[start]
        env.append(math.sqrt(s / win))
    return env


def smooth(xs, k):
    if k <= 1:
        return xs[:]
    out, hs = [], k // 2
    for i in range(len(xs)):
        lo, hi = max(0, i - hs), min(len(xs), i + hs + 1)
        out.append(sum(xs[lo:hi]) / (hi - lo))
    return out


def percentile(sorted_xs, p):
    if not sorted_xs:
        return 0.0
    idx = min(len(sorted_xs) - 1, max(0, int(round(p / 100.0 * (len(sorted_xs) - 1)))))
    return sorted_xs[idx]


def db(x):
    return 20.0 * math.log10(x) if x > 0 else -999.0


def main():
    ap = argparse.ArgumentParser(description="线阵 4 麦通道顺序标定（能量包络峰值排序）")
    ap.add_argument("wav", help="4ch 16k 16bit WAV（mic4-record16.sh 原始路径录音）")
    ap.add_argument("--win-ms", type=float, default=40.0, help="包络窗长 ms（默认 40）")
    ap.add_argument("--hop-ms", type=float, default=20.0, help="包络步进 ms（默认 20）")
    ap.add_argument("--snr-db", type=float, default=15.0, help="峰值信噪比门槛 dB（默认 15）")
    ap.add_argument("--min-sep-win", type=int, default=3, help="相邻通道峰值时刻最小间隔（窗，默认 3）")
    ap.add_argument("--smooth-win", type=int, default=5, help="包络滑动平均窗（默认 5）")
    args = ap.parse_args()

    chans, rate = read_wav(args.wav)
    win = max(16, int(rate * args.win_ms / 1000))
    hop = max(8, int(rate * args.hop_ms / 1000))

    print(f"[calib] {args.wav}: 4ch {rate}Hz {len(chans[0]) / rate:.1f}s，"
          f"包络窗 {args.win_ms}ms/步进 {args.hop_ms}ms，平滑 {args.smooth_win} 窗")

    results = []  # (ch, peak_idx, peak_snr_db, floor)
    for c in range(4):
        env = smooth(envelope(chans[c], win, hop), args.smooth_win)
        srt = sorted(env)
        floor = percentile(srt, 10)                      # 噪声底：10 分位
        peak_i = max(range(len(env)), key=lambda i: env[i])
        snr = db(env[peak_i]) - db(max(floor, 1e-9))
        results.append((c, peak_i, snr, floor))
        print(f"  ch{c}: 包络峰 t={peak_i * hop / rate:6.2f}s  峰SNR={snr:6.1f}dB  "
              f"噪声底RMS={floor:8.1f}")

    # ---- 不确定度门槛：任一通道峰 SNR 不足 → 拒绝下结论 ----
    weak = [f"ch{c}(SNR {snr:.1f}dB)" for c, _, snr, _ in results if snr < args.snr_db]
    if weak:
        print(f"[calib] 不确定度不足：{', '.join(weak)} 峰信噪比 < {args.snr_db}dB 门槛。")
        print("[calib] 请人工放源：让声源（说话/拍手）沿线阵法向从一端慢速走到另一端，")
        print("[calib] 或依次凑近每个麦 1~2s；环境底噪无法定位通道顺序。")
        return 2

    ranked = sorted(results, key=lambda r: r[1])         # 按峰值时刻排序
    # ---- 门槛 2：相邻峰值时刻间隔 ≥ min-sep-win（区分度不足=疑似单点声源）----
    for (c1, t1, _, _), (c2, t2, _, _) in zip(ranked, ranked[1:]):
        if t2 - t1 < args.min_sep_win:
            print(f"[calib] 不确定度不足：ch{c1} 与 ch{c2} 峰值时刻仅隔 {(t2 - t1) * hop / rate:.2f}s"
                  f"（< {args.min_sep_win * hop / rate:.2f}s），像固定点声源而非沿线走动。")
            print("[calib] 请重录：声源保持移动（每麦间 ≥1s），或改用依次凑近每个麦的方式。")
            return 2

    print("[calib] 按包络峰值时刻排序（早=先经过=走线起点端）:")
    mic_map = []
    for pos, (c, t, snr, _) in enumerate(ranked):
        print(f"  位置{pos} ← ch{c}   峰 t={t * hop / rate:6.2f}s  SNR={snr:5.1f}dB")
        mic_map.append(c)

    print()
    print(f"[calib] 推断映射（位置→通道）: {mic_map}")
    print(f"[calib] 建议引擎参数文案: --mic-map {','.join(map(str, mic_map))}")
    print("[calib] 注意：位置 0 = 走线起点端（先到达的麦）。若想以另一端为位置 0，")
    print("[calib] 反转顺序即可：--mic-map " + ",".join(map(str, mic_map[::-1])))
    print("[calib] 验证建议：用该映射跑引擎，对已知方向声源看 doa_latest 是否一致（±10°）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
