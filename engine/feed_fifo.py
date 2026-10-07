#!/usr/bin/env python3
# feed_fifo.py — 以精确 48000 样本/s 的节奏把 wav 灌进 fifo（回采通路测试）
# 用法: feed_fifo.py <48k_mono_s16.wav> <fifo> [秒数上限]
import sys
import time
import wave

wav, fifo = sys.argv[1], sys.argv[2]
limit = float(sys.argv[3]) if len(sys.argv) > 3 else 1e9

w = wave.open(wav)
assert w.getframerate() == 48000 and w.getnchannels() == 1, "需要 48k mono"
d = w.readframes(w.getnframes())

CH = 1920          # 40ms 一块（板端 python sleep 粒度 ~10ms，10ms 块会系统性降速；引擎 fifo 高水位 48ms 已覆盖 40ms 块型，速率由 fifo 背压+节拍共同保证）
BYTES = CH * 2
i = 0
sent = 0
# 节拍锚定在 open 成功之后：open 会阻塞到读端出现（引擎 spk 线程打开 fifo），
# 若把 t0 放在 open 之前，阻塞期间的时长会被当作“已播放”而突发追赶，
# 使 ref 相对 mic 超前/滞后到 AEC 滤波器跨度（320ms）之外 → 消回声失效。
with open(fifo, 'wb', buffering=0) as f:
    t0 = time.monotonic()
    while i + BYTES <= len(d) and (time.monotonic() - t0) < limit:
        f.write(d[i : i + BYTES])
        i += BYTES
        sent += CH
        tgt = i / (2 * 48000.0)   # 按样本推进（i 为字节数，S16 单声道×2B；块大小无关）
        dt = tgt - (time.monotonic() - t0)
        if dt > 0:
            time.sleep(dt)
print(f"fed {sent} samples in {time.monotonic()-t0:.1f}s ({sent/(time.monotonic()-t0):.0f}/s)")
