#!/usr/bin/env python3
# check_rms.py — 无声验证 /tmp/voice_mic.fifo 抽出的 16k S16 mono 流
# 用法: check_rms.py <file.raw> [sp_lo sp_hi si_lo si_hi]
#   默认窗口（秒）：语音 [1,8)（读流时间），静音 [11,18)
#   依据 test_engine.sh 时序：读者在引擎启动 ~1s 后接上，读流 t 对应引擎 t+1s，
#   故读流 [1,8) = 引擎 [2,9)s = 语音段；读流 [11,18) = 引擎 [12,19)s = 静音段。
import array
import math
import sys

FS = 16000
a = array.array('h')
a.frombytes(open(sys.argv[1], 'rb').read())
sp_lo, sp_hi, si_lo, si_hi = 1, 8, 11, 18
if len(sys.argv) >= 6:
    sp_lo, sp_hi, si_lo, si_hi = map(int, sys.argv[2:6])


def rms(seg):
    if not len(seg):
        return 0.0
    return math.sqrt(sum(float(x) * x for x in seg) / len(seg))


need = max(sp_hi, si_hi) * FS
if len(a) < need:
    print(f"FAIL: 数据不足 {len(a)}/{need} 样本")
    sys.exit(2)

sp = rms(a[sp_lo * FS : sp_hi * FS])
si = rms(a[si_lo * FS : si_hi * FS])
si_eff = si if si > 0 else 1e-3
db = 20 * math.log10(max(sp, 1e-9) / si_eff)
print(f"speech_rms={sp:.1f} silence_rms={si:.1f} ratio={db:.1f}dB")
sys.exit(0 if db > 20.0 else 1)
