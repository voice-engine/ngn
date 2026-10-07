#!/bin/bash
# test_fifo_race.sh — fifo 读路径竞态回归（SHM_IPC_REPORT §7.5 遗留清账后的防复发测试）
#
# 覆盖 2026-10-06 修复的三缺陷（ENGINE_REPORT §9）：
#   T1 spk fifo 50ms 盲窗：0.2s 短文件 aplay（生命 ~7ms）数据必须全额收割
#      （修复前：内核在写者 close 时丢弃管道缓冲，fifo_read=0）
#   T2 ref 状态机活锁：灌 <cushion(4096) 的滞留 ref 尾巴后块率不得塌缩
#      （修复前：HOLD↔DRAINED 无限乒乓，63→2 块/窗）
#   T3 长跑节拍：140s 真采集 + 读者/写者剧本，stats 时间戳单调、间隔 ≈2s、
#      块率不塌（可多轮：ROUND=n）
#
# 板上运行: cd /home/i/voice/engine && ./test_fifo_race.sh [rounds，默认 1]
# 默认 mock 引擎（同一 fifo/ref/stats 代码路径，rtf≈0 不发热）；BIN=voice_engine
# 跑真算法（注意无散热 H3 会升温，脚本带 >92°C 护栏，见 ENGINE_REPORT §9.5）。
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ENG="$HERE"
T=/tmp/fifo_race
ROUNDS=${1:-1}
BIN=${BIN:-voice_engine_mock}
mkdir -p "$T"
PASS=0; FAIL=0
ok()  { echo "  [PASS] $1"; PASS=$((PASS+1)); }
bad() { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }
section() { echo; echo "===== $1 ====="; }

pkill_engines() { for p in $(pgrep -f "voice_eng""ine"); do sudo kill -9 $p 2>/dev/null; done; sleep 0.5; }

section "0. 素材"
python3 ../shm_wavtool.py gen "$T/s440.wav" 48000 8 440 0.4 >/dev/null
python3 ../shm_wavtool.py gen "$T/burst02.wav" 48000 0.2 440 0.4 >/dev/null
python3 - << 'PY' || exit 1
import struct, math, wave
with open("/tmp/fifo_race/tail.wav","wb") as f:   # 0.1875s@48k -> 3000 ref 样本 < cushion
    w = wave.open(f); w.setnchannels(1); w.setsampwidth(2); w.setframerate(48000)
    w.writeframes(b"".join(struct.pack("<h", int(0.4*32767*math.sin(2*math.pi*440*i/48000))) for i in range(9000)))
    w.close()
print("  素材: s440(8s) burst02(0.2s) tail(0.1875s)")
PY

section "T1. 短命写者数据存活（盲窗回归）"
pkill_engines
rm -f /tmp/voice_mic.fifo /tmp/voice_spk.fifo "$T/t1.log"
"$ENG/$BIN" --ipc fifo --simulate-capture "$ENG/sim4.wav" --null-playback \
    --stats-interval 2 > "$T/t1.log" 2>&1 &
EP=$!; sleep 1.5
arecord -D voice_mic -d 12 -f S16_LE -r 16000 "$T/t1_mic.wav" 2>/dev/null &
sleep 2
aplay -D voice_spk "$T/burst02.wav" >/dev/null 2>&1
sleep 5
kill $EP 2>/dev/null; sleep 0.5; kill -9 $EP 2>/dev/null; pkill_engines
FRD=$(grep -ao 'fifo_read=[0-9]*' "$T/t1.log" | tail -1 | cut -d= -f2)
echo "  引擎 fifo_read=$FRD（24000B=12000 帧）"
[ -n "$FRD" ] && [ "$FRD" -ge 12000 ] && ok "0.2s 短写者全额收割" || bad "短写者数据丢失（fifo_read=${FRD:-无}）"

section "T2. 滞留尾巴活锁回归（<cushion 尾巴后块率不塌缩）"
pkill_engines
rm -f /tmp/voice_mic.fifo /tmp/voice_spk.fifo "$T/t2.log"
"$ENG/$BIN" --ipc fifo --simulate-capture "$ENG/sim4.wav" --null-playback \
    --stats-interval 2 > "$T/t2.log" 2>&1 &
EP=$!; sleep 1.5
arecord -D voice_mic -d 30 -f S16_LE -r 16000 "$T/t2_mic.wav" 2>/dev/null &
sleep 3
python3 - << 'PY'    # 慢速写者 3 chunk（存活 ~0.6s，绕过任何盲窗），确保尾巴入环
import time
data = open("/tmp/fifo_race/tail.wav","rb").read()[44:]
fd = open("/tmp/voice_spk.fifo","wb", buffering=0)
for i in range(3):
    fd.write(data[i*6000:(i+1)*6000]); time.sleep(0.2)
fd.close()
PY
sleep 22
kill $EP 2>/dev/null; sleep 0.5; kill -9 $EP 2>/dev/null; pkill_engines
BLK=$(python3 - "$T/t2.log" << 'PY'
import sys,re
ls=[int(m.group(1)) for l in open(sys.argv[1]) if (m:=re.search(r"aec_blocks=(\d+)",l))]
print(min(b-a for a,b in zip(ls,ls[1:])) if len(ls)>2 else 0)
PY
)
DISC=$(grep -ac "丢弃滞留 ref" "$T/t2.log")
echo "  灌尾后 min 块/窗=$BLK（正常 ~63，活锁 ~2）；丢弃日志=$DISC 条"
[ "$BLK" -ge 60 ] && [ "$DISC" -ge 1 ] && ok "活锁消除 + 滞留丢弃路径触发" || bad "块率塌缩或未触发丢弃（min=$BLK discard=$DISC）"

section "T3. 长跑节拍（$ROUNDS 轮 × 140s，$BIN，真采集）"
for r in $(seq 1 "$ROUNDS"); do
    pkill_engines
    rm -f "$T/t3_$r.log" /tmp/voice_mic.fifo /tmp/voice_spk.fifo
    sudo ~/ac108/mic4-record16.sh cfg >/dev/null 2>&1
    T0=$SECONDS
    sudo "$ENG/$BIN" --ipc fifo --capture-fs 16k --stats-interval 2 --null-playback > "$T/t3_$r.log" 2>&1 &
    EP=$!
    sleep 3
    ( while [ -e /proc/$EP ]; do sleep 5; [ "$(cat /sys/class/thermal/thermal_zone0/temp)" -gt 92000 ] \
        && { echo "[THERMAL] abort" >> "$T/t3_$r.th"; sudo kill -9 $EP; }; done ) &
    TGP=$!
    arecord -D voice_mic -d 125 -f S16_LE -r 16000 "$T/t3_${r}_mic.wav" 2>/dev/null &
    ARP=$!
    ( sleep 5;  aplay -D voice_spk "$T/s440.wav"  >/dev/null 2>&1
      sleep 12; aplay -D voice_spk "$T/burst02.wav" >/dev/null 2>&1
      sleep 10; aplay -D voice_spk "$T/burst02.wav" >/dev/null 2>&1
      sleep 10; python3 - << 'PY'
import time
data = open("/tmp/fifo_race/tail.wav","rb").read()[44:]
fd = open("/tmp/voice_spk.fifo","wb", buffering=0)
for i in range(3):
    fd.write(data[i*6000:(i+1)*6000]); time.sleep(0.2)
fd.close()
PY
    ) &
    CLP=$!
    ELAPSED=$((SECONDS-T0)); [ $ELAPSED -lt 140 ] && sleep $((140-ELAPSED))
    sudo kill -INT $EP 2>/dev/null; sleep 2; sudo kill -9 $EP 2>/dev/null
    kill $ARP $CLP $TGP 2>/dev/null; pkill_engines
    V=$(python3 - "$T/t3_$r.log" << 'PY'
import sys,re
ls=[(float(m.group(1)),int(m.group(2))) for l in open(sys.argv[1]) if (m:=re.search(r"\[stats\] t=([0-9.]+)s.*?aec_blocks=(\d+)",l))]
if len(ls)<50: print("FAIL stats<50"); sys.exit()
gaps=[b-a for (a,_),(b,_) in zip(ls,ls[1:])]
bws=[b-a for (_,a),(_,b) in zip(ls,ls[1:])]
mono=all(b>a for (a,_),(b,_) in zip(ls,ls[1:]))
ok = mono and max(gaps)<=4.5 and min(bws)>=30
print(f"{'PASS' if ok else 'FAIL'} stats={len(ls)} mono={mono} maxgap={max(gaps):.2f} meangap={sum(gaps)/len(gaps):.2f} minblk={min(bws)}")
PY
)
    echo "  round $r: $V"
    case "$V" in PASS*) ok "长跑 round $r";; *) bad "长跑 round $r";; esac
done

echo
echo "===== RESULT: PASS=$PASS FAIL=$FAIL ====="
[ "$FAIL" = 0 ]
