#!/bin/bash
# run_round.sh <feed.wav> <tag> — 步骤2 单轮：引擎(simulate) + 定速灌 spk fifo + arecord voice_mic
# 顺序：feeder 先起（open 阻塞等引擎读端；t0 锚定在 open 之后）→ 引擎 → arecord。
# ref 相对 mic 的滞后 ≈ 引擎 spk 线程打开 fifo 的时间差（~50ms 内）<< AEC 跨度 320ms。
set -e
cd /home/i/voice/engine
FEED=$1; TAG=$2
rm -f /tmp/voice_mic.fifo /tmp/voice_spk.fifo out_$TAG.wav engine_$TAG.log feed_$TAG.log arecord_$TAG.log
python3 feed_fifo.py "$FEED" /tmp/voice_spk.fifo >feed_$TAG.log 2>&1 &
FEEDPID=$!
./voice_engine --simulate-capture sim4.wav --null-playback --stats-interval 2 >engine_$TAG.log 2>&1 &
ENG=$!
arecord -D voice_mic -f S16_LE -r 16000 -c 1 -d 20 out_$TAG.wav >arecord_$TAG.log 2>&1 || true
kill -INT $ENG 2>/dev/null || true
wait $ENG || true
wait $FEEDPID 2>/dev/null || true
echo "== round $TAG done"
ls -l out_$TAG.wav
echo "--- last stats:"
grep '\[stats\]' engine_$TAG.log | tail -3
echo "--- feed log:"; cat feed_$TAG.log
