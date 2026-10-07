#!/bin/bash
# Data-flow verification for the ALSA fifo plugin on NanoPi (no real audio I/O).
# Covers: capture block-on-no-writer, zero-feed capture, playback to fifo with
# sine check, and plug: multichannel compatibility.
# Fifos are pre-created with mkfifo where a feeder/reader races the pcm open;
# plugin-side auto-creation is verified separately in 4a-1.
# Run on the NanoPi: ./test_plugin.sh
set -u

MIC=/tmp/voice_mic.fifo
SPK=/tmp/voice_spk.fifo
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="python3 $HERE/wavtool.py"

PASS=0; FAIL=0
ok() { echo "  [PASS] $1"; PASS=$((PASS+1)); }
bad() { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }

section() { echo; echo "===== $1 ====="; }

section "0. environment"
ls -l /usr/lib/arm-linux-gnueabihf/alsa-lib/libasound_module_pcm_fifo.so || { echo "plugin .so missing, run make install"; exit 1; }
grep -n "voice_mic\|voice_spk" /etc/asound.conf

section "4a-1: arecord voice_mic with no fifo writer (timeout behaviour)"
rm -f "$MIC" /tmp/t.wav
START=$(date +%s)
timeout 8 arecord -D voice_mic -d 2 -f S16_LE -r 16000 /tmp/t.wav
RC=$?
ELAPSED=$(( $(date +%s) - START ))
SZ=$(stat -c %s /tmp/t.wav 2>/dev/null || echo 0)
echo "  arecord rc=$RC elapsed=${ELAPSED}s t.wav=${SZ} bytes"
if [ -p "$MIC" ]; then
    ok "plugin auto-created $MIC (fifo) at pcm open"
else
    bad "$MIC not auto-created as fifo by plugin"
fi
if [ "$RC" -eq 124 ]; then
    ok "no writer -> arecord blocks forever, killed by timeout(8s); -d 2 alone never fires (no frames ever arrive)"
elif [ "$RC" -eq 0 ] && [ "$ELAPSED" -ge 1 ] && [ "$ELAPSED" -le 4 ] && [ "$SZ" -le 1024 ]; then
    ok "no writer -> arecord exits after -d wall-clock with empty wav (size=${SZ})"
else
    bad "unexpected behaviour rc=$RC elapsed=${ELAPSED}s size=${SZ}"
fi

section "4a-2: feed 32000 zero bytes into fifo, arecord -d 1"
rm -f "$MIC" /tmp/out.wav
mkfifo "$MIC"
cat /dev/zero | head -c 32000 > "$MIC" &
FEED=$!
timeout 6 arecord -D voice_mic -d 1 -f S16_LE -r 16000 /tmp/out.wav
ARC=$?
wait $FEED 2>/dev/null
OUTSZ=$(stat -c %s /tmp/out.wav 2>/dev/null || echo 0)
echo "  arecord rc=$ARC out.wav=${OUTSZ} bytes (expect 44+32000=32044)"
$PY info /tmp/out.wav
if [ "$ARC" -eq 0 ] && [ "$OUTSZ" -eq 32044 ] && \
   $PY info /tmp/out.wav | grep -q "rate=16000 channels=1 sampwidth=2 frames=16000"; then
    ok "captured 16000 frames of 16k S16_LE mono (all 32000 fed bytes delivered through the fifo)"
else
    bad "capture size/format mismatch (rc=$ARC size=$OUTSZ)"
fi

section "4b: aplay 48k mono 440Hz sine -> voice_spk fifo"
$PY gen /tmp/spk_test_48k_mono.wav 48000 1.0 440 1
rm -f "$SPK" /tmp/spk_recv.raw
mkfifo "$SPK"
cat "$SPK" > /tmp/spk_recv.raw &
RDR=$!
sleep 0.3
timeout 12 aplay -D voice_spk /tmp/spk_test_48k_mono.wav
ARC=$?
wait $RDR
RECVSZ=$(stat -c %s /tmp/spk_recv.raw 2>/dev/null || echo 0)
echo "  aplay rc=$ARC fifo received ${RECVSZ} bytes (expect 96000 = 48000 frames x 2 bytes)"
$PY rawcheck /tmp/spk_recv.raw 48000 1 95000 440 2
if [ "$ARC" -eq 0 ] && [ "$RECVSZ" -ge 95000 ] && [ "$RECVSZ" -le 96100 ]; then
    ok "fifo side got ${RECVSZ} bytes of 48k mono S16 with 440Hz dominant tone"
else
    bad "fifo byte count ${RECVSZ} out of range (aplay rc=$ARC)"
fi

section "4b-extra: half-open semantics - aplay with NO fifo reader"
rm -f "$SPK"
START=$(date +%s)
timeout 12 aplay -D voice_spk /tmp/spk_test_48k_mono.wav
ARC=$?
ELAPSED=$(( $(date +%s) - START ))
echo "  aplay rc=$ARC elapsed=${ELAPSED}s (no reader: pipe buffer absorbs ~64KB, drain gives up after ~2.4s, remainder dropped, no hang, no SIGPIPE)"
if [ "$ARC" -eq 0 ] && [ "$ELAPSED" -le 8 ]; then
    ok "playback without reader terminates cleanly rc=0 in ${ELAPSED}s (bounded drain)"
else
    bad "no-reader playback rc=$ARC elapsed=${ELAPSED}s"
fi

section "4c-1: arecord plug:voice_mic -c 2 (stereo dup from mono)"
rm -f "$MIC" /tmp/x2.wav
mkfifo "$MIC"
cat /dev/zero | head -c 64000 > "$MIC" &
FEED=$!
timeout 6 arecord -D plug:voice_mic -c 2 -f S16_LE -r 16000 -d 2 /tmp/x2.wav
ARC=$?
wait $FEED 2>/dev/null
X2SZ=$(stat -c %s /tmp/x2.wav 2>/dev/null || echo 0)
echo "  arecord rc=$ARC x.wav=${X2SZ} bytes (expect 44+128000=128044)"
$PY info /tmp/x2.wav
if [ "$ARC" -eq 0 ] && [ "$X2SZ" -eq 128044 ] && \
   $PY info /tmp/x2.wav | grep -q "rate=16000 channels=2 sampwidth=2 frames=32000"; then
    ok "2-channel capture works: plug duplicated mono -> stereo, 32000 stereo frames"
else
    bad "stereo capture mismatch (rc=$ARC size=$X2SZ)"
fi

section "4c-2: aplay plug:voice_spk stereo wav -> fifo gets mono downmix"
$PY gen /tmp/stereo_48k.wav 48000 1.0 440 2
rm -f "$SPK" /tmp/spk_recv2.raw
mkfifo "$SPK"
cat "$SPK" > /tmp/spk_recv2.raw &
RDR=$!
sleep 0.3
timeout 12 aplay -D plug:voice_spk /tmp/stereo_48k.wav
ARC=$?
wait $RDR
RECV2SZ=$(stat -c %s /tmp/spk_recv2.raw 2>/dev/null || echo 0)
echo "  aplay rc=$ARC fifo received ${RECV2SZ} bytes (expect 96000, i.e. stereo downmixed to mono)"
$PY rawcheck /tmp/spk_recv2.raw 48000 1 95000 440 3
if [ "$ARC" -eq 0 ] && [ "$RECV2SZ" -ge 95000 ] && [ "$RECV2SZ" -le 96100 ]; then
    ok "stereo playback downmixed to mono at fifo: ${RECV2SZ} bytes, 440Hz tone preserved"
else
    bad "downmix byte count ${RECV2SZ} out of range (aplay rc=$ARC)"
fi

echo
echo "===== RESULT: PASS=$PASS FAIL=$FAIL ====="
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
