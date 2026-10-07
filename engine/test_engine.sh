#!/bin/bash
# test_engine.sh — voice-engine 板上无声测试（mock 桩模式，全部无音频播放）
#   test0  多相抽取器自检（48k→16k 频响）
#   test1  --simulate-capture → /tmp/voice_mic.fifo 输出 RMS/通断（语音 vs 静音 ≥20dB）
#   test2  回采通路：48k 定速灌 /tmp/voice_spk.fifo → fifo_read≈48000/s、ref_samples≈16000/s
#   test3  60s 稳定性：无崩溃、stats 持续、RTF<1（含 fifo 突发灌入的溢出路径）
#   test4  真实采集冒烟（hw:1,0，默认 16k 直采）：captured_samples、tag 分布、环境音 RMS（硬件可用时）
# 用法: ./test_engine.sh [test编号...]   默认 0 1 2 3 4
set -u
cd "$(dirname "$0")"
E=./voice_engine
MIC=${MIC_FIFO:-/tmp/voice_mic.fifo}
SPK=${SPK_FIFO:-/tmp/voice_spk.fifo}
PASS=0; FAIL=0; RUN=${*:-0 1 2 3 4}

ok(){ echo "  PASS: $1"; PASS=$((PASS+1)); }
bad(){ echo "  FAIL: $1"; FAIL=$((FAIL+1)); }
field(){ grep -o "$1=[0-9.-]*" "$2" | tail -1 | cut -d= -f2; }
wait_fifo(){ local i=0; while [ ! -p "$1" ] && [ $i -lt 100 ]; do sleep 0.1; i=$((i+1)); done; }
cleanup(){ kill $(jobs -p) 2>/dev/null; sleep 0.5; }
trap cleanup EXIT

[ -x "$E" ] || { echo "缺 $E（先 make）"; exit 9; }
MODE=$($E --help 2>&1 | head -1)

for T in $RUN; do
case $T in
0)
  echo "== test0: 多相抽取器自检 =="
  $E --selftest >/tmp/t0.out 2>&1 && ok "selftest 频响" || { bad "selftest 频响"; cat /tmp/t0.out; }
  ;;

1)
  echo "== test1: simulate 捕获 → mic fifo RMS/通断 =="
  if [ ! -f test4ch16k.wav ]; then
    echo "  缺 test4ch16k.wav → gen_test_wav.py 生成"
    python3 gen_test_wav.py >/dev/null 2>&1 || { echo "  SKIP: 生成失败"; continue; }
  fi
  rm -f "$MIC" "$SPK"
  $E --ipc fifo --simulate-capture test4ch16k.wav --stats-interval 2 >/dev/null 2>/tmp/t1.log &
  EP=$!
  sleep 1
  head -c 768000 "$MIC" > /tmp/t1.raw    # 读 24s @16k S16 mono（实时节奏）
  python3 check_rms.py /tmp/t1.raw >/tmp/t1.rms 2>&1 && ok "语音段 RMS 超静音段 20dB: $(cat /tmp/t1.rms)" \
      || bad "RMS 比不达标: $(cat /tmp/t1.rms)"
  CAP=$(field captured_samples /tmp/t1.log); FW=$(field fifo_written /tmp/t1.log)
  AB=$(field aec_blocks /tmp/t1.log)
  [ -n "$CAP" ] && [ "$CAP" -gt 700000 ] && ok "captured_samples 增长 ($CAP)" || bad "captured_samples 异常 ($CAP)"
  [ -n "$AB" ] && [ "$AB" -gt 700 ] && ok "aec_blocks 增长 ($AB ≈ 24s×31.25)" || bad "aec_blocks 异常 ($AB)"
  [ -n "$FW" ] && [ "$FW" -gt 360000 ] && ok "fifo_written ≈ 读走样本数 ($FW)" || bad "fifo_written 异常 ($FW)"
  kill -INT $EP 2>/dev/null; wait $EP 2>/dev/null
  ;;

2)
  echo "== test2: 回采通路（spk fifo → ref 环） =="
  [ -f test48k.wav ] || { echo "  SKIP: 缺 test48k.wav"; continue; }
  rm -f "$MIC" "$SPK"
  $E --ipc fifo --simulate-capture test4ch16k.wav --stats-interval 2 >/dev/null 2>/tmp/t2.log &
  EP=$!
  sleep 1
  python3 feed_fifo.py test48k.wav "$SPK" 30 >/tmp/t2.feed 2>&1 &
  FD=$!
  sleep 4
  R1=$(field fifo_read /tmp/t2.log); S1=$(field ref_samples /tmp/t2.log)
  sleep 4
  R2=$(field fifo_read /tmp/t2.log); S2=$(field ref_samples /tmp/t2.log)
  if [ -n "$R1" ] && [ -n "$R2" ] && [ "$R2" -gt "$R1" ]; then
    RATE=$(( (R2 - R1) / 4 ))
    [ "$RATE" -gt 43200 ] && [ "$RATE" -lt 52800 ] && ok "fifo_read 速率 ${RATE}/s ≈ 48000/s" \
        || bad "fifo_read 速率 ${RATE}/s 偏离 48000/s"
  else
    bad "fifo_read 无增长 ($R1 → $R2)"
  fi
  if [ -n "$S1" ] && [ -n "$S2" ] && [ "$S2" -gt "$S1" ]; then
    RS=$(( (S2 - S1) / 4 ))
    [ "$RS" -gt 14000 ] && [ "$RS" -lt 18000 ] && ok "ref_samples 速率 ${RS}/s ≈ 16000/s（AEC ref 环收到数据）" \
        || bad "ref_samples 速率 ${RS}/s 偏离 16000/s"
  else
    bad "ref_samples 无增长 ($S1 → $S2)"
  fi
  kill $FD $EP 2>/dev/null; wait $EP 2>/dev/null
  ;;

3)
  echo "== test3: 60s 稳定性 =="
  if [ ! -f test4ch16k.wav ]; then
    echo "  缺 test4ch16k.wav → gen_test_wav.py 生成"
    python3 gen_test_wav.py >/dev/null 2>&1 || { echo "  SKIP: 生成失败"; continue; }
  fi
  rm -f "$MIC" "$SPK"
  timeout -s INT 66 $E --ipc fifo --simulate-capture test4ch16k.wav --stats-interval 5 >/dev/null 2>/tmp/t3.log &
  EP=$!
  wait_fifo "$MIC"; cat "$MIC" > /dev/null &            # 持续读者
  wait_fifo "$SPK"
  (for i in 1 2 3; do cat test48k.wav > "$SPK" 2>/dev/null; sleep 1; done) &  # 突发灌入（溢出路径）
  wait $EP; RC=$?
  sleep 0.5
  N=$(grep -c '\[stats\]' /tmp/t3.log)
  CAP=$(field captured_samples /tmp/t3.log)
  FR=$(field fifo_read /tmp/t3.log)
  RTFMAX=$(grep -o 'rtf=[0-9.]*' /tmp/t3.log | cut -d= -f2 | sort -g | tail -1)
  { [ "$RC" = "0" ] || [ "$RC" = "124" ]; } && grep -q '\[engine\] bye' /tmp/t3.log \
      && ok "66s 后 SIGINT 优雅关停" || bad "退出码 $RC / 无 bye"
  [ -n "$N" ] && [ "$N" -ge 11 ] && ok "stats 持续更新 (${N} 行)" || bad "stats 行数 $N"
  [ -n "$FR" ] && [ "$FR" -gt 2800000 ] && ok "突发灌入被吞（fifo_read=$FR，含 ref 环溢出路径）" || bad "fifo_read=$FR"
  [ -n "$CAP" ] && [ "$CAP" -gt 3600000 ] && [ "$CAP" -lt 4600000 ] && ok "captured_samples ≈ 16000×4×66 ($CAP)" \
      || bad "captured_samples=$CAP"
  [ -n "$RTFMAX" ] && python3 -c "exit(0 if $RTFMAX < 1.0 else 1)" && ok "RTF<1 (max=$RTFMAX)" || bad "RTF max=$RTFMAX"
  grep -q "Segmentation\|Aborted\|internal error" /tmp/t3.log && bad "崩溃痕迹" || ok "无崩溃"
  ;;

4)
  echo "== test4: 真实采集冒烟（hw:1,0, 16k 直采默认, 不播放） =="
  if ! arecord -l 2>/dev/null | grep -qE "micgen|voicen4mic"; then
    echo "  SKIP: 无 micgen 声卡（硬件不可用）"; continue
  fi
  # 引擎默认 --capture-fs 16k：先写 16k 配方（内核模块开机写的是 48k 配方）
  if [ -x ~/ac108/mic4-record16.sh ]; then
    sudo -n ~/ac108/mic4-record16.sh cfg >/dev/null 2>&1 \
      && echo "  note: 已写 16k 配方（mic4-record16.sh cfg）" \
      || echo "  note: 16k 配方写入失败（引擎若报 hw_params/_tag 异常先检查 ~/ac108/README16.md）"
  else
    echo "  note: 缺 ~/ac108/mic4-record16.sh，沿用当前 AC108 配方"
  fi
  rm -f "$MIC" "$SPK"
  $E --ipc fifo --stats-interval 2 >/dev/null 2>/tmp/t4.log &
  EP=$!
  sleep 0.5
  # 用户态打不开/配置不了硬件则 sudo 重试
  if kill -0 $EP 2>/dev/null && grep -Eq "open .* 失败|hw_params 失败|readi error" /tmp/t4.log; then
    kill -INT $EP 2>/dev/null; wait $EP 2>/dev/null
    echo "  note: 用户无权限，sudo 重试"
    rm -f "$MIC" "$SPK"
    sudo -n $E --ipc fifo --stats-interval 2 >/dev/null 2>/tmp/t4.log &
  EP=$!
  sleep 0.3
  fi
  if ! kill -0 $EP 2>/dev/null && grep -q "\[capture\]" /tmp/t4.log; then
    bad "采集线程启动即退：$(grep capture /tmp/t4.log | tail -2)"
    continue
  fi
  wait_fifo "$MIC"
  head -c 320000 "$MIC" > /tmp/t4.raw &   # 读 10s
  sleep 12
  kill -INT $EP 2>/dev/null; wait $EP 2>/dev/null
  sudo -n pkill -INT -f "voice_engine --stats" 2>/dev/null; sleep 1

  CAP=$(field captured_samples /tmp/t4.log)
  TAG=$(grep '\[stats\]' /tmp/t4.log | tail -1 | grep -o 'tag=\[[^]]*\]' | cut -d'[' -f2 | cut -d']' -f1)
  if grep -q "tag 分布异常" /tmp/t4.log; then
    bad "tag 分布异常（AC108 未配置？引擎已提示跑 ~/ac108/mic4-record.sh 1 /dev/null）"
  else
    RUNT=$(grep '\[stats\]' /tmp/t4.log | tail -1 | grep -o ' t=[0-9.]*' | cut -d= -f2)
    if [ -n "$CAP" ] && [ -n "$RUNT" ] && \
       python3 -c "import sys; c=$CAP; t=$RUNT; sys.exit(0 if abs(c-64000*t) < 6400*t else 1)" 2>/dev/null; then
      ok "captured_samples ≈ 16000×4×t ($CAP @ t=${RUNT}s)"
    else
      bad "captured_samples=$CAP @ t=${RUNT}s（期望 ≈64000×t）"
    fi
    T_OK=$(python3 -c "t=[float(x) for x in '$TAG'.split()]; print(1 if t and all(15<=x<=35 for x in t) else 0)" 2>/dev/null)
    [ "$T_OK" = "1" ] && ok "tag 分布各≈25% [$TAG]" || bad "tag 分布 [$TAG]"
    SZ=$(stat -c %s /tmp/t4.raw 2>/dev/null || echo 0)
    if [ "$SZ" -ge 300000 ]; then
      RMS=$(python3 check_rms.py /tmp/t4.raw 1 9 1 9 2>/dev/null | grep -o 'speech_rms=[0-9.]*' | cut -d= -f2)
      [ -n "$RMS" ] && python3 -c "exit(0 if $RMS > 20 else 1)" && ok "fifo 输出 RMS 合理（环境音非零 rms=$RMS）" \
          || bad "fifo 输出 RMS 过低 ($RMS)"
    else
      bad "t4.raw 太小 ($SZ 字节)"
    fi
  fi
  ;;
esac
done

echo
echo "==== $MODE | summary: PASS=$PASS FAIL=$FAIL ===="
[ "$FAIL" -eq 0 ]
