#!/bin/bash
# test_system.sh — 语音链路系统联调可重复流程（NanoPi Air 板上执行）
# 封装步骤 1-4；素材分析（corr/RMS/回声残留）在 Mac 侧 analyze_system.py 完成。
#
# 用法:
#   ./test_system.sh          # 全流程（约 2.5 分钟）
#   ./test_system.sh 1|2|3|4  # 单步
#
# 前置:
#   - /home/i/ngn/algo/libvoice_algo.so（真算法库）
#   - 素材 far48/echo16/sim4/zero48.wav（Mac: gen_system_material.py 生成后 scp）
#   - speaker 物理断开（全流程不播放任何音频；aplay 仅写 fifo）
set -u
cd "$(dirname "$0")"
STEP="${1:-all}"
GOV_FILE=/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

step1() {
    echo "===== 步骤1: 板上重编 engine 链真库 ====="
    make clean >/dev/null
    if make 2>&1 | tee /tmp/ts_build.log | grep -q "链接真库"; then
        echo "[OK] 链接真库（无 -DVA_MOCK）"
    else
        echo "[FAIL] 未检测到真库链接"; exit 1
    fi
    grep -q "error" /tmp/ts_build.log && { echo "[FAIL] 编译错误"; exit 1; }
    ./voice_engine --help >/dev/null 2>&1 && echo "[OK] --help 冒烟 rc=0" || { echo "[FAIL] --help"; exit 1; }
    [ "$(cat $GOV_FILE 2>/dev/null)" = performance ] || \
        echo "[注意] cpufreq governor=$(cat $GOV_FILE 2>/dev/null)（rtf 波动会增大；sudo 可设 performance）"
}

step2() {
    echo "===== 步骤2: 端到端 AEC 数据流（A=灌 far48 / B=灌静音）====="
    for w in far48.wav echo16.wav sim4.wav zero48.wav; do
        [ -f "$w" ] || { echo "[FAIL] 缺素材 $w（Mac: gen_system_material.py）"; exit 1; }
    done
    ./run_round.sh far48.wav A
    ./run_round.sh zero48.wav B
    echo
    echo "[下一步·Mac] scp nanopi:/home/i/ngn/engine/{out_A.wav,out_B.wav,engine_A.log,engine_B.log} ."
    echo "[下一步·Mac] python3 analyze_system.py <dir>   # ①②③ 全量校验"
}

step3() { echo "===== 步骤3: 多应用并发行为 ====="; ./test_t3.sh; }
step4() { echo "===== 步骤4: 真实采集 hw:1,0 + aplay 数据流 ====="; ./test_t4.sh; }

case "$STEP" in
    1) step1 ;; 2) step2 ;; 3) step3 ;; 4) step4 ;;
    all) step1; step2; step3; step4 ;;
    *) echo "用法: $0 [1|2|3|4|all]"; exit 2 ;;
esac
echo "===== test_system.sh ($STEP) 完成 ====="
