#!/bin/bash
# calib_mic_order.sh — 板端一键「麦序标定」：录 12s 原始 4ch + 分析通道顺序
# 用法（板上）: ~/voice/tools/calib_mic_order.sh [秒数，默认 12] [输出wav，默认 /tmp/calib.wav]
#
# 操作要领（录音期间）：说话或拍手者从线阵一端开始、沿阵列法线方向（垂直于
# 阵列的直线，距阵列 30~50cm）慢速走向另一端，全程持续出声（1-2-3-4 数数即可）。
# 或依次凑近每个麦说 1~2 秒。详见 CALIB.md。
set -u
DUR=${1:-12}
OUT=${2:-/tmp/calib.wav}
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "[calib] 录 $DUR 秒原始 4ch16k（通道N=物理麦N）→ $OUT"
echo "[calib] >>> 现在开始沿线阵法向从一端慢速走到另一端，持续说话/拍手 <<<"
sudo ~/ac108/mic4-record16.sh "$DUR" "$OUT" || { echo "[calib] 录音失败"; exit 1; }

python3 "$HERE/calib_mic_order.py" "$OUT"
