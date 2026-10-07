#!/bin/bash
# voice/algo/sync_build.sh — 同步源码到 NanoPi Air 并在板上原生构建 + 单测。
# 用法：./sync_build.sh [build|test|neon-stat|all]（默认 all = clean 构建 + 测试 + NEON 统计）
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE=${NGN_HOST:-nanopi}
RDIR=${NGN_REMOTE:-/home/i/ngn/algo}
MODE="${1:-all}"

FILES="voice_algo.h voice_algo.cpp aec_kf.hpp aec_kf.cpp bf.hpp bf.cpp fft_bf.hpp test_algo.cpp Makefile"

echo "== 同步 -> $REMOTE:$RDIR"
ssh "$REMOTE" "mkdir -p $RDIR"
for f in $FILES; do
  if [ -f "$DIR/$f" ]; then
    cat "$DIR/$f" | ssh "$REMOTE" "cat > $RDIR/$f"
    echo "  $f"
  fi
done

echo "== 板上构建/测试 ($MODE)"
case "$MODE" in
  build) ssh "$REMOTE" "cd $RDIR && make clean >/dev/null && make -j4 all" ;;
  test)  ssh "$REMOTE" "cd $RDIR && make -j4 all && ./test_algo" ;;
  neon-stat) ssh "$REMOTE" "cd $RDIR && make neon-stat" ;;
  all)
    ssh "$REMOTE" "cd $RDIR && make clean >/dev/null && make -j4 all && ./test_algo && make neon-stat"
    ;;
  *) echo "unknown mode: $MODE"; exit 1 ;;
esac
