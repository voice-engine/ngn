#!/bin/zsh
# deploy.sh — Mac 同步 engine/ 到 NanoPi Air 并板上原生编译（含测试素材）
# 用法: ./deploy.sh [--wav-only]
set -e
cd "$(dirname "$0")"
REMOTE=${NGN_REMOTE:-/home/i/ngn/engine}

echo "[1/3] rsync → nanopi:$REMOTE"
ssh nanopi "mkdir -p $REMOTE"
rsync -av --exclude voice_engine --exclude '*.o' --exclude .DS_Store \
      main_engine.cpp va_mock.h va_mock.cpp Makefile test_engine.sh feed_fifo.py gen_test_wav.py nanopi:$REMOTE/

if [ "${1:-}" != "--wav-only" ]; then
  echo "[2/3] 板上编译（含 mock——test_shm.sh 用它起段，漏编会造旧布局段令插件 EINVAL）"
  ssh nanopi "make -C $REMOTE && make -C $REMOTE mock"
else
  echo "[2/3] 跳过编译（--wav-only）"
fi

echo "[3/3] 同步测试素材（存在时）"
for f in test4ch16k.wav test48k.wav; do
  [ -f "$f" ] && rsync -a "$f" nanopi:$REMOTE/ && echo "  $f"
done
echo "done. 板上: cd $REMOTE && ./test_engine.sh"
