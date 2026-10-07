#!/bin/zsh
# recover_and_verify.sh — 板子重新上电后的一键验证（IO 8ms + 算法 32ms 设计全链）
# 依次：环境检查 → 部署引擎+插件(槽512) → 回归套件 → 欠载/水位 → RTF → DOA 重测素材
# 用法: ./recover_and_verify.sh   （DOA 分析与延迟细测在采集完成后由 analyze 脚本接手）
set -e
cd "$(dirname "$0")/.."

echo "== [0] 板上环境 =="
ssh -o ConnectTimeout=8 nanopi 'uptime; cat /sys/class/thermal/thermal_zone0/temp; systemctl is-active voice-ac108-cfg'
ssh nanopi 'sudo ~/ac108/mic4-record16.sh cfg'

echo "== [1] 部署（引擎 + vshm 插件，shm_voice.h 槽 512 同头）=="
(cd engine && ./deploy.sh | tail -1)
rsync -q shm_voice.h nanopi:/home/i/ngn/shm_voice.h
rsync -q plugin/{pcm_shm.c,pcm_fifo.c,Makefile} nanopi:/home/i/ngn/plugin/
ssh nanopi 'cd /home/i/ngn/plugin && make >/dev/null && sudo make install 2>&1 | tail -1'

echo "== [2] 清旧段 + 回归 =="
ssh nanopi 'pkill -x voice_engine 2>/dev/null; rm -f /tmp/voice_mic.fifo /tmp/voice_spk.fifo /dev/shm/voice_mic /dev/shm/voice_spk'
ssh nanopi 'cd /home/i/ngn/engine && ./test_engine.sh 2>&1 | tail -1'
ssh nanopi 'cd /home/i/ngn && ./test_shm.sh 2>&1 | tail -1'

echo "== [3] 播放欠载/水位/RTF（24s 白噪，低音量 DAC25）=="
ssh nanopi 'amixer -c 0 cset numid=1 25 >/dev/null
  pkill -x voice_engine 2>/dev/null; rm -f /dev/shm/voice_mic /dev/shm/voice_spk
  VSHM_SPK_TRACE=1 nohup /home/i/ngn/engine/voice_engine --capture-fs 16k --device hw:0,0 --stats-interval 1 --mic-order 0,3,2,1 > /tmp/acoustic/engine8ms.log 2>&1 &'
sleep 5
for i in 1 2 3; do ssh nanopi 'aplay -q -D voice_spk_shm /tmp/acoustic/np_white48.wav'; done
ssh nanopi 'pkill -x voice_engine; grep -oE "pb_err=[0-9]+" /tmp/acoustic/engine8ms.log | tail -1'
scp -q nanopi:/tmp/acoustic/engine8ms.log /tmp/acoustic/mac/engine8ms.log
python3 - <<'PY'
import re, numpy as np
rows=[]
for ln in open('/tmp/acoustic/mac/engine8ms.log'):
    m=re.search(r"gap=([\d.]+)ms n=(\d+) delay=(-?\d+)", ln)
    if m: rows.append((float(m.group(1)),int(m.group(2)),int(m.group(3))))
if rows:
    gap,n,dl=map(np.array,zip(*rows))
    print(f"spk-trace: 写块 p50={int(np.median(n))} 间隔 p95={np.percentile(gap,95):.1f}ms "
          f"delay p50={int(np.median(dl))} 带宽[{dl.min()},{dl.max()}] (水位1152, 期望 [1024,1152])")
rtf=[float(m.group(1)) for ln in open('/tmp/acoustic/mac/engine8ms.log')
     if (m:=re.search(r" rtf=([\d.]+)",ln))]
if rtf: print(f"引擎 rtf: max={max(rtf):.3f}（须 <0.9）")
PY

echo "== [4] DOA 重测素材采集（新几何：Mac 与阵有夹角）=="
python3 - <<'PY'
import sys, time, os
sys.path.insert(0,'.')
import importlib.util
spec=importlib.util.spec_from_file_location("rf","run_full_rounds.py")
rf=importlib.util.module_from_spec(spec); spec.loader.exec_module(rf)
os.system("osascript -e 'set volume output volume 55'")
rf.sh(f"ssh {rf.NANOPI} 'pkill -x voice_engine 2>/dev/null; true'")
time.sleep(1)
fails=[]
for name, wav, stim in (("raw_calib","calib2_L.wav",9.6), ("raw_calib_R","calib2_R.wav",9.6),
                        ("raw_doa_Lonly","doa_Lonly.wav",8.0), ("raw_doa_Ronly","doa_Ronly.wav",8.0),
                        ("raw_doa_both","doa_both.wav",8.0)):
    dur = int(stim + 8.4)
    if not rf.run_stim_round(name, dur, wav, stim, name, 8.0):
        fails.append(name)
print("DOA 素材:", "全部达标 ✔" if not fails else f"FAIL {fails}")
print("下一步: python3 analyze_geometry.py  # 三方法互验 + 双源绝对角")
PY
