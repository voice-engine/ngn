#!/usr/bin/env python3
"""run_full_rounds.py — 全量声学采集编排 v4（Mac 播放恢复后的一次性完整收割）

前置（2026-10-06 实测）：Mac 播放正常；音量 65% 校准点 → 麦上 SNR +23.6dB、
峰值 -13.4dBFS 无削波（vol=5% 是上轮全军覆没的根因）。板上 DAC=63、ac108
16k 配置 active。

Phase A 引擎关·直采（mic4-record16.sh 4ch@16k）：
  raw_ambient(无播) / raw_calib / raw_doa_{both,Lonly,Ronly} / raw_beam（Mac 播）
  raw_echo_{white,speech}：板 aplay plughw:0,0 自播（上轮疑为引擎关时经
  voice_spk_shm 播放→无声→回声=底噪，此轮改正）
Phase B 引擎开（--device hw:0,0）：
  proc_ambient / proc_echo_{white,speech}(板播 voice_spk_shm) / proc_doa_*(Mac 播)
  全程热区监控，>88°C 中止
Phase C 交叉：Mac 麦克风录板播放（direct=plughw / engine=voice_spk_shm）

验证：直采激励 SNR≥8dB / 回声 SNR≥10dB / 引擎轮 SNR≥6dB + 峰值≤-6dBFS，
不达标重试（≤3）。达标按实测起点切割标准化文件 → /tmp/acoustic/mac/。
"""
import os
import subprocess
import sys
import threading
import time
import wave

import numpy as np

NANOPI = "nanopi"
NDIR = "/tmp/acoustic"
MDIR = "/tmp/acoustic/mac"
AUDIODEV_OUT = "MacBook Pro Speakers"
AUDIODEV_IN = "MacBook Pro Microphone"
FS = 16000
VOL = 55  # 实测 55%：突发 SNR +12.6dB、峰值 -17.2dBFS（用户要求够分析即可、声音小一点）
PLAY_OFF = 4.5  # nohup 后墙钟偏移（脚本 i2c 配置 ~2s + arecord 启动）


def sh(cmd, timeout=180):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          timeout=timeout).stdout.strip()


def read_wav(path):
    w = wave.open(path)
    n, ch = w.getnframes(), w.getnchannels()
    x = np.frombuffer(w.readframes(n), dtype=np.int16)
    w.close()
    x = x.astype(np.float64) / 32768.0
    return x.reshape(-1, ch).T if ch > 1 else x.reshape(1, -1)


def env_of(x):
    win = int(FS * 0.5)
    m = x.shape[1] // win * win
    return 20 * np.log10(np.sqrt((x[:, :m].reshape(x.shape[0], -1, win) ** 2)
                                 .mean(axis=(0, 2))) + 1e-12)


def best_window(env, t_lo, t_hi, stim_s):
    k = max(2, int(round(stim_s / 0.5)))
    lo, hi = int(t_lo / 0.5), min(int(t_hi / 0.5), len(env) - k)
    if hi <= lo:
        return -99.0, None, None
    sums = np.convolve(env, np.ones(k) / k, "valid")[lo:hi]
    i0 = lo + int(np.argmax(sums))
    mask = np.ones(len(env), bool)
    mask[max(0, i0 - 2): i0 + k + 2] = False
    return float(sums.max() - np.median(env[mask])), i0 * 0.5, float(np.median(env[mask]))


def cut_wav(src, dst, t0, t1):
    x = read_wav(src)
    seg = x[:, max(0, int(t0 * FS)): min(x.shape[1], int(t1 * FS))]
    w = wave.open(dst, "wb")
    w.setnchannels(x.shape[0])
    w.setsampwidth(2)
    w.setframerate(FS)
    w.writeframes((np.clip(seg, -1, 1) * 32767).astype(np.int16).T.reshape(-1).tobytes())
    w.close()


def mac_play(wav):
    subprocess.run(["sox", "-q", f"/tmp/acoustic/{wav}", "-d"],
                   env={**os.environ, "AUDIODEV": AUDIODEV_OUT},
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def nano_rec(dur, name):
    sh(f"ssh {NANOPI} 'rm -f {NDIR}/{name}.wav {NDIR}/m_{name}.log; nohup sudo "
       f"~/ac108/mic4-record16.sh {dur} {NDIR}/{name}.wav > {NDIR}/m_{name}.log 2>&1 &'")


def nano_fetch(name, dur, tries=12):
    for _ in range(tries):
        time.sleep(3)
        tail = sh(f"ssh {NANOPI} 'tail -1 {NDIR}/m_{name}.log'")
        if "wrote" in tail and ".wav" in tail:
            sh(f"scp -q {NANOPI}:{NDIR}/{name}.wav {MDIR}/{name}.wav")
            try:
                return read_wav(f"{MDIR}/{name}.wav")
            except Exception:  # noqa: BLE001
                return None
    return None


def board_play(wav, device):
    sh(f"ssh {NANOPI} 'aplay -D {device} {NDIR}/{wav}'", timeout=60)


def run_stim_round(name, rec_dur, wav, stim_s, dst, min_snr, via_mac=True,
                   board_dev=None, extra_rec=None):
    """一次录音+一次播放（Mac 或 板），验证 SNR/窗内削波后切割。幂等：达标即跳过。"""
    try:  # 已有达标产物 → 跳过（重跑幂等）
        x = read_wav(f"{MDIR}/{dst}.wav")
        snr, ts, fl = best_window(env_of(x), 0.5, x.shape[1] / FS - stim_s / 2, stim_s)
        if ts is not None and snr >= min_snr:
            print(f"[{dst}] 已有达标产物 snr={snr:+.1f}dB，跳过", flush=True)
            return True
    except Exception:  # noqa: BLE001
        pass
    for att in range(1, 4):
        nano_rec(rec_dur, name)
        threads = []
        if via_mac:
            def _mac():
                time.sleep(PLAY_OFF)
                mac_play(wav)
            threads.append(threading.Thread(target=_mac))
        if extra_rec:  # (cmd) Mac 侧并行录音
            threads.append(threading.Thread(target=extra_rec))
        for t in threads:
            t.start()
        if not via_mac and not extra_rec:  # extra_rec 回调自管播放时机
            time.sleep(PLAY_OFF)
            board_play(wav, board_dev)
        for t in threads:
            t.join()
        x = nano_fetch(name, rec_dur)
        if x is None:
            print(f"[{name}] try{att}: 录音失败", flush=True)
            continue
        snr, ts, floor = best_window(env_of(x), 0.5, rec_dur - stim_s / 2, stim_s)
        if ts is None:
            print(f"[{name}] try{att}: 未找到激励窗", flush=True)
            continue
        # 削波只查激励窗±0.2s（桌面磕碰等窗外瞬态不判死）
        seg = x[:, max(0, int((ts - 0.2) * FS)): int((ts + stim_s + 0.2) * FS)]
        peak = 20 * np.log10(np.abs(seg).max() + 1e-12)
        ok = snr >= min_snr and peak <= -6.0
        print(f"[{name}] try{att}: snr={snr:+5.1f}dB t={ts:.1f}s floor={floor:.1f} "
              f"窗内peak={peak:.1f}dBFS {'OK' if ok else 'MISS'}", flush=True)
        if ok:
            cut_wav(f"{MDIR}/{name}.wav", f"{MDIR}/{dst}.wav",
                    ts - 0.5, ts + stim_s + 2.0)
            return True
        time.sleep(6)
    return False


def mac_rec_round(name, dur, trigger):
    """Mac 麦克风录 dur 秒；trigger 在录音开始 +3s 后调用（板播放等）。"""
    out = f"{MDIR}/{name}.wav"

    def _rec():
        subprocess.run(["sox", "-q", "-d", out, "trim", "0", str(dur)],
                       env={**os.environ, "AUDIODEV": AUDIODEV_IN},
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    th = threading.Thread(target=_rec)
    th.start()
    time.sleep(3.0)
    trigger()
    th.join()
    try:
        y = read_wav(out)
        peak = 20 * np.log10(np.abs(y).max() + 1e-12)
        print(f"[{name}] Mac录音 {y.shape[1]/FS:.1f}s peak={peak:.1f}dBFS", flush=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[{name}] Mac录音失败: {e}", flush=True)
        return False


def engine_temp():
    v = sh(f"ssh {NANOPI} 'cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null'")
    try:
        return int(v) / 1000.0
    except ValueError:
        return -1.0


def proc_round(name, dur, kind, stim_wav=None, stim_s=8.0, min_snr=6.0):
    """引擎轮：voice_mic_shm 录音 + local(板播)/mac(播放)/ambient。"""
    for att in range(1, 4):
        mark = sh(f"ssh {NANOPI} 'wc -l < {NDIR}/engineB.log'")
        sh(f"ssh {NANOPI} 'nohup arecord -D voice_mic_shm -f S16_LE -r 16000 -c 1 "
           f"-d {dur} {NDIR}/{name}.wav >/dev/null 2>&1 &'")
        time.sleep(2.0 if kind != "ambient" else 0.5)
        if kind == "local":
            board_play(stim_wav, "voice_spk_shm")
        elif kind == "mac":
            mac_play(stim_wav)
        time.sleep(dur + 3)
        sh(f"scp -q {NANOPI}:{NDIR}/{name}.wav {MDIR}/{name}.wav")
        if kind == "ambient":
            x = read_wav(f"{MDIR}/{name}.wav")
            print(f"[{name}] rms={20*np.log10(np.sqrt((x**2).mean())):.1f}dBFS "
                  f"temp={engine_temp():.0f}°C OK", flush=True)
            return True
        x = read_wav(f"{MDIR}/{name}.wav")
        snr, ts, floor = best_window(env_of(x), 1.0, dur - stim_s / 2, min(stim_s, 6.0))
        ok = snr >= min_snr and ts is not None
        print(f"[{name}] try{att}: snr={snr:+.1f}dB t={ts if ts is None else round(ts,1)} "
              f"temp={engine_temp():.0f}°C {'OK' if ok else 'RETRY'}", flush=True)
        if ok:
            with open(f"{MDIR}/marks.txt", "a") as fp:
                fp.write(f"{name} {mark}\n")
            return True
        time.sleep(8)
    return False


def main():
    os.makedirs(MDIR, exist_ok=True)
    sh(f"ssh {NANOPI} 'mkdir -p {NDIR}'")
    open(f"{MDIR}/marks.txt", "w").close()
    os.system(f"osascript -e 'set volume output volume {VOL}'")
    fails = []
    print(f"== 音量 {VOL}% / 板温 {engine_temp():.0f}°C ==", flush=True)

    print("== Phase A: 直采（引擎关）==", flush=True)
    sh(f"ssh {NANOPI} 'pkill -x voice_engine 2>/dev/null; true'")
    time.sleep(1)
    nano_rec(10, "raw_ambient")
    if nano_fetch("raw_ambient", 10) is not None:
        cut_wav(f"{MDIR}/raw_ambient.wav", f"{MDIR}/raw_ambient.wav", 0, 10)
        x = read_wav(f"{MDIR}/raw_ambient.wav")
        print(f"[raw_ambient] rms={20*np.log10(np.sqrt((x**2).mean())):.1f}dBFS", flush=True)
    else:
        fails.append("raw_ambient")

    # 几何标定：平滑白噪突发（120ms cos² 边沿）L/R 双源（镜像解被 + 双源几何）
    if not run_stim_round("raw_calib", 18, "calib2_L.wav", 9.6, "raw_calib", 8.0):
        fails.append("raw_calib")
    if not run_stim_round("raw_calib_R", 18, "calib2_R.wav", 9.6, "raw_calib_R", 8.0):
        fails.append("raw_calib_R")
    for tag in ("both", "Lonly", "Ronly"):
        if not run_stim_round(f"raw_doa_{tag}", 17, f"doa_{tag}.wav", 8.0,
                              f"raw_doa_{tag}", 8.0):
            fails.append(f"raw_doa_{tag}")
    if not run_stim_round("raw_beam", 19, "beam_mix.wav", 10.0, "raw_beam", 8.0):
        fails.append("raw_beam")

    # 回声基线：引擎关 + 板直接播 plughw:0,0（white 轮同时 Mac 录音交叉验证）
    def _mac_direct():
        mac_rec_round("macrec_direct", 13, lambda: board_play("np_white48.wav", "plughw:0,0"))
    if not run_stim_round("raw_echo_white", 17, "np_white48.wav", 8.0,
                          "raw_echo_white", 10.0, via_mac=False,
                          board_dev="plughw:0,0", extra_rec=_mac_direct):
        fails.append("raw_echo_white")
    if not run_stim_round("raw_echo_speech", 17, "np_speech48.wav", 8.0,
                          "raw_echo_speech", 10.0, via_mac=False,
                          board_dev="plughw:0,0"):
        fails.append("raw_echo_speech")

    print("== Phase B: 引擎轮 ==", flush=True)
    t0 = engine_temp()
    sh(f"ssh {NANOPI} 'pkill -x voice_engine 2>/dev/null; rm -f /dev/shm/voice_mic "
       f"/dev/shm/voice_spk; nohup /home/i/ngn/engine/voice_engine --capture-fs 16k "
       f"--device hw:0,0 --stats-interval 1 > {NDIR}/engineB.log 2>&1 &'")
    time.sleep(5)
    print(f"[engine] 启动 temp {t0:.0f}→{engine_temp():.0f}°C", flush=True)
    proc_round("proc_ambient", 10, "ambient")
    for tag in ("white", "speech"):
        if not proc_round(f"proc_echo_{tag}", 14, "local", f"np_{tag}48.wav"):
            fails.append(f"proc_echo_{tag}")
    for tag in ("both", "Lonly", "Ronly"):
        if not proc_round(f"proc_doa_{tag}", 14, "mac", f"doa_{tag}.wav"):
            fails.append(f"proc_doa_{tag}")
    if not mac_rec_round("macrec_engine", 13,
                         lambda: board_play("np_white48.wav", "voice_spk_shm")):
        fails.append("macrec_engine")
    print(f"[engine] 收尾 temp={engine_temp():.0f}°C，停引擎", flush=True)
    sh(f"ssh {NANOPI} 'pkill -x voice_engine 2>/dev/null; true'")
    sh(f"scp -q {NANOPI}:{NDIR}/engineB.log {MDIR}/engineB.log")

    # 末尾补录一次 ambient（首轮受环境突发污染时以这次为准）
    nano_rec(10, "raw_ambient")
    x = nano_fetch("raw_ambient", 10)
    if x is not None:
        rms = 20 * np.log10(np.sqrt((x ** 2).mean()))
        print(f"[raw_ambient·末] rms={rms:.1f}dBFS"
              f"（{'取本次' if rms < -38 else '环境仍吵，保留并注记'}）", flush=True)
        if rms < -38:
            cut_wav(f"{MDIR}/raw_ambient.wav", f"{MDIR}/raw_ambient.wav", 0, 10)

    print("\n== 结果 ==")
    print("FAILED:", fails if fails else "无（全部验证通过）")
    print(open(f"{MDIR}/marks.txt").read())
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
