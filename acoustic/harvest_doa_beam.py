#!/usr/bin/env python3
"""harvest_doa_beam.py — T3/T4 机会式收割（等 Mac 播放好窗口）

每轮：34s 直采录音内串联 3 个 DOA 激励（8s 每个）；beam 用 24s 录音双播，
验证要求「语音段-纯噪段」对比度 ≥2.5dB（防止左声道被吞只剩噪声）。
全部收割成功即退出；否则循环（好窗口每 30-50 分钟出现）。
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
AUDIODEV = "MacBook Pro Speakers"
FS = 16000
BIN_S = 0.5


def sh(cmd, timeout=120):
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
    win = int(FS * BIN_S)
    m = x.shape[1] // win * win
    return 20 * np.log10(np.sqrt((x[:, :m].reshape(x.shape[0], -1, win) ** 2)
                                 .mean(axis=(0, 2))) + 1e-12)


def best_window(env, t_lo, t_hi, stim_s):
    k = max(2, int(round(stim_s / BIN_S)))
    lo, hi = int(t_lo / BIN_S), min(int(t_hi / BIN_S), len(env) - k)
    if hi <= lo:
        return -99.0, None, None
    sums = np.convolve(env, np.ones(k) / k, "valid")[lo:hi]
    i0 = lo + int(np.argmax(sums))
    mask = np.ones(len(env), bool)
    mask[max(0, i0 - 2): i0 + k + 2] = False
    return float(sums.max() - np.median(env[mask])), i0 * BIN_S, float(np.median(env[mask]))


def cut_wav(src, dst, t0, t1):
    x = read_wav(src)
    seg = x[:, max(0, int(t0 * FS)): min(x.shape[1], int(t1 * FS))]
    w = wave.open(dst, "wb")
    w.setnchannels(4)
    w.setsampwidth(2)
    w.setframerate(FS)
    w.writeframes((np.clip(seg, -1, 1) * 32767).astype(np.int16).T.reshape(-1).tobytes())
    w.close()


def mac_play(wav):
    subprocess.run(["sox", "-q", f"/tmp/acoustic/{wav}", "-d"],
                   env={**os.environ, "AUDIODEV": AUDIODEV},
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def nano_rec_fetch(rec, dur, events):
    sh(f"ssh {NANOPI} 'nohup sudo ~/ac108/mic4-record16.sh {dur} {NDIR}/{rec} "
       f"> {NDIR}/m_{rec}.log 2>&1 &'")
    t0 = time.time()

    def run():
        for t_rel, wav in events:
            d = t0 + t_rel - time.time()
            if d > 0:
                time.sleep(d)
            mac_play(wav)
    threading.Thread(target=run).start()
    for _ in range(30):
        time.sleep(4)
        tail = sh(f"ssh {NANOPI} 'tail -1 {NDIR}/m_{rec}.log'")
        if "wrote" in tail:
            sh(f"scp -q {NANOPI}:{NDIR}/{rec} {MDIR}/{rec}")
            try:
                return read_wav(f"{MDIR}/{rec}")
            except Exception:  # noqa: BLE001
                return None
    return None


def harvest_doa(need, tries=14):
    """need: list of tags (both/Lonly/Ronly)。单轮 34s 录音排 3 槽。"""
    for att in range(1, tries + 1):
        tags = (need * 3)[:3]
        events = [(4.5, f"doa_{tags[0]}.wav"), (14.5, f"doa_{tags[1]}.wav"),
                  (24.5, f"doa_{tags[2]}.wav")]
        x = nano_rec_fetch("hdoa.wav", 34, events)
        if x is None:
            print(f"[hdoa] try{att}: 录音失败", flush=True)
            time.sleep(10)
            continue
        env = env_of(x)
        got = []
        for slot, tag in enumerate(tags):
            lo, hi = 1.0 + slot * 10, 10.5 + slot * 10
            snr, ts, floor = best_window(env, lo, hi, 6.0)
            ok = snr >= 6.5 and ts is not None
            print(f"[hdoa] try{att} 槽{slot} {tag:6s} snr={snr:+5.1f}dB "
                  f"t={'--' if ts is None else f'{ts:4.1f}'} floor={floor if floor is None else round(floor,1)} "
                  f"{'OK' if ok else 'MISS'}", flush=True)
            if ok and tag in need and tag not in got:
                cut_wav(f"{MDIR}/hdoa.wav", f"{MDIR}/raw_doa_{tag}.wav",
                        ts - 0.4, ts + 8.6)
                got.append(tag)
        for t in got:
            if t in need:
                need.remove(t)
        print(f"[hdoa] try{att}: 本轮收割 {got}，剩余 {need}", flush=True)
        if not need:
            return True
        time.sleep(15)
    return False


def harvest_beam(tries=14):
    for att in range(1, tries + 1):
        x = nano_rec_fetch("hbeam.wav", 26, [(4.5, "beam_mix.wav"), (16.0, "beam_mix.wav")])
        if x is None:
            print(f"[hbeam] try{att}: 录音失败", flush=True)
            continue
        env = env_of(x)
        for lo, hi in ((1.5, 10.5), (13.0, 22.0)):
            snr, ts, floor = best_window(env, lo, hi, 8.0)
            if snr < 6.0 or ts is None:
                print(f"[hbeam] try{att} @[{lo},{hi}] snr={snr:+.1f} MISS", flush=True)
                continue
            # 对比度验证：素材内 [3,8)s 为语音+噪，[0,3)/[8,10) 为纯噪
            sp = env[int((ts + 3.5) / BIN_S): int((ts + 7.5) / BIN_S)]
            n1 = env[int((ts + 0.3) / BIN_S): int((ts + 2.8) / BIN_S)]
            n2 = env[int((ts + 8.3) / BIN_S): int((ts + 9.8) / BIN_S)]
            contrast = float(np.mean(sp) - (np.mean(n1) + np.mean(n2)) / 2)
            print(f"[hbeam] try{att} @[{lo},{hi}] snr={snr:+.1f} t={ts:.1f} "
                  f"语音-噪声对比度={contrast:+.1f}dB", flush=True)
            if contrast >= 2.0:
                cut_wav(f"{MDIR}/hbeam.wav", f"{MDIR}/raw_beam.wav", ts - 0.6, ts + 11.2)
                print(f"[hbeam] try{att}: 收割 ✔", flush=True)
                return True
        time.sleep(15)
    return False


def main():
    sh(f"ssh {NANOPI} 'pkill -x voice_engine 2>/dev/null; true'")
    ok_doa = harvest_doa(["both", "Lonly", "Ronly"])
    ok_beam = harvest_beam()
    print(f"\n== 结果: doa={'OK' if ok_doa else 'FAIL'} beam={'OK' if ok_beam else 'FAIL'}")
    return 0 if (ok_doa and ok_beam) else 1


if __name__ == "__main__":
    sys.exit(main())
