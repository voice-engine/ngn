#!/usr/bin/env python3
"""run_verified_rounds.py — Mac 侧自愈式采集编排器（窗口收割机 v3）

背景：Mac 播放被 coreaudiod 间歇吞（设备层，第三方驱动注入；无 sudo 无法重启）。
好窗口每 10~40 分钟出现、持续 1~3 分钟。对策：长录音串联多激励（单次播放 ≤10s），
逐激励定点验证（最优连续窗 vs 窗外中位底噪），达标录音按实测起点切割成标准文件；
等待远程脚本 log 出现 "wrote" 再取文件（避免半截解码文件）。

轮次：
  A1 32s 直采：calib@+4.5 → doa_both@+13.5 → doa_Lonly@+23.5（≥2/3 过 → 收割）
  A2 32s 直采：doa_Ronly@+4.5 → beam_mix@+16（各自 ≥6dB → 收割）
  引擎轮：proc_ambient(无播) / proc_echo_{white,speech}(板端 aplay)
          / proc_doa_{both,Lonly,Ronly}(Mac sox，重试)
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
    win = int(FS * BIN_S)
    m = x.shape[1] // win * win
    return 20 * np.log10(np.sqrt((x[:, :m].reshape(x.shape[0], -1, win) ** 2)
                                 .mean(axis=(0, 2))) + 1e-12)


def stim_window(env, t_lo, t_hi, stim_s):
    """[t_lo,t_hi] 内找时长 stim_s 的最优连续窗 → (snr_db, t_start, floor_db)。"""
    k = max(2, int(round(stim_s / BIN_S)))
    lo, hi = int(t_lo / BIN_S), min(int(t_hi / BIN_S), len(env) - k)
    if hi <= lo:
        return -99.0, None, None
    sums = np.convolve(env, np.ones(k) / k, "valid")[lo:hi]
    i0 = lo + int(np.argmax(sums))
    mask = np.ones(len(env), bool)
    mask[max(0, i0 - 2): i0 + k + 2] = False
    floor = float(np.median(env[mask]))
    return float(sums.max() - floor), i0 * BIN_S, floor


def cut_wav(src, dst, t0, t1):
    x = read_wav(src)
    seg = x[:, max(0, int(t0 * FS)): min(x.shape[1], int(t1 * FS))]
    w = wave.open(dst, "wb")
    w.setnchannels(4)
    w.setsampwidth(2)
    w.setframerate(FS)
    w.writeframes((np.clip(seg, -1, 1) * 32767).astype(np.int16).T.reshape(-1).tobytes())
    w.close()


def nano_rec(dur, out):
    sh(f"ssh {NANOPI} 'nohup sudo ~/ac108/mic4-record16.sh {dur} {NDIR}/{out} "
       f"> {NDIR}/m_{out}.log 2>&1 &'")


def nano_fetch(out, dur, tries=8):
    """等远程 log 出现 wrote 再 scp；返回本地路径或 None。"""
    for _ in range(tries):
        time.sleep(4)
        tail = sh(f"ssh {NANOPI} 'tail -1 {NDIR}/m_{out}.log'")
        if "wrote" in tail and ".wav" in tail:
            sh(f"scp -q {NANOPI}:{NDIR}/{out} {MDIR}/{out}")
            try:
                read_wav(f"{MDIR}/{out}")
                return f"{MDIR}/{out}"
            except Exception:  # noqa: BLE001
                pass
        time.sleep(4)
    return None


def mac_play(wav):
    subprocess.run(["sox", "-q", f"/tmp/acoustic/{wav}", "-d"],
                   env={**os.environ, "AUDIODEV": AUDIODEV},
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def schedule(t0, events):
    def run():
        for t_rel, wav in events:
            d = t0 + t_rel - time.time()
            if d > 0:
                time.sleep(d)
            mac_play(wav)
    th = threading.Thread(target=run)
    th.start()
    return th


def megaround(tag, rec_dur, plan, need_ok, tries=9, min_snr=6.5):
    """plan: [(name, wav, play_off, t_lo, t_hi, stim_s, dst 或 None)]"""
    for att in range(1, tries + 1):
        nano_rec(rec_dur, f"mega{tag}.wav")
        t0 = time.time()
        schedule(t0, [(off, wav) for (_, wav, off, _, _, _) in plan]).join()
        path = nano_fetch(f"mega{tag}.wav", rec_dur)
        if not path:
            print(f"[mega{tag}] try{att}: 录音/解码未就绪，重试", flush=True)
            time.sleep(8)
            continue
        env = env_of(read_wav(path))
        n_ok, results = 0, []
        for name, wav, off, (lo, hi), stim_s, dst in plan:
            snr, ts, floor = stim_window(env, lo, hi, stim_s)
            ok = snr >= min_snr and ts is not None
            n_ok += ok
            results.append((name, snr, ts, floor, stim_s, dst, ok))
            print(f"[mega{tag}] try{att} {name:10s} snr={snr:+5.1f}dB "
                  f"t={'--' if ts is None else f'{ts:4.1f}'} "
                  f"floor={'--' if floor is None else f'{floor:5.1f}'} "
                  f"{'OK' if ok else 'MISS'}", flush=True)
        if n_ok >= need_ok:
            for name, snr, ts, floor, stim_s, dst, ok in results:
                if ok and dst:
                    cut_wav(path, f"{MDIR}/{dst}.wav", ts - 0.5, ts + stim_s + 2.0)
            print(f"[mega{tag}] try{att}: 收割 {n_ok}/{len(plan)} ✔", flush=True)
            return n_ok
        time.sleep(18)
    return 0


def round_engine(name, dur, kind, stim_wav=None, stim_s=6.0, tries=10, min_snr=6.0):
    for att in range(1, tries + 1):
        mark = sh(f"ssh {NANOPI} 'wc -l < {NDIR}/engineB.log'")
        sh(f"ssh {NANOPI} 'nohup arecord -D voice_mic_shm -f S16_LE -r 16000 -c 1 "
           f"-d {dur} {NDIR}/{name} >/dev/null 2>&1 &'")
        time.sleep(2.0 if kind != "ambient" else 0.5)
        if kind == "local":
            sh(f"ssh {NANOPI} 'aplay -D voice_spk_shm {NDIR}/{stim_wav}'")
        elif kind == "mac":
            mac_play(stim_wav)
        time.sleep(dur + 4)
        sh(f"scp -q {NANOPI}:{NDIR}/{name} {MDIR}/{name}")
        if kind == "ambient":
            x = read_wav(f"{MDIR}/{name}")
            print(f"[{name}] rms={20*np.log10(np.sqrt((x**2).mean())):.1f}dBFS OK",
                  flush=True)
            return True
        env = env_of(read_wav(f"{MDIR}/{name}"))
        snr, ts, floor = stim_window(env, 1.0, dur - stim_s / 2, min(stim_s, 6.0))
        ok = snr >= min_snr and ts is not None
        print(f"[{name}] try{att}: snr={snr:+.1f}dB "
              f"({'--' if ts is None else f't={ts:.1f}'}) {'OK' if ok else 'RETRY'}",
              flush=True)
        if ok:
            with open(f"{MDIR}/marks.txt", "a") as fp:
                fp.write(f"{name} {mark}\n")
            return True
        time.sleep(12)
    return False


def main():
    os.makedirs(MDIR, exist_ok=True)
    sh(f"ssh {NANOPI} 'mkdir -p {NDIR}'")
    open(f"{MDIR}/marks.txt", "w").close()
    fails = []

    print("== Phase A: 直采（窗口收割）==", flush=True)
    sh(f"ssh {NANOPI} 'pkill -x voice_engine 2>/dev/null; true'")
    time.sleep(1)
    nano_rec(10, "raw_ambient")
    p = nano_fetch("raw_ambient", 10)
    if p:
        x = read_wav(p)
        print(f"[raw_ambient] rms={20*np.log10(np.sqrt((x**2).mean())):.1f}dBFS",
              flush=True)
    else:
        fails.append("raw_ambient")

    plan_a1 = [
        ("calib", "calib_L.wav", 4.5, (1.0, 9.0), 4.0, "raw_calib"),
        ("doa_both", "doa_both.wav", 13.5, (10.0, 20.0), 6.0, "raw_doa_both"),
        ("doa_Lonly", "doa_Lonly.wav", 23.5, (20.0, 30.5), 6.0, "raw_doa_Lonly"),
    ]
    n = megaround("A1", 32, plan_a1, need_ok=2)
    if n < 2:
        fails.append(f"megaA1({n}/3)")
    plan_a2 = [
        ("doa_Ronly", "doa_Ronly.wav", 4.5, (1.0, 9.0), 6.0, "raw_doa_Ronly"),
        ("beam", "beam_mix.wav", 16.0, (12.0, 23.0), 8.0, "raw_beam"),
    ]
    if megaround("A2", 32, plan_a2, need_ok=1) < 1:
        fails.append("megaA2")

    print("== Phase B: 引擎轮 ==", flush=True)
    sh(f"ssh {NANOPI} 'pkill -x voice_engine 2>/dev/null; rm -f /dev/shm/voice_mic "
       f"/dev/shm/voice_spk; nohup /home/i/ngn/engine/voice_engine --capture-fs 16k "
       f"--device hw:0,0 --stats-interval 1 > {NDIR}/engineB.log 2>&1 &'")
    time.sleep(5)
    round_engine("proc_ambient", 10, "ambient")
    if not round_engine("proc_echo_white", 12, "local", "np_white48.wav"):
        fails.append("proc_echo_white")
    if not round_engine("proc_echo_speech", 12, "local", "np_speech48.wav"):
        fails.append("proc_echo_speech")
    for tag in ("both", "Lonly", "Ronly"):
        if not round_engine(f"proc_doa_{tag}", 12, "mac", f"doa_{tag}.wav"):
            fails.append(f"proc_doa_{tag}")
    sh(f"ssh {NANOPI} 'pkill -x voice_engine 2>/dev/null; true'")
    sh(f"scp -q {NANOPI}:{NDIR}/engineB.log {MDIR}/engineB.log")

    print("\n== 结果 ==")
    print("FAILED:", fails if fails else "无（全部验证通过）")
    print(open(f"{MDIR}/marks.txt").read())
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
