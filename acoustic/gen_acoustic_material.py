#!/usr/bin/env python3
"""gen_acoustic_material.py — 首批真实声学测试素材生成（Mac 侧）

深夜桌面环境音量约束：全部素材 RMS 有效幅度 = 0.05（≈ -26.0 dBFS）。
Mac 播放：afplay（stereo wav，左/右声道独立源）+ 系统音量 5%。
nanopi 播放：aplay -D voice_spk_shm（48k mono S16，引擎→DA→喇叭）。

产物（/tmp/acoustic/）：
  trial.wav      3s   stereo  L=白噪（试跑确认麦上有信号）
  calib_L.wav    6s   stereo  L=白噪突发 0.5s on/0.5s off ×6（几何/麦序标定）
  doa_both.wav   8s   stereo  L=带限白噪(300-3400Hz) + R=语音（双源 DOA）
  doa_Lonly.wav  8s   stereo  仅 L=带限白噪
  doa_Ronly.wav  8s   stereo  仅 R=语音
  beam_mix.wav   10s  stereo  L=语音(仅[3,8)s) + R=连续带限白噪（波束增益；
                        0-3s/8-10s 为纯噪声段，3-8s 为语音+噪声段）
  np_white48.wav 8s   mono 48k 白噪（nanopi 自播 AEC）
  np_speech48.wav 8s  mono 48k 语音（nanopi 自播 AEC）

语音源：macOS `say` 中文 TTS（Eddy）；失败则回退 voice/engine/far48.wav（类语音合成）。
"""
import os
import subprocess
import sys
import wave

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

FS = 48000
RMS = 0.05  # -26.02 dBFS
OUT = "/tmp/acoustic"
FALLBACK_SPEECH = os.path.join(os.path.dirname(__file__), "..", "engine", "far48.wav")
SAY_TEXT = (
    "声学测试开始。今天我们测量麦克风阵列的到达角分辨能力。"
    "第一个声源在左边，第二个声源在右边。请记录每个通道的信号强度。"
    "波束成形可以增强目标方向的声音，同时抑制其他方向的噪声。"
    "回声消除利用参考信号估计扬声器到麦克风的声学路径。"
    "房间混响会让相关峰变宽，但相位信息仍然可用。"
    "测试结束，感谢收听。"
)


def bandlimit(x, lo, hi):
    sos = butter(8, [lo, hi], btype="band", fs=FS, output="sos")
    y = sosfilt(sos, x)
    # 去滤波器启动瞬态
    n = FS // 10
    return y[n:]


def rms(x):
    return float(np.sqrt(np.mean(x**2) + 1e-20))


def norm_rms(x, target=RMS):
    return x * (target / rms(x))


def fade(x, ms=5):
    n = int(FS * ms / 1000)
    if n > 0 and len(x) > 2 * n:
        w = np.linspace(0, 1, n)
        x[:n] *= w
        x[-n:] *= w[::-1]
    return x


def write_wav(path, data, ch=1):
    data = np.clip(data, -0.98, 0.98)
    if ch == 2:
        inter = np.empty(data.shape[0] * 2)
        inter[0::2] = data[:, 0]
        inter[1::2] = data[:, 1]
    else:
        inter = data
    pcm = (inter * 32767.0).astype(np.int16)
    w = wave.open(path, "wb")
    w.setnchannels(ch)
    w.setsampwidth(2)
    w.setframerate(FS)
    w.writeframes(pcm.tobytes())
    w.close()
    if ch == 2:
        r_db = "+".join(f"{20*np.log10(rms(data[:, i])):+.1f}" for i in range(ch))
    else:
        r_db = f"{20*np.log10(rms(data)):+.1f}"
    print(f"  {os.path.basename(path):18s} {len(pcm)//ch/FS:5.1f}s {ch}ch "
          f"rms[L/R]={r_db}dBFS "
          f"peak={20*np.log10(np.abs(inter).max()+1e-12):+.1f}dBFS")


def make_speech(dur):
    """say 中文 TTS → dur 秒连续语音，RMS 归一。失败回退 far48.wav。"""
    raw_aiff = os.path.join(OUT, "say_raw.aiff")
    raw = os.path.join(OUT, "say_raw.wav")
    try:
        # 注意：say 短名 "-v Eddy" 会命中未下载完成的增值语音→静音，必须用完整名；
        # 且 say 直出 .wav 对长文本不可靠，稳定路径是 AIFF + sox 转换。
        subprocess.run(
            ["say", "-v", "Tingting", "-o", raw_aiff, SAY_TEXT],
            check=True, timeout=90,
        )
        subprocess.run(
            ["sox", raw_aiff, "-r", str(FS), "-b", "16", raw],
            check=True, timeout=60,
        )
        sr, d = wavfile.read(raw)
        x = d.astype(np.float64) / 32768.0
        if x.ndim > 1:
            x = x.mean(axis=1)
        assert sr == FS and len(x) > FS, f"say 输出异常: sr={sr} n={len(x)}"
        assert np.abs(x).max() > 1e-3, "say 输出为静音"
        src = "say Tingting zh_CN"
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] say 失败({e})，回退 {FALLBACK_SPEECH}")
        sr, d = wavfile.read(FALLBACK_SPEECH)
        x = d.astype(np.float64) / 32768.0
        if sr != FS:
            x = np.repeat(x, FS // sr, axis=0)  # 粗降/升采样（48k 源通常已对）
        src = "far48.wav (fallback)"
    # 去首尾静音
    env = np.abs(x)
    th = env.max() * 0.02
    idx = np.where(env > th)[0]
    x = x[idx[0]: idx[-1] + 1] if len(idx) > FS else x
    # 补齐/截断到 dur
    n = int(dur * FS)
    if len(x) >= n:
        x = x[:n]
    else:
        x = np.concatenate([x, np.zeros(n - len(x))])
    print(f"  speech 源: {src}, 截取 {dur}s")
    return norm_rms(fade(x))


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(20261005)

    print("[1] 白噪/带限白噪")
    white3 = fade(norm_rms(rng.standard_normal(int(3.0 * FS))))
    white8 = fade(norm_rms(rng.standard_normal(int(8.0 * FS))))
    band8 = fade(norm_rms(bandlimit(rng.standard_normal(int(8.3 * FS)), 300, 3400))[: int(8.0 * FS)])
    band10 = fade(norm_rms(bandlimit(rng.standard_normal(int(10.3 * FS)), 300, 3400))[: int(10.0 * FS)])

    print("[2] 语音")
    sp8 = make_speech(8.0)
    sp5 = make_speech(5.0)

    print("[3] 落盘 stereo（Mac afplay）")
    st = lambda l, r: np.stack([l, r], axis=1)  # noqa: E731
    write_wav(f"{OUT}/trial.wav", st(white3, np.zeros_like(white3)), 2)
    # 标定：0.5s on / 0.5s off ×6 = 6s
    bursts = []
    for _ in range(6):
        bursts.append(norm_rms(rng.standard_normal(int(0.5 * FS))))
        bursts.append(np.zeros(int(0.5 * FS)))
    cal = fade(np.concatenate(bursts))
    write_wav(f"{OUT}/calib_L.wav", st(cal, np.zeros_like(cal)), 2)
    write_wav(f"{OUT}/doa_Lonly.wav", st(band8, np.zeros_like(band8)), 2)
    write_wav(f"{OUT}/doa_Ronly.wav", st(np.zeros_like(sp8), sp8), 2)
    write_wav(f"{OUT}/doa_both.wav", st(band8, sp8), 2)
    beam_l = np.zeros(int(10.0 * FS))
    beam_l[int(3.0 * FS): int(8.0 * FS)] = sp5
    write_wav(f"{OUT}/beam_mix.wav", st(beam_l, band10), 2)

    print("[4] 落盘 mono（nanopi aplay -D voice_spk_shm，48k）")
    write_wav(f"{OUT}/np_white48.wav", white8, 1)
    write_wav(f"{OUT}/np_speech48.wav", sp8, 1)
    print("done ->", OUT)


if __name__ == "__main__":
    sys.exit(main())
