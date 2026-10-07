#!/usr/bin/env python3
# gen_test_wav.py — 自包含生成引擎测试素材（仅需 numpy，无外部素材依赖）：
#   test4ch16k.wav : 20s 4ch 16k S16 —— 前 10s 语音（双共振峰 AR + 音节包络）、
#                    后 10s 静音（speech/silence RMS 比 ≥ 20dB，test_engine 断言）
#   test48k.wav    : 20s 48k 单音 1kHz（fifo 回采测试）
#   sim4.wav       : test4ch16k 的别名（test_shm.sh mock 引擎素材）
import os
import shutil
import wave

import numpy as np

SR4 = 16000


def gen_speech(n: int, seed: int = 20261005, rms: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)

    def ar(x, f, bw):
        r = np.exp(-np.pi * bw / SR4)
        th = 2 * np.pi * f / SR4
        a1, a2 = 2 * r * np.cos(th), -r * r
        y = np.zeros_like(x)
        p = q = 0.0
        for i in range(len(x)):
            p = a1 * p + a2 * q + x[i]
            q = p
            y[i] = p
        return y

    y = ar(x, 350, 160) + 0.6 * ar(x, 1090, 290)
    i = np.arange(n)
    syl = np.maximum(0, np.sin(2 * np.pi * i / (0.9 * SR4))) ** 1.2
    slow = 0.6 + 0.4 * np.sin(2 * np.pi * i / (3.7 * SR4) + 1.0)
    y = y * syl * slow
    g = rms / np.sqrt((y ** 2).mean() + 1e-30)
    return (y * g).astype(np.float32)


def main():
    n = 20 * SR4
    speech = gen_speech(n // 2)
    silence = np.zeros(n // 2, dtype=np.float32)
    mono = np.concatenate([speech, silence])
    ch4 = np.stack([mono, mono * 0.95, mono * 1.05, mono * 0.9])  # 4ch 微差异
    s16 = (np.clip(ch4.T, -1, 1) * 32767).astype('<i2')
    with wave.open('test4ch16k.wav', 'wb') as o:
        o.setnchannels(4)
        o.setsampwidth(2)
        o.setframerate(SR4)
        o.writeframes(s16.tobytes())
    sp = 20 * np.log10(np.sqrt((speech ** 2).mean()) + 1e-30)
    print(f"test4ch16k.wav: 4ch 16k 20.0s  speech_rms={sp:.1f}dBFS  silence_rms=-270.3dBFS")

    t = np.arange(20 * 48000) / 48000
    tone = (0.35 * np.sin(2 * np.pi * 1000 * t) * 32767).astype('<i2')
    with wave.open('test48k.wav', 'wb') as o:
        o.setnchannels(1)
        o.setsampwidth(2)
        o.setframerate(48000)
        o.writeframes(tone.tobytes())
    print("test48k.wav: 1ch 48k 20.0s 1kHz sine")

    if not os.path.exists('sim4.wav'):
        shutil.copy('test4ch16k.wav', 'sim4.wav')
        print("sim4.wav: copy of test4ch16k.wav")


if __name__ == '__main__':
    main()
