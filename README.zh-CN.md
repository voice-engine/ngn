# NGN — VOICEN Linear 4 Mic Array Kit 语音引擎

[English](README.md) | [简体中文](README.zh-CN.md)

为 **VOICEN Linear 4 Mic Array Kit**（AC108 四麦均匀线阵，间距 40mm，NanoPi Air）
设计的实时语音引擎。以守护进程方式运行在板上，提供：

- **AEC** 回声消除（合成 21.9dB / 真实音乐 10–13dB ERLE）
- **DOA** 定向（真实声源在线 ±3°）
- **波束成形** DSB 输出，阵增益 6.2dB
- **虚拟声卡** —— 任意 ALSA 应用直接录制处理后的麦克风
  （`arecord -D voice_mic_shm`）、播放自动接入回声参考
  （`aplay -D voice_spk_shm`）

全链路**单核实时**（rtf ≈ 0.4–0.5）。

## 快速开始

板上（算法单测也可在任意 Linux 主机运行）：

```sh
make -C algo && (cd algo && ./test_algo)      # 算法单测
make -C engine && ./engine/test_engine.sh     # 引擎测试
make -C plugin && sudo make -C plugin install # ALSA 插件
test/test_shm.sh                              # IPC 集成测试
```

运行引擎：

```sh
engine/voice_engine --ipc shm --mic-order 0,3,2,1 --stats-interval 10
```

应用侧：

```sh
arecord -D voice_mic_shm -f S16_LE -r 16000 -d 10 rec.wav   # 录音（无回声）
aplay  -D voice_spk_shm music.wav                           # 播放（自动回声参考）
```

从开发机部署：`algo/sync_build.sh` 与 `engine/deploy.sh`
（默认目标 ssh 主机 `nanopi`、路径 `/home/i/ngn`）。

## 文档

- [docs/PIPELINE.md](docs/PIPELINE.md) —— 架构 / C ABI / 内部实现（开发者）

## License

MIT
