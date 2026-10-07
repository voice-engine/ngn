# NGN — Voice Engine for VOICEN Linear 4-Mic Array Kit

[English](README.md) | [简体中文](README.zh-CN.md)

Real-time voice front-end for the **VOICEN Linear 4 Mic Array Kit**
(AC108 4-mic uniform linear array, 40 mm spacing, on a NanoPi Air).
It runs as a daemon on the board and provides:

- **AEC** — echo cancellation (21.9 dB synthetic / 10–13 dB real-music ERLE)
- **DOA** — direction finding (±3° online on real sources)
- **Beamforming** — DSB output, 6.2 dB array gain
- **Virtual sound cards** — any ALSA app can record the processed mic
  (`arecord -D voice_mic_shm`) and play back with automatic echo reference
  (`aplay -D voice_spk_shm`)

The full pipeline runs in real time on a **single core** (rtf ≈ 0.4–0.5).

## Quick start

On the board (or any Linux host for the algorithm tests):

```sh
make -C algo && (cd algo && ./test_algo)      # algorithm unit tests
make -C engine && ./engine/test_engine.sh     # engine suite
make -C plugin && sudo make -C plugin install # ALSA plugins
test/test_shm.sh                              # IPC integration suite
```

Run the engine:

```sh
engine/voice_engine --ipc shm --mic-order 0,3,2,1 --stats-interval 10
```

Then from any application:

```sh
arecord -D voice_mic_shm -f S16_LE -r 16000 -d 10 rec.wav   # record (echo-free)
aplay  -D voice_spk_shm music.wav                          # play (echo-cancelling)
```

Deploy from a dev machine: `algo/sync_build.sh` and `engine/deploy.sh`
(target defaults to ssh host `nanopi`, path `/home/i/ngn`).

## Documentation

- [docs/PIPELINE.md](docs/PIPELINE.md) — architecture, C ABI, internals (for developers)

## License

MIT
