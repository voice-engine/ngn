# gtcrn_bench — GTCRN 部署方案对比基准（ORT / ncnn vs 手写 C）

选型实验工具，结论见 docs/PIPELINE.md §7 与提交记录。手写 C 实现为主链
（algo/gtcrn.*），本目录仅保留对照基准与模型转换脚本。

## 文件

| 文件 | 用途 |
|---|---|
| `gtcrn_ort.cpp` | onnxruntime 逐帧流式基准（对拍 + RTF；板上 1.38 → 否决） |
| `gtcrn_ncnn.cpp` | ncnn 整段离线基准（STFT/WOLA 与 ort 版共用骨架） |
| `gru_bench_ncnn.cpp` | ncnn GRU 层微基准（单层 33 步 vs 手写 gru_step） |
| `gtcrn_export_trace.py` | 导出友好版 GTCRN（chunk→split/Unfold→pad+slice+stack 等改写），trace → pnnx/onnx 用 |

依赖：官方 repo `~/lab/nanopi-neo-air/gtcrn`（clone）+ 其 stream checkpoint。

## 结论摘要（2026-10）

- **ORT**：板上 RTF 1.38（小算子调度开销主导），否决。
- **ncnn**：完整模型三条转换链路均受阻——pnnx 生成动态 -233 Slice 静态图无法执行、
  旧 onnx2ncnn 不支持 Unfold/Gather/ConstantOfShape；消除 Unfold/chunk/索引后仍卡
  GRU 前的 Reshape 维度。**部署摩擦大，GRU 型微模型不友好**。
  GRU 层微基准（33 步 H=4 双向）：Mac 20µs ≈ 手写单向 2.5×。
- **手写 C**（采用）：板上 RTF ~0.18（冷），输出与 torch 流式参考 -48.4dB。

## 板上编译 ncnn（对照用）

```sh
# 板上：git clone 太慢，从 Mac 打包传源码后
sudo apt-get install -y cmake
cd /tmp/ncnn && mkdir -p build && cd build
cmake -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF .. && make -j1 ncnn  # -j2 会触发热关机！
g++ -std=c++17 -O3 -mcpu=cortex-a7 -mfpu=neon-vfpv4 -I../src -Isrc \
    gru_bench_ncnn.cpp src/libncnn.a -lpthread -lm -o gru_bench
taskset -c 3 ./gru_bench <param> <bin>
```
