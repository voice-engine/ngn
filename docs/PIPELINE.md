# ngn 技术参考（开发者文档）

> 架构 / 硬件 / C ABI / 内部实现。快速上手见 [README](../README.md)。

## 1. 系统总览

```
[其他应用]                          [voice-engine 守护进程]                    [硬件]
arecord -D voice_mic_shm  ← vshm插件 ← /dev/shm/voice_mic ← BF输出(16k mono)
(多读者，各自独立游标)                                  ▲
                                             AEC×4 → DOA → DSB
                                                        ▲
 AC108 hw:1,0（32k S32 线上流 → tag解码 4ch@16k）┘        （16k 直采，无软件抽取）

aplay -D voice_spk_shm → vshm插件 → /dev/shm/voice_spk(写者槽)
（多写者）                        → 引擎收割 float求和混音 → 永续 PCM 播放
                                    ├→ hw:0,0 speaker(48k 2ch)
                                    └→ ÷3 重采样 → 16k ref → AEC
```

算法封装为独立库 **libvoice_algo**（C ABI），三线程单核化：采集（核0）、
管线+算法（核2）、回采/播放（核1）。

## 2. 硬件与采集

| 项 | 内容 |
|---|---|
| 板 | NanoPi Air（Allwinner H3，Cortex-A7×4，484MB） |
| 麦阵 | VOICEN Linear 4 Mic Array Kit（**均匀线阵，间距 40mm**；麦序 [0,3,2,1]——引擎须 `--mic-order 0,3,2,1`） |
| ADC | AC108 编码模式：DATO1 单线打包 4ch，32bit 线上字 = [30bit 音频][2bit tag] |
| 声卡 | card1（AC108，采集）；card0（H3 codec，播放；音量档映射非线性，测试常用 22/31） |
| 直采 | 线上流 32k 立体声 S32，解码即 4ch@16k，无软件抽取。ac108 out-of-tree 驱动在 hw_params 自动配置，免用户态服务 |

## 3. 算法库 libvoice_algo

板上编译 `-O3 -mcpu=cortex-a7 -mfpu=neon-vfpv4`。核心实现要点：

- **AEC（PBFDKF）**：overlap-save 分块频域卡尔曼（16k：512 帧 × 10 块 = 320ms）+
  双路径（背景自适应 / 前景输出）。关键组件：残差 PSD 非对称 EMA 的 R 估计
  （天然双讲保护，无 Geigel 开关）、证据自适应晚期块上限、欠消联动 Q 自适应、
  后置维纳滤波（相干门控）、流式任意 n 零错位（状态快照回滚）。
- **DOA**：GCC-PHAT 全麦对互相关 + 方位网格搜索（ULA 单侧 ±90°），旋转相量
  分数 lag 精确求值；限带 c/(2D)=1.43kHz。流式走**增量槽环**（满块 FFT 缓存，
  估计只重求和+搜索）。
- **DSB/MVDR**：分数延迟对齐平均（主链）；MVDR（STFT+SMI 协方差+对角加载）
  用于定向干扰场景。
- **性能优化**（16k 路径）：Hermitian 半谱（AEC 热循环与存储减半）、投影错峰、
  参考静音门控（连续静音 ≥20 块触发，语音期逐位等价）、共享参考谱
  （RefEngine：多通道共用一次参考 FFT）。48k 路径与参考实现逐位一致（回归基线）。

### 3.1 C ABI

```c
va_aec*     va_aec_create(int sr);                     // 16k；每麦道一个实例
int         va_aec_process(va_aec*, const float* const* mic,
                           const float* ref, float* const* out, int n);
va_aec_ref* va_aec_ref_create(int sr);                 // 共享参考谱服务（仅 16k）
int         va_aec_ref_push(va_aec_ref*, const float* ref, int n);  // n=512 倍数
va_aec*     va_aec_create_shared(int sr, va_aec_ref*); // 挂接共享谱
int         va_aec_process_shared(va_aec*, const float* mic,
                                  const float* ref, float* out, int n);
va_doa_bf*  va_doa_bf_create(int nch, int sr, const char* array_type, double geom);
int         va_doa_bf_process(va_doa_bf*, const float* const* in, float* out, int n);
float       va_doa_bf_get_doa(const va_doa_bf*);
```

## 4. 引擎 voice-engine

**永续播放架构**：PCM 自启动持续 RUNNING——无写者时写静音垫（同步过 ÷3
抽取进 ref 环），ref 与 PCM 严格恒速同流：应用断续只是内容断续，流永不断
（AEC 参考游标零漂移，无欠载空洞）。

| 线程 | 核 | 节拍 | 说明 |
|---|---|---|---|
| 采集 | 0 | wire period 5ms | tag 解码+float 一步 → 16k 4ch |
| 管线+算法 | 2 | 32ms（512 帧） | 共享参考谱 → AEC×4 → DOA/BF → 出口 |
| 回采/播放 | 1 | 8ms | 收割混音 → PCM 补到水位 [0,1024] 帧（静音垫） |

- 延迟：spk-out ≈ 17-21ms（fill 实测 p50 17ms）；mic-in ≈ 40-55ms
  （结构下限，AEC 块长主导）；AEC ref 净滞后 ≈ 15ms 恒定。
- 性能：全链 rtf 满频 ~0.5、待机 0.40（静音门控）；单核（核 3 空闲）。
- CLI：`--ipc {shm,fifo}`、`--device <pcm>`（失败自动回退 null）、
  `--mic-order 0,3,2,1`、`--simulate-capture <wav>`（无声联调）、
  `--dump-mix`、`--stats-interval N`。
- 诊断 dump：`VA_DUMP_MIC/REF/AEC/BF=1` → 自动落 `./dumps/`（成对时间戳，
  `VA_DUMP_DIR`/`VA_DUMP_MAX_MB` 可配，默认 500MB 轮转）。

## 5. IPC 与 ALSA 接入

| 设备 | 机制 | 语义 |
|---|---|---|
| `voice_mic_shm` | /dev/shm/voice_mic：读者表+S16 环 | **多读者**（各自游标/追赶）；引擎重启自动重注册 |
| `voice_spk_shm` | /dev/shm/voice_spk：写者槽 | **多写者**：引擎收割求和混音；写者断续不断流 |
| `voice_mic` / `voice_spk` | /tmp fifo | 单读者/单写者（兼容路径） |

asound.conf 定义（驱动更新可能覆盖 /etc/asound.conf，条目丢失需补回）：

```
pcm.voice_mic_shm { type vshm  shm "/dev/shm/voice_mic" }
pcm.voice_spk_shm { type vshm  shm "/dev/shm/voice_spk" }
```

插件 type 名 `vshm`（避开 alsa-lib 内置 shm 截胡）。测试 41 项（test_shm.sh）。

## 6. 部署与测试

- 部署：`algo/sync_build.sh [build|test|all]`、`engine/deploy.sh`
  （`NGN_HOST`/`NGN_REMOTE` 可覆盖目标；板上默认 `/home/i/ngn`）。
- 测试：`algo/test_algo`（算法全量）、`engine/test_engine.sh`（16 项）、
  `engine/test_system.sh`（端到端）、`test/test_shm.sh`（IPC 集成）、
  `engine/test_fifo_race.sh`（竞态回归）。CI（x86）跑算法全测+编译+simulate 冒烟。
- 麦序标定：`tools/calib_mic_order.py`（见 tools/CALIB.md）。

## 7. 指标基线（板上实测）

| 项 | 指标 |
|---|---|
| AEC 合成单讲稳态 ERLE | 21.9 dB |
| AEC 真实音乐回声（板喇叭近场） | 10–13 dB（残差 87% 为播放链非线性，即线性上限） |
| DOA 合成精度 / 在线稳定性 | 1.3° / ±3° |
| DSB 阵增益 | +6.2 dB（4 麦理论 6.02） |
| 播放 fill p50 / mic-in 延迟 | 17ms / 40–55ms |
| 全链 rtf（满频 / 待机） | ~0.5 / 0.40（单核） |
| ADC↔DAC 时钟偏差 | −0.27 ppm |

## 8. 待办

1. DOA 绝对精度验证：broadside±45° 已知角度摆放重测（早前端射布局本征病态）
2. 真实 AEC 深挖：逐通道输出口径、更长收敛素材、真实双讲场景
3. H3 散热方案后启用引擎常驻（无散热长时间满载会过温关机）
4. VAD：Silero C++ 移植替换特征版
5. 计算量增量项：DSB float 化、实输入半复 FFT、热循环手工 NEON
