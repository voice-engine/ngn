// voice/algo/voice_algo.h — libvoice_algo C ABI（16k 音频算法库）
//
// 封装 PBFDKF 卡尔曼 AEC（单通道实例）与 GCC-PHAT DOA + DSB 波束（多通道），
// 供引擎 dlopen 或直接链接。全部接口线程语义：每个句柄单线程使用。
//
//   AEC：单通道实例（mic[0] → out[0]，ref 单通道）。多麦克风阵列由引擎
//        逐通道各建一个实例（参考信号共享）。
//   DOA/BF：in[nch][n] 去交错平面波波束 → out[n]；DOA 由 getter 读取，
//        库内自动周期重估（首估 >= 0.4 s 数据，之后每 0.5 s 增量刷新，
//        分析窗取最近 4 s）。
//
// 采样率：推荐 16000（内部 AEC 帧 512 = 32 ms、覆盖 320 ms；BF STFT/GCC
// 全部按采样率参数化）。48000 走旧参数（回归兼容），其他 8k–96k 按时长换算。
// 处理长度 n 任意（内部按块缓冲，AEC 块尾不足部分零填充"预览"输出且不污染
// 自适应状态）。

#ifndef VOICE_ALGO_H
#define VOICE_ALGO_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct va_aec va_aec;

va_aec* va_aec_create(int sr);               // sr=16000；失败返回 NULL
void    va_aec_destroy(va_aec*);

// n 任意；mic/ref/out 均 deinterleaved；ref 单通道(16k)；返回 0 成功，
// 非法参数返回 -1。mic[0]/out[0] 为该实例的输入/输出通道。
int     va_aec_process(va_aec*, const float* const* mic, const float* ref,
                       float* const* out, int n);

// ---- 共享参考谱（多通道共享同一远端参考时省 N-1 遍 x FFT+谱环）-------- //
// 流程：refsvc = va_aec_ref_create(sr) → 各通道 va_aec_create_shared 挂接 →
// 每块先 va_aec_ref_push(refsvc, ref, n) 再逐通道 va_aec_process_shared。
// 仅 sr=16000（Hermitian 半谱路径）。共享实例输出与独立实例逐位一致
//（同 FFTPlan/同 nx 表达式/同遍历序）。
typedef struct va_aec_ref va_aec_ref;

va_aec_ref* va_aec_ref_create(int sr);       // 仅 16000；失败 NULL
void        va_aec_ref_destroy(va_aec_ref*);
int         va_aec_ref_push(va_aec_ref*, const float* ref, int n);  // n=B(512) 倍数

va_aec*     va_aec_create_shared(int sr, va_aec_ref*);  // 失败 NULL
// n=B(512) 倍数（无预览路径）；ref 为本段时域参考（能量统计用，谱已由
// refsvc 承载）；非法/未挂接返回 -1
int         va_aec_process_shared(va_aec*, const float* mic, const float* ref,
                                  float* out, int n);

typedef struct va_doa_bf va_doa_bf;

// nch>=2；array_type: "ula"（geom=相邻间距 m，默认 0.04）| "uca"（geom=半径 m，
// 默认 0.035）；NULL → "ula"。失败返回 NULL。
va_doa_bf* va_doa_bf_create(int nch, int sr, const char* array_type, double geom);
void    va_doa_bf_destroy(va_doa_bf*);

// 波束：in[nch][n] → out[n]（DSB，导向角 = 库内最近 DOA 估计）；返回 0 成功。
int     va_doa_bf_process(va_doa_bf*, const float* const* in, float* out, int n);

// 最近一次 DOA 估计（度；ULA ∈ [-90,90] broadside，UCA ∈ [0,360)）。
float   va_doa_bf_get_doa(const va_doa_bf*);

#ifdef __cplusplus
}
#endif

#endif  // VOICE_ALGO_H
