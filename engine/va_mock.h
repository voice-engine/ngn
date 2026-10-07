// va_mock.h - mock stubs for libvoice_algo (used when -DVA_MOCK at build time)
//
// 签名已与真库 ../algo（库源码）/voice_algo.h 完全对齐（平面指针数组 ABI）：
//   va_aec_create(16000)                          -> 单通道 AEC 实例
//   va_aec_process(aec, mic, ref, out, n)         -> mic/out 为平面指针数组，
//     mic[0] -> out[0] 为该实例的输入/输出通道；ref 单通道共享。
//     引擎对 4 个通道各建一个实例（4 x va_aec），共享同一 ref。
//   va_doa_bf_create(4, 16000, "ula", 0.04)       -> 波束形成器
//   va_doa_bf_process(bf, in, out, n)             -> in 为 4ch 平面 in[nch][n]
//   va_doa_bf_get_doa(bf)                         -> 最近 DOA 估计（度）
//
// mock 语义（任务规定）：AEC: out = mic（ref 忽略）；BF: out = ch0；DOA 恒 0。
// 真库落地后 Makefile 自动改链 libvoice_algo.so（无需此文件）。
#ifndef VA_MOCK_H
#define VA_MOCK_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct va_aec va_aec;

va_aec* va_aec_create(int sr);               // sr=16000；失败返回 NULL
void    va_aec_destroy(va_aec*);
int     va_aec_process(va_aec*, const float* const* mic, const float* ref,
                       float* const* out, int n);

// 共享参考谱（与真库 voice_algo.h 对齐）。mock 语义：ref push 无操作、
// create_shared 等同 create（仅 16k）、process_shared out=mic（n 须 512 倍数）
typedef struct va_aec_ref va_aec_ref;
va_aec_ref* va_aec_ref_create(int sr);       // 仅 16000；失败 NULL
void        va_aec_ref_destroy(va_aec_ref*);
int         va_aec_ref_push(va_aec_ref*, const float* ref, int n);
va_aec*     va_aec_create_shared(int sr, va_aec_ref*);
int         va_aec_process_shared(va_aec*, const float* mic, const float* ref,
                                  float* out, int n);

typedef struct va_doa_bf va_doa_bf;

va_doa_bf* va_doa_bf_create(int nch, int sr, const char* array_type, double geom);
void    va_doa_bf_destroy(va_doa_bf*);
int     va_doa_bf_process(va_doa_bf*, const float* const* in, float* out, int n);
float   va_doa_bf_get_doa(const va_doa_bf*);

#ifdef __cplusplus
}
#endif

#endif // VA_MOCK_H
