// gtcrn — GTCRN 单通道降噪（ICASSP'24, 23.7K 参数, 33 MMACs/s, 全因果）
//
// 16k mono 逐帧流式：STFT(512/256, sqrt-Hann) → ERB/SFE → 编码器 →
// 2×DPGRNN → 解码器 → CRM → iSTFT(WOLA)。算法延迟 32ms（2 hop）。
// 权重由 tools/export_gtcrn_weights.py 从官方 checkpoint 导出（BN 已折叠），
// 语义对拍基准：tools/gtcrn_check_export.py（numpy vs torch 流式模型）。
//
// 线程安全：实例不共享即可多线程；单实例非线程安全。
#ifndef GTCRN_H
#define GTCRN_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct va_gtcrn va_gtcrn;

va_gtcrn* va_gtcrn_create(void);
void      va_gtcrn_destroy(va_gtcrn*);
void      va_gtcrn_reset(va_gtcrn*);

// 16k mono，n 为 256 的倍数。每 256 样本一帧，输出同长度；
// 输出相对输入恒延迟 1 帧（16ms，重叠相加的 lookahead），模型因果窗
// 另含 1 帧，总算法延迟 32ms。返回处理的帧数，出错返回负值。
int va_gtcrn_process(va_gtcrn*, const float* x, float* y, int n);

#ifdef __cplusplus
}
#endif

#endif
