// voice/algo/voice_algo.cpp — libvoice_algo C ABI 包装实现。
//
// 内部组合拷贝改造后的 C++ 实现（aec_kf.* / bf.*，不动 afecpp 原文件）：
//   va_aec     → PBFDKF（分块频域卡尔曼 AEC，单通道实例）
//   va_doa_bf  → afe::DOA（GCC-PHAT 导向网格搜索）+ 窗 sinc 分数延迟 DSB
//
// AEC 帧长/块数按采样率选择（覆盖 ~320–341 ms）：
//   48000 → (1024, 16) = 341 ms   ——与 afecpp 旧默认逐位一致（回归基线）
//   16000 → (512, 10)  = 320 ms   ——32 ms 帧、10 块
//   其他  → 帧 = next_pow2(21.3 ms)，块数 = round(0.33 s / 帧)，[4, 64]
//
// DOA/BF 流式策略：输入累积（保留最近 4 s + 窗余量），DOA 首次 0.4 s 数据
// 后估计、此后每 0.5 s 新增数据用最近 4 s 刷新；DSB 输出按当前导向角对每次
// 调用精确生成本次 n 个样本（子段窗口 ±margin，窗内窗 sinc 对齐后平均；
// 流首/尾 margin 内样本因零边界有轻微瞬态，属正常启动边界）。

#include "voice_algo.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "aec_kf.hpp"
#include "bf.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ------------------------------------------------------------------ //
// AEC
// ------------------------------------------------------------------ //
static void va_aec_params(int sr, int& frame, int& blocks) {
    if (sr == 48000) { frame = 1024; blocks = 16; return; }  // 旧默认（341 ms）
    if (sr == 16000) { frame = 512; blocks = 10; return; }   // 32 ms × 10 = 320 ms
    frame = 256;
    while (frame < (int)(0.02133 * sr + 0.5)) frame <<= 1;   // ≈ 21.3 ms 帧
    blocks = (int)(0.33 * sr / frame + 0.5);                 // ≈ 330 ms 覆盖
    if (blocks < 4) blocks = 4;
    if (blocks > 64) blocks = 64;
}

struct va_aec {
    PBFDKF aec;
    bool shared = false;
    va_aec(int sr, int frame, int blocks, bool half) : aec(frame, blocks, sr, half) {}
};

struct va_aec_ref {
    int sr = 0, frame = 0, blocks = 0;
    RefEngine re;    // 半谱参考谱服务（16k herm 专用）
};

extern "C" {

va_aec* va_aec_create(int sr) {
    if (sr < 8000 || sr > 96000) return nullptr;
    int frame, blocks;
    va_aec_params(sr, frame, blocks);
    // 16k（引擎路径）启用 Hermitian 半谱快速路径；48k 走原全谱路径，
    // 与 afecpp 参照保持逐位一致（回归基线）
    return new va_aec(sr, frame, blocks, sr == 16000);
}

va_aec_ref* va_aec_ref_create(int sr) {
    if (sr != 16000) return nullptr;   // 共享参考谱仅支持 herm 半谱路径
    va_aec_ref* h = new va_aec_ref();
    va_aec_params(sr, h->frame, h->blocks);
    h->sr = sr;
    h->re.init(h->frame, h->blocks);
    return h;
}

void va_aec_ref_destroy(va_aec_ref* h) { delete h; }

int va_aec_ref_push(va_aec_ref* h, const float* ref, int n) {
    if (!h || !ref || n <= 0 || n % h->frame) return -1;
    for (int off = 0; off < n; off += h->frame) h->re.push(ref + off);
    return 0;
}

va_aec* va_aec_create_shared(int sr, va_aec_ref* refsvc) {
    if (sr != 16000 || !refsvc || refsvc->sr != sr) return nullptr;
    va_aec* h = new va_aec(sr, refsvc->frame, refsvc->blocks, true);
    h->aec.attach_shared(&refsvc->re);
    if (!h->aec.shared_attached()) { delete h; return nullptr; }
    h->shared = true;
    return h;
}

int va_aec_process_shared(va_aec* h, const float* mic, const float* ref,
                          float* out, int n) {
    if (!h || !h->shared || !mic || !ref || !out || n <= 0 || n % h->aec.frame_size())
        return -1;
    h->aec.process_shared(mic, ref, out, n);
    return 0;
}

void va_aec_destroy(va_aec* h) { delete h; }

int va_aec_process(va_aec* h, const float* const* mic, const float* ref,
                   float* const* out, int n) {
    if (!h || !mic || !ref || !out || n <= 0) return -1;
    if (!mic[0] || !out[0]) return -1;
    if (h->shared) return -1;   // 共享实例只允许 process_shared（块对齐）
    h->aec.process(mic[0], ref, out[0], n);
    return 0;
}

}  // extern "C"

// ------------------------------------------------------------------ //
// DOA + DSB 波束
// ------------------------------------------------------------------ //
struct va_doa_bf {
    int nch = 4;
    double sr = 16000.0;
    afe::DOA doa;                            // 默认构造后于 create 重建
    std::vector<std::vector<float>> buf;     // buf[m] 覆盖全局 [base, total)
    long long base = 0;                      // buf[*][0] 的全局样本下标
    long long total = 0;                     // 已接收样本数
    long long emitted = 0;                   // 已输出样本数
    long long since_est = 0;                 // 上次 DOA 估计后新增样本
    bool est_done = false;
    double theta = 0.0;
    int margin = 32;                         // DSB 子段窗口前后余量（样点）
    std::vector<double> tau;
    std::vector<float> win;                  // 通道窗口 scratch
};

extern "C" {

va_doa_bf* va_doa_bf_create(int nch, int sr, const char* array_type, double geom) {
    if (nch < 2 || nch > 16) return nullptr;
    if (sr < 8000 || sr > 96000) return nullptr;
    const char* at = array_type ? array_type : "ula";
    if (std::strcmp(at, "ula") != 0 && std::strcmp(at, "uca") != 0) return nullptr;
    if (!(geom > 0.0)) geom = (std::strcmp(at, "ula") == 0) ? 0.04 : 0.035;

    va_doa_bf* h = new va_doa_bf();
    h->nch = nch;
    h->sr = (double)sr;
    h->doa = afe::DOA(nch, sr, geom, 343.0, 1.0, 0.05, 4.0, at);
    // 最大阵内延迟 |tau|max = 孔径/(2c) = 1/(4·fmax)；余量 = sinc 核 half(16)
    // + |tau|max·sr + 8
    h->margin = 16 + (int)std::ceil((double)sr / (4.0 * h->doa.fmax())) + 8;
    h->buf.assign(nch, {});
    return h;
}

void va_doa_bf_destroy(va_doa_bf* h) { delete h; }

int va_doa_bf_process(va_doa_bf* h, const float* const* in, float* out, int n) {
    if (!h || !in || !out || n <= 0) return -1;
    for (int m = 0; m < h->nch; m++)
        if (!in[m]) return -1;
    const int nch = h->nch;

    // -- 1) 追加输入 -------------------------------------------------- //
    for (int m = 0; m < nch; m++)
        h->buf[m].insert(h->buf[m].end(), in[m], in[m] + n);
    h->total += n;
    h->since_est += n;

    // -- 2) 历史裁剪：保留 4 s 分析窗 + 本次调用所需窗余量 ------------ //
    {
        const long long keep = (long long)(4.0 * h->sr) + 2 * h->margin + n;
        const long long have = h->total - h->base;
        if (have > keep) {
            const size_t drop = (size_t)(have - keep);
            for (int m = 0; m < nch; m++)
                h->buf[m].erase(h->buf[m].begin(), h->buf[m].begin() + (std::ptrdiff_t)drop);
            h->base += (long long)drop;
        }
    }

    // -- 3) DOA 周期重估（增量槽环：满块才 FFT，估计只重求和+搜索）-------- //
    //    分析窗 = 最近完整块集（@16k 稳态 8×8192 ≈ 4.1 s，略去尾块）；首次
    //    估计在首个完整块完成时（0.512s，旧批式为 0.4s），此后每 ≥0.5s 且
    //    恰有新完整块时刷新。相比旧路径（每次 4s 全量重 FFT，每 0.5s 32 个
    //    FFT-8192）FFT 数降为每块 nch 个。
    {
        const int done = h->doa.inc_push(in, n);
        if (done > 0 && (!h->est_done ||
                         h->since_est >= (long long)(0.5 * h->sr))) {
            h->theta = h->doa.inc_estimate();
            h->est_done = true;
            h->since_est = 0;
        }
    }

    // -- 4) DSB：输出全局 [emitted, emitted+n)，子段化精确窗口 ---------- //
    h->doa.delays(h->theta * M_PI / 180.0, h->tau);
    const int step = (int)(0.25 * h->sr);
    const float invn = 1.0f / (float)nch;
    const long long e_end = h->emitted + n;
    for (long long e = h->emitted; e < e_end;) {
        const long long len = std::min<long long>((long long)step, e_end - e);
        const long long g0 = (e - h->margin > h->base) ? e - h->margin : h->base;
        const long long g1 = (e + len + h->margin < h->total) ? e + len + h->margin
                                                              : h->total;
        const size_t wlen = (size_t)(g1 - g0);
        const size_t woff = (size_t)(e - g0);
        float* o = out + (e - h->emitted);
        for (int m = 0; m < nch; m++) {
            const float* src = h->buf[m].data() + (size_t)(g0 - h->base);
            h->win.assign(src, src + wlen);
            // frac_delay(x,d)(t) = x(t-d)；d = -tau_m → 对齐 x_m(t+tau_m)
            std::vector<float> sh = afe::frac_delay(h->win, -h->tau[m], h->sr, 16);
            if (m == 0) {
                for (size_t i = 0; i < (size_t)len; i++) o[i] = sh[woff + i];
            } else {
                for (size_t i = 0; i < (size_t)len; i++) o[i] += sh[woff + i];
            }
        }
        for (size_t i = 0; i < (size_t)len; i++) o[i] *= invn;
        e += len;
    }
    h->emitted += n;
    return 0;
}

float va_doa_bf_get_doa(const va_doa_bf* h) {
    if (!h) return 0.0f;
    return (float)h->theta;
}

}  // extern "C"
