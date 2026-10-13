// gtcrn.cpp — GTCRN 流式降噪，C/C++ 实现（固定形状，逐帧状态机）
//
// 语义与 tools/gtcrn_check_export.py 的 numpy 参考逐算子一致（该参考已与
// 官方 StreamGTCRN 对拍通过）。权重布局见 tools/export_gtcrn_weights.py：
// BN 已折叠；解码 GT depthwise 已做 T/F 双轴翻转（与编码同一相关路径）；
// pc1/pc2 统一 (out,in)；GRU 双偏置 b_ih/b_hh 分离（n 门门控语义）。
//
// 布局约定：激活一律 [C][F] 平面（f 连续）；DPGRNN 内部转 [F][C]。
// GT depthwise 历史帧按 [n][16][35] 存（频轴两侧各 pad 1 零），抽头免边界判断。
// GRU 权重在 create 时转置为 (I,3H)/(H,3H) 的 axpy 布局（NEON 友好）。
// sigmoid/tanh 用多项式快速实现（|err|≲3e-6，已随整链对拍验证）。
//
// 逐帧流程：STFT(512/256) → feat[mag,re,im] → ERB(129) → SFE(9ch) →
// en0(65)/en1(33) → GT×3(dil 1/2/5) → DPGRNN×2 → GT×3(dil 5/2/1, 残差) →
// de3(65)/de4(129,tanh) → iERB → CRM → iFFT → WOLA。
#include "gtcrn.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "gtcrn_weights.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef VA_PROFILE
#include <ctime>
struct GtcrnProf {
    double stft, erb, en, gt, dp, de, ifft;
    int n;
};
GtcrnProf g_prof = {};
static inline double prof_now() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static inline void prof_step(double& acc, double& t0) {
    acc += prof_now() - t0;
    t0 = prof_now();
}
#define PSTEP(slot) prof_step(g_prof.slot, _tp)
#else
#define PSTEP(slot) ((void)0)
#endif

namespace {

constexpr int HOP = 256;
constexpr int NFFT = 512;
constexpr int F257 = 257;
constexpr int FL = 65;   // ERB 低频直通带
constexpr int FH = 192;  // ERB 高频输入带
constexpr int EB = 64;   // ERB 输出带
constexpr int FB = 129;  // ERB 后总带宽
constexpr int FM = 33;   // 主带
constexpr int FP = 35;   // 主带 + 频轴两侧 pad
constexpr int CH = 16;

inline float preluf_(float x, float w) { return x > 0.f ? x : w * x; }

// ---- 快速 exp/sigmoid/tanh（多项式，|err|≲3e-6；整链对拍验证覆盖） ----
inline float fast_expf_(float x) {
    // e^x = 2^k · e^r，|r|≤ln2/2；e^r 用 5 阶 Horner（r^6/720 ≤ 2.4e-6）
    if (x > 80.f) return INFINITY;
    if (x < -80.f) return 0.f;
    const float INV_LN2 = 1.44269504088896f;
    const float LN2 = 0.69314718055995f;
    float k = rintf(x * INV_LN2);
    float r = x - k * LN2;
    float p = 1.f / 120.f;
    p = p * r + 1.f / 24.f;
    p = p * r + 1.f / 6.f;
    p = p * r + 0.5f;
    p = p * r + 1.f;
    p = p * r + 1.f;
    union { uint32_t i; float f; } sc;
    sc.i = (uint32_t)((int)k + 127) << 23;
    return p * sc.f;
}
inline float sigmoidf_(float x) {
    float e = fast_expf_(-fabsf(x));
    return x >= 0.f ? 1.f / (1.f + e) : e / (1.f + e);
}
inline float tanhf_(float x) {
    float ax = fabsf(x);
    if (ax > 15.f) return x >= 0.f ? 1.f : -1.f;
    float e = fast_expf_(-2.f * ax);
    float t = 2.f / (1.f + e) - 1.f;
    return x >= 0.f ? t : -t;
}

#ifdef __ARM_NEON
// 4-lane 快速超越（GRU 门控热路径）。r∈(−ln2,ln2]，8 阶 Horner，
// 相对误差 ≲3e-6，与标量版同级（整链对拍验证覆盖）。
inline float32x4_t fast_expf4(float32x4_t x) {
    x = vminq_f32(x, vdupq_n_f32(80.f));
    x = vmaxq_f32(x, vdupq_n_f32(-80.f));
    const float32x4_t magic = vdupq_n_f32(12582912.0f);  // 2^23+2^22
    float32x4_t kf = vmulq_f32(x, vdupq_n_f32(1.44269504088896f));
    kf = vsubq_f32(vaddq_f32(kf, magic), magic);  // round-to-nearest-even
    float32x4_t r = vmlsq_f32(x, kf, vdupq_n_f32(0.69314718055995f));
    float32x4_t p = vdupq_n_f32(1.f / 5040.f);
    p = vmlaq_f32(vdupq_n_f32(1.f / 720.f), p, r);
    p = vmlaq_f32(vdupq_n_f32(1.f / 120.f), p, r);
    p = vmlaq_f32(vdupq_n_f32(1.f / 24.f), p, r);
    p = vmlaq_f32(vdupq_n_f32(1.f / 6.f), p, r);
    p = vmlaq_f32(vdupq_n_f32(0.5f), p, r);
    p = vmlaq_f32(vdupq_n_f32(1.f), p, r);
    p = vmlaq_f32(vdupq_n_f32(1.f), p, r);
    int32x4_t ki = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(kf), vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(ki));
}
inline float32x4_t sigmoidf4(float32x4_t x) {
    float32x4_t e = fast_expf4(vnegq_f32(vabsq_f32(x)));
    float32x4_t d = vaddq_f32(e, vdupq_n_f32(1.f));
    float32x4_t inv = vrecpeq_f32(d);
    inv = vmulq_f32(vrecpsq_f32(d, inv), inv);
    float32x4_t neg = vmulq_f32(e, inv);
    return vbslq_f32(vcgeq_f32(x, vdupq_n_f32(0.f)), inv, neg);
}
inline float32x4_t tanhf4(float32x4_t x) {
    float32x4_t ax = vabsq_f32(x);
    float32x4_t e = fast_expf4(vmulq_f32(vdupq_n_f32(-2.f), ax));
    float32x4_t d = vaddq_f32(e, vdupq_n_f32(1.f));
    float32x4_t inv = vrecpeq_f32(d);
    inv = vmulq_f32(vrecpsq_f32(d, inv), inv);
    float32x4_t t = vsubq_f32(vmulq_f32(vdupq_n_f32(2.f), inv), vdupq_n_f32(1.f));
    t = vminq_f32(t, vdupq_n_f32(1.f));
    return vbslq_f32(vcgeq_f32(x, vdupq_n_f32(0.f)), t, vnegq_f32(t));
}
#endif

// ---- 512 点 radix-2 复 FFT（迭代，位反转，表驱动） ----
struct FFT {
    float wr[NFFT], wi[NFFT];
    uint16_t rev[NFFT];
    FFT() {
        int lg = 0;
        while ((1 << lg) < NFFT) lg++;
        for (int i = 0; i < NFFT; i++) {
            wr[i] = cosf(-2.f * (float)M_PI * i / NFFT);
            wi[i] = sinf(-2.f * (float)M_PI * i / NFFT);
        }
        rev[0] = 0;  // 递推依赖 rev[0]=0，堆上构造时不可依赖零初始化
        for (int i = 1; i < NFFT; i++)
            rev[i] = (uint16_t)((rev[i >> 1] >> 1) | ((i & 1) << (lg - 1)));
    }
    void run(float* re, float* im, bool inv) const {
        for (int i = 0; i < NFFT; i++) {
            uint16_t j = rev[i];
            if ((uint16_t)i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
        }
        for (int len = 2; len <= NFFT; len <<= 1) {
            int step = NFFT / len;
            for (int i = 0; i < NFFT; i += len) {
                for (int k = 0; k < len / 2; k++) {
                    int tw = k * step;
                    float c = wr[tw], s = inv ? -wi[tw] : wi[tw];
                    float ur = re[i + k], ui = im[i + k];
                    float xr = re[i + k + len / 2], xi = im[i + k + len / 2];
                    float vr = xr * c - xi * s;
                    float vi = xr * s + xi * c;
                    re[i + k] = ur + vr; im[i + k] = ui + vi;
                    re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
                }
            }
        }
        if (inv) {
            for (int i = 0; i < NFFT; i++) { re[i] /= NFFT; im[i] /= NFFT; }
        }
    }
};

// ---- 向量辅助（无 NEON 时标量回退，行为一致） ----
__attribute__((always_inline)) inline void axpy_(float* y, const float* x, float a, int n) {
#ifdef __ARM_NEON
    float32x4_t va = vdupq_n_f32(a);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t s0 = vld1q_f32(y + i), s1 = vld1q_f32(y + i + 4);
        vst1q_f32(y + i, vmlaq_f32(s0, vld1q_f32(x + i), va));
        vst1q_f32(y + i + 4, vmlaq_f32(s1, vld1q_f32(x + i + 4), va));
    }
    for (; i + 4 <= n; i += 4)
        vst1q_f32(y + i, vmlaq_f32(vld1q_f32(y + i), vld1q_f32(x + i), va));
    for (; i < n; i++) y[i] += a * x[i];
#else
    for (int i = 0; i < n; i++) y[i] += a * x[i];
#endif
}

__attribute__((always_inline)) inline float dot_(const float* a, const float* b, int n) {
#ifdef __ARM_NEON
    float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        s0 = vmlaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
        s1 = vmlaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    float32x4_t s = vaddq_f32(s0, s1);
    for (; i + 4 <= n; i += 4) s = vmlaq_f32(s, vld1q_f32(a + i), vld1q_f32(b + i));
    float t[4];
    vst1q_f32(t, s);
    float r = t[0] + t[1] + t[2] + t[3];
    for (; i < n; i++) r += a[i] * b[i];
    return r;
#else
    float r = 0.f;
    for (int i = 0; i < n; i++) r += a[i] * b[i];
    return r;
#endif
}

// y[O][F] = b ⊕ Σ_i W[o][i]·x[i][*]（行 axpy，f 连续）
void gemm_rows(const float* W, const float* b, const float* x, float* y,
               int O, int I, int F) {
    for (int o = 0; o < O; o++) {
        float* yo = y + (size_t)o * F;
        float bo = b ? b[o] : 0.f;
        for (int f = 0; f < F; f++) yo[f] = bo;
        const float* Wo = W + (size_t)o * I;
        for (int i = 0; i < I; i++)
            axpy_(yo, x + (size_t)i * F, Wo[i], F);
    }
}

// GRU 单步：x(I) h(H) → hout(H)。权重原始行主序：wih (3H,I) whh (3H,H)。
// dot 形式（寄存器内完成，无中间缓冲往返）；r/z 门偏置可合并，n 门分离
// （torch 语义：n = tanh(gi_n + r⊙gh_n)）。gate 顺序 [r, z, n]，3H ≤ 48。
void gru_step(const float* x, const float* h, float* hout,
              const float* wih, const float* whh,
              const float* bi, const float* bh, int I, int H) {
    float gr[16], gz[16], gn[16], gh2[16];
    for (int r = 0; r < H; r++) {
        gr[r] = dot_(wih + (size_t)r * I, x, I) + bi[r] +
                dot_(whh + (size_t)r * H, h, H) + bh[r];
        gz[r] = dot_(wih + (size_t)(H + r) * I, x, I) + bi[H + r] +
                dot_(whh + (size_t)(H + r) * H, h, H) + bh[H + r];
    }
    for (int r = 0; r < H; r++) {
        gh2[r] = dot_(whh + (size_t)(2 * H + r) * H, h, H) + bh[2 * H + r];
        gn[r] = dot_(wih + (size_t)(2 * H + r) * I, x, I) + bi[2 * H + r];
    }
#ifdef __ARM_NEON
    for (int r = 0; r < H; r += 4) {
        float32x4_t rg = sigmoidf4(vld1q_f32(gr + r));
        float32x4_t zg = sigmoidf4(vld1q_f32(gz + r));
        float32x4_t ng = tanhf4(vaddq_f32(vld1q_f32(gn + r),
                                          vmulq_f32(rg, vld1q_f32(gh2 + r))));
        float32x4_t hv = vld1q_f32(h + r);
        vst1q_f32(hout + r, vaddq_f32(ng, vmulq_f32(zg, vsubq_f32(hv, ng))));
    }
#else
    for (int r = 0; r < H; r++) {
        float rg = sigmoidf_(gr[r]);
        float zg = sigmoidf_(gz[r]);
        float ng = tanhf_(gn[r] + rg * gh2[r]);
        hout[r] = ng + zg * (h[r] - ng);
    }
#endif
}

// LayerNorm 整块 (33,16) + 逐位置仿射（biased 方差，eps=1e-8）
void layernorm_528(float* x, const float* w, const float* b) {
    constexpr int N = FM * CH;
    float mu = 0.f;
    for (int i = 0; i < N; i++) mu += x[i];
    mu /= N;
    float var = 0.f;
    for (int i = 0; i < N; i++) {
        float d = x[i] - mu;
        var += d * d;
    }
    var /= N;
    float inv = 1.f / sqrtf(var + 1e-8f);
    for (int i = 0; i < N; i++) x[i] = (x[i] - mu) * inv * w[i] + b[i];
}

}  // namespace

struct GTW {  // GT 块权重组（tra GRU 为转置布局）
    const float *pc1_w, *pc1_b, *pc1_a;
    const float *dw_w, *dw_b, *dw_a;
    const float *pc2_w, *pc2_b;
    const float *tra_wih, *tra_whh, *tra_bih, *tra_bhh, *tra_fc_w, *tra_fc_b;
};

struct DPW {  // DPGRNN 权重组（全部 GRU 为转置布局）
    const float *i1f_wih, *i1f_whh, *i1f_bih, *i1f_bhh;
    const float *i1b_wih, *i1b_whh, *i1b_bih, *i1b_bhh;
    const float *i2f_wih, *i2f_whh, *i2f_bih, *i2f_bhh;
    const float *i2b_wih, *i2b_whh, *i2b_bih, *i2b_bhh;
    const float *e1_wih, *e1_whh, *e1_bih, *e1_bhh;
    const float *e2_wih, *e2_whh, *e2_bih, *e2_bhh;
    const float *ifc_w, *ifc_b, *iln_w, *iln_b;
    const float *rfc_w, *rfc_b, *rln_w, *rln_b;
};

struct va_gtcrn {
    GTW gtw[2][3];
    DPW dpw[2];

    float hist[HOP];   // 上一跳样本（当前帧窗口前半）
    float ola[NFFT];   // WOLA 累加
    FFT fft;
    float win[NFFT];

    // GT depthwise 历史（en/de 各 3 块；dil 1/2/5 → 2/4/10 帧）
    float ecc[3][10][CH][FP];
    float dcc[3][10][CH][FP];
    float eth[3][CH], dth[3][CH];  // TRA GRU hidden
    float ih[2][FM][CH];           // inter GRU hidden（dp1/dp2）

    // 工作平面
    float wbuf[NFFT];
    float fre[NFFT], fim[NFFT];
    float feat[3][F257];
    float bm3[3][FB];
    float s9[9][FB];
    float hp9[9][FB + 4];
    float a65[CH][65], a33[CH][FM], xin[CH][FM];
    float skip0[CH][65], skip1[CH][FM], skips[3][CH][FM];
    float hp16[CH][69];
    float xp33[CH][FP], xp65[CH][67];
    float de3e[CH][33], de3o[CH][32], de4e[2][65], de4o[2][64];
    float m129[2][129];

    int frame = 0;
    int dump_frames = 0;
    char dump_dir[256] = {0};
};

namespace {

void dump_stage(const va_gtcrn* g, const char* name, const void* p, int rows, int cols) {
    if (!g->dump_frames || g->frame >= g->dump_frames) return;
    char path[320];
    snprintf(path, sizeof path, "%s/f%03d_%s.bin", g->dump_dir, g->frame, name);
    FILE* f = fopen(path, "wb");
    if (f) {
        fwrite(p, 4, (size_t)rows * cols, f);
        fclose(f);
    }
}

// SFE：dst (C*3,F)，通道 c*3+j = x[c][f+j-1]
void sfe_band(const float* x, int C, int F, float* dst) {
    for (int c = 0; c < C; c++) {
        const float* src = x + (size_t)c * F;
        for (int j = 0; j < 3; j++) {
            float* d = dst + (size_t)(c * 3 + j) * F;
            for (int f = 0; f < F; f++) {
                int q = f + j - 1;
                d[f] = (q >= 0 && q < F) ? src[q] : 0.f;
            }
        }
    }
}

// GT 块：x (16,33) → out (16,33)。dil∈{1,2,5}；hist 为该块 [2dil][16][35]。
// W[kt] 对应时间抽头 t-(2-kt)·d（历史帧/当前帧均已频轴 pad 1）。
void gt_block(va_gtcrn* g, const float x[][FM], int side, int j, int dil,
              float hist[][CH][FP], float* tra_h, float out[][FM], const char* tag) {
    const GTW& w = g->gtw[side][j];
    static thread_local float s24[24][FM];
    static thread_local float h1[CH][FM];
    static thread_local float dwout[CH][FM];
    static thread_local float p2[8][FM];
    static thread_local float curp[CH][FP];
    char nm[32];

    // SFE(前 8 通道) + pc1 + PReLU
    sfe_band(&x[0][0], 8, FM, &s24[0][0]);
    gemm_rows(w.pc1_w, w.pc1_b, &s24[0][0], &h1[0][0], 16, 24, FM);
    for (int c = 0; c < CH; c++)
        for (int f = 0; f < FM; f++) h1[c][f] = preluf_(h1[c][f], w.pc1_a[0]);
    snprintf(nm, sizeof nm, "%s.pc1", tag);
    dump_stage(g, nm, h1, CH, FM);

    // depthwise (3,3) dil
    for (int c = 0; c < CH; c++) {
        memset(curp[c], 0, sizeof curp[c]);
        memcpy(&curp[c][1], h1[c], sizeof(float) * FM);
    }
    const int nd = 2 * dil;
    for (int c = 0; c < CH; c++) {
        const float* wc = w.dw_w + (size_t)c * 9;
        const float* taps[3] = {hist[0][c], hist[dil][c], curp[c]};
        float acc[FM];
        for (int f = 0; f < FM; f++) acc[f] = w.dw_b[c];
        for (int kt = 0; kt < 3; kt++) {
            const float* fr = taps[kt];
            const float* wk = wc + kt * 3;
            for (int kf = 0; kf < 3; kf++)
                axpy_(acc, fr + kf, wk[kf], FM);
        }
        for (int f = 0; f < FM; f++) dwout[c][f] = preluf_(acc[f], w.dw_a[0]);
    }
    snprintf(nm, sizeof nm, "%s.dw", tag);
    dump_stage(g, nm, dwout, CH, FM);

    // pc2 → (8,33)
    gemm_rows(w.pc2_w, w.pc2_b, &dwout[0][0], &p2[0][0], 8, CH, FM);
    snprintf(nm, sizeof nm, "%s.pc2", tag);
    dump_stage(g, nm, p2, 8, FM);

    // TRA：zt = mean(x²) → GRU(8→16) → fc → sigmoid → 乘
    float zt[8];
    for (int c = 0; c < 8; c++) {
        float s2 = 0.f;
        for (int f = 0; f < FM; f++) s2 += p2[c][f] * p2[c][f];
        zt[c] = s2 / FM;
    }
    float hh[16];
    gru_step(zt, tra_h, hh, w.tra_wih, w.tra_whh, w.tra_bih, w.tra_bhh, 8, 16);
    memcpy(tra_h, hh, sizeof hh);
    for (int c = 0; c < 8; c++) {
        float at = sigmoidf_(dot_(w.tra_fc_w + c * 16, hh, 16) + w.tra_fc_b[c]);
        for (int f = 0; f < FM; f++) p2[c][f] *= at;
    }
    snprintf(nm, sizeof nm, "%s.tra", tag);
    dump_stage(g, nm, p2, 8, FM);

    // shuffle 交错：out[2c]=pc2', out[2c+1]=x2
    for (int c = 0; c < 8; c++) {
        memcpy(out[2 * c], p2[c], sizeof(float) * FM);
        memcpy(out[2 * c + 1], x[8 + c], sizeof(float) * FM);
    }

    // 历史推进：左移 1 帧，末位 = 当前帧
    memmove(hist[0], hist[1], sizeof(float) * (size_t)(nd - 1) * CH * FP);
    memcpy(hist[nd - 1], curp, sizeof curp);
}

// DPGRNN：a33 [16][33] 原位更新；ih [33][16] 为 inter hidden
void dpgrnn(va_gtcrn* g, float a33[][FM], int j, float ih[][CH]) {
    const DPW& w = g->dpw[j - 1];
    static thread_local float X[FM][CH];
    static thread_local float y[FM][CH];
    static thread_local float intra_out[FM][CH];
    static thread_local float z[FM][CH];
    static thread_local float hf[4], hb[4], hn[8];
    static thread_local float of[FM][4], ob[FM][4];
    static thread_local float fcbuf[FM][CH];

    for (int c = 0; c < CH; c++)
        for (int f = 0; f < FM; f++) X[f][c] = a33[c][f];
    dump_stage(g, j == 1 ? "dp1.X" : "dp2.X", X, FM, CH);

    // intra：half 0/1（rnn1/rnn2）各含 fwd/bwd GRU(8→4)
    for (int half = 0; half < 2; half++) {
        const float* qw[2][4] = {
            {half ? w.i2f_wih : w.i1f_wih, half ? w.i2f_whh : w.i1f_whh,
             half ? w.i2f_bih : w.i1f_bih, half ? w.i2f_bhh : w.i1f_bhh},
            {half ? w.i2b_wih : w.i1b_wih, half ? w.i2b_whh : w.i1b_whh,
             half ? w.i2b_bih : w.i1b_bih, half ? w.i2b_bhh : w.i1b_bhh}};
        memset(hf, 0, sizeof hf);
        memset(hb, 0, sizeof hb);
        for (int f = 0; f < FM; f++) {
            gru_step(X[f] + half * 8, hf, hf, qw[0][0], qw[0][1], qw[0][2], qw[0][3], 8, 4);
            memcpy(of[f], hf, sizeof hf);
        }
        for (int f = FM - 1; f >= 0; f--) {
            gru_step(X[f] + half * 8, hb, hb, qw[1][0], qw[1][1], qw[1][2], qw[1][3], 8, 4);
            memcpy(ob[f], hb, sizeof hb);
        }
        for (int f = 0; f < FM; f++) {
            memcpy(&y[f][half * 8], of[f], 4 * sizeof(float));
            memcpy(&y[f][half * 8 + 4], ob[f], 4 * sizeof(float));
        }
        char sub[40];
        snprintf(sub, sizeof sub, "dp%d.h%dof", j, half);
        dump_stage(g, sub, of, FM, 4);
        snprintf(sub, sizeof sub, "dp%d.h%dob", j, half);
        dump_stage(g, sub, ob, FM, 4);
    }
    dump_stage(g, j == 1 ? "dp1.intra_rnn" : "dp2.intra_rnn", y, FM, CH);

    // intra fc → LN → 残差 +X（fc 输出经独立缓冲，避免原地覆写）
    for (int f = 0; f < FM; f++)
        for (int o = 0; o < CH; o++)
            fcbuf[f][o] = dot_(w.ifc_w + o * CH, y[f], CH) + w.ifc_b[o];
    memcpy(&y[0][0], &fcbuf[0][0], sizeof y);
    layernorm_528(&y[0][0], w.iln_w, w.iln_b);
    for (int f = 0; f < FM; f++)
        for (int c = 0; c < CH; c++) intra_out[f][c] = X[f][c] + y[f][c];
    dump_stage(g, j == 1 ? "dp1.intra" : "dp2.intra", intra_out, FM, CH);

    // inter：逐频点一步 GRU(8→8) ×2 组
    for (int f = 0; f < FM; f++)
        for (int half = 0; half < 2; half++) {
            gru_step(intra_out[f] + half * 8, ih[f] + half * 8, hn,
                     half ? w.e2_wih : w.e1_wih, half ? w.e2_whh : w.e1_whh,
                     half ? w.e2_bih : w.e1_bih, half ? w.e2_bhh : w.e1_bhh, 8, 8);
            memcpy(z[f] + half * 8, hn, sizeof hn);
            memcpy(ih[f] + half * 8, hn, sizeof hn);
        }
    dump_stage(g, j == 1 ? "dp1.inter_rnn" : "dp2.inter_rnn", z, FM, CH);

    // inter fc → LN → 残差 +intra_out → 转回 [C][F]
    for (int f = 0; f < FM; f++)
        for (int o = 0; o < CH; o++)
            fcbuf[f][o] = dot_(w.rfc_w + o * CH, z[f], CH) + w.rfc_b[o];
    memcpy(&z[0][0], &fcbuf[0][0], sizeof z);
    layernorm_528(&z[0][0], w.rln_w, w.rln_b);
    for (int c = 0; c < CH; c++)
        for (int f = 0; f < FM; f++) a33[c][f] = intra_out[f][c] + z[f][c];
}

}  // namespace

// ---- 逐帧处理：x/y 各 256 样本 ----
static void gtcrn_frame(va_gtcrn* g, const float* x, float* y) {
    static const int dils[3] = {1, 2, 5};
    char nm[16];
#ifdef VA_PROFILE
    g_prof.n++;
    double _tp = prof_now();
#endif

    // 1. STFT：窗口 = [hist | x]
    memcpy(g->wbuf, g->hist, HOP * sizeof(float));
    memcpy(g->wbuf + HOP, x, HOP * sizeof(float));
    memcpy(g->hist, x, HOP * sizeof(float));
    for (int i = 0; i < NFFT; i++) {
        g->fre[i] = g->wbuf[i] * g->win[i];
        g->fim[i] = 0.f;
    }
    g->fft.run(g->fre, g->fim, false);

    // 2. 特征 → ERB → SFE 9ch
    for (int f = 0; f < F257; f++) {
        float re = g->fre[f], im = g->fim[f];
        g->feat[0][f] = sqrtf(re * re + im * im + 1e-12f);
        g->feat[1][f] = re;
        g->feat[2][f] = im;
    }
    dump_stage(g, "feat", g->feat, 3, F257);
    for (int c = 0; c < 3; c++) {
        memcpy(g->bm3[c], g->feat[c], FL * sizeof(float));
        const float* src = g->feat[c] + FL;
        for (int b = 0; b < EB; b++)
            g->bm3[c][FL + b] = dot_(gw_erb_fc + (size_t)b * FH, src, FH);
    }
    dump_stage(g, "bm", g->bm3, 3, FB);
    sfe_band(&g->bm3[0][0], 3, FB, &g->s9[0][0]);
    dump_stage(g, "sfe9", g->s9, 9, FB);
    PSTEP(stft);

    // 3. en0：conv(1,5) s2 p2，129→65
    memset(g->hp9, 0, sizeof g->hp9);
    for (int c = 0; c < 9; c++) memcpy(g->hp9[c] + 2, g->s9[c], FB * sizeof(float));
    for (int o = 0; o < CH; o++) {
        const float* wr = gw_en0_w + (size_t)o * 45;
        for (int fo = 0; fo < 65; fo++) {
            float a = gw_en0_b[o];
            for (int ci = 0; ci < 9; ci++)
                for (int k = 0; k < 5; k++) a += wr[ci * 5 + k] * g->hp9[ci][2 * fo + k];
            g->a65[o][fo] = preluf_(a, gw_en0_act[0]);
        }
    }
    dump_stage(g, "en0", g->a65, CH, 65);
    memcpy(g->skip0, g->a65, sizeof g->skip0);
    PSTEP(erb);

    // 4. en1：g2 conv(1,5) s2 p2，65→33
    memset(g->hp16, 0, sizeof g->hp16);
    for (int c = 0; c < CH; c++) memcpy(g->hp16[c] + 2, g->a65[c], 65 * sizeof(float));
    for (int o = 0; o < CH; o++) {
        const float* wr = gw_en1_w + (size_t)o * 40;
        int og = (o / 8) * 8;
        for (int fo = 0; fo < FM; fo++) {
            float a = gw_en1_b[o];
            for (int ci = 0; ci < 8; ci++)
                for (int k = 0; k < 5; k++) a += wr[ci * 5 + k] * g->hp16[og + ci][2 * fo + k];
            g->a33[o][fo] = preluf_(a, gw_en1_act[0]);
        }
    }
    dump_stage(g, "en1", g->a33, CH, FM);
    memcpy(g->skip1, g->a33, sizeof g->skip1);
    PSTEP(en);

    // 5. GT 编码块 dil 1/2/5（skip 依序保存：gte0/1/2）
    for (int j = 0; j < 3; j++) {
        snprintf(nm, sizeof nm, "gte%d", j);
        gt_block(g, g->a33, 0, j, dils[j], g->ecc[j], g->eth[j], g->a33, nm);
        memcpy(g->skips[j], g->a33, sizeof g->skips[j]);
        dump_stage(g, nm, g->a33, CH, FM);
    }
    PSTEP(gt);

    // 6. DPGRNN ×2
    dpgrnn(g, g->a33, 1, g->ih[0]);
    dump_stage(g, "dp1", g->a33, CH, FM);
    dpgrnn(g, g->a33, 2, g->ih[1]);
    dump_stage(g, "dp2", g->a33, CH, FM);

    // 7. GT 解码块 dil 5/2/1，块外加 skip 残差（gte2/1/0）
    for (int j = 0; j < 3; j++) {
        for (int c = 0; c < CH; c++)
            for (int f = 0; f < FM; f++) g->xin[c][f] = g->a33[c][f] + g->skips[2 - j][c][f];
        snprintf(nm, sizeof nm, "gtd%d", j);
        gt_block(g, g->xin, 1, j, dils[2 - j], g->dcc[j], g->dth[j], g->a33, nm);
        dump_stage(g, nm, g->a33, CH, FM);
    }
    PSTEP(dp);

    // 8. de3：ConvT g2 (1,5) s2 p2，33→65（先加 skip en1）。
    // 输出按偶奇拆分（o=2j−2+k → k≡o mod 2），tap 变连续 axpy。
    for (int c = 0; c < CH; c++)
        for (int f = 0; f < FM; f++) g->xin[c][f] = g->a33[c][f] + g->skip1[c][f];
    dump_stage(g, "de3_in", g->xin, CH, FM);
    for (int c = 0; c < CH; c++) {
        memset(g->xp33[c], 0, sizeof g->xp33[c]);
        memcpy(&g->xp33[c][1], g->xin[c], FM * sizeof(float));
    }
    for (int o = 0; o < CH; o++) {
        int g0 = (o / 8) * 8, oc = o % 8;
        float* ye = g->de3e[o];
        float* yo = g->de3o[o];
        for (int m = 0; m < 33; m++) ye[m] = gw_de3_b[o];
        for (int m = 0; m < 32; m++) yo[m] = gw_de3_b[o];
        for (int ci = g0; ci < g0 + 8; ci++) {
            const float* wr = gw_de3_w + (size_t)ci * 40 + oc * 5;
            axpy_(ye, g->xp33[ci], wr[4], 33);
            axpy_(ye, g->xp33[ci] + 1, wr[2], 33);
            axpy_(ye, g->xp33[ci] + 2, wr[0], 33);
            axpy_(yo, g->xp33[ci] + 1, wr[3], 32);
            axpy_(yo, g->xp33[ci] + 2, wr[1], 32);
        }
        for (int m = 0; m < 32; m++) {
            g->a65[o][2 * m] = preluf_(ye[m], gw_de3_act[0]);
            g->a65[o][2 * m + 1] = preluf_(yo[m], gw_de3_act[0]);
        }
        g->a65[o][64] = preluf_(ye[32], gw_de3_act[0]);
    }
    dump_stage(g, "de3", g->a65, CH, 65);

    // 9. de4：ConvT (1,5) s2 p2，65→129 + tanh（先加 skip en0，偶奇拆分）
    for (int c = 0; c < CH; c++)
        for (int f = 0; f < 65; f++) g->a65[c][f] += g->skip0[c][f];
    for (int c = 0; c < CH; c++) {
        memset(g->xp65[c], 0, sizeof g->xp65[c]);
        memcpy(&g->xp65[c][1], g->a65[c], 65 * sizeof(float));
    }
    for (int o = 0; o < 2; o++) {
        float* ye = g->de4e[o];
        float* yo = g->de4o[o];
        for (int m = 0; m < 65; m++) ye[m] = gw_de4_b[o];
        for (int m = 0; m < 64; m++) yo[m] = gw_de4_b[o];
        for (int ci = 0; ci < CH; ci++) {
            const float* wr = gw_de4_w + (size_t)ci * 10 + o * 5;
            axpy_(ye, g->xp65[ci], wr[4], 65);
            axpy_(ye, g->xp65[ci] + 1, wr[2], 65);
            axpy_(ye, g->xp65[ci] + 2, wr[0], 65);
            axpy_(yo, g->xp65[ci] + 1, wr[3], 64);
            axpy_(yo, g->xp65[ci] + 2, wr[1], 64);
        }
        for (int m = 0; m < 64; m++) {
            g->m129[o][2 * m] = tanhf_(ye[m]);
            g->m129[o][2 * m + 1] = tanhf_(yo[m]);
        }
        g->m129[o][128] = tanhf_(ye[64]);
    }
    dump_stage(g, "de4", g->m129, 2, 129);

    // 10. iERB → 257 + CRM
    float m[2][F257];
    for (int c = 0; c < 2; c++) {
        memcpy(m[c], g->m129[c], FL * sizeof(float));
        for (int k = 0; k < FH; k++)
            m[c][FL + k] = dot_(gw_ierb_fc + (size_t)k * EB, g->m129[c] + FL, EB);
    }
    dump_stage(g, "m", m, 2, F257);
    float enh_r[F257], enh_i[F257];
    for (int f = 0; f < F257; f++) {
        float sr = g->fre[f], si = g->fim[f];
        enh_r[f] = sr * m[0][f] - si * m[1][f];
        enh_i[f] = si * m[0][f] + sr * m[1][f];
    }
    {
        static thread_local float enh2[F257][2];
        for (int f = 0; f < F257; f++) {
            enh2[f][0] = enh_r[f];
            enh2[f][1] = enh_i[f];
        }
        dump_stage(g, "enh", enh2, F257, 2);
    }
    PSTEP(de);

    // 11. iFFT + WOLA：帧 t 定稿时间线 [256t, 256t+256)（帧 t-1 后半 +
    // 帧 t 前半）→ 发前半；后半滚动为下一帧缓冲前半
    for (int f = 0; f < F257; f++) {
        g->fre[f] = enh_r[f];
        g->fim[f] = enh_i[f];
        if (f > 0 && f < 256) {
            g->fre[NFFT - f] = enh_r[f];
            g->fim[NFFT - f] = -enh_i[f];
        }
    }
    g->fim[0] = g->fim[256] = 0.f;
    g->fft.run(g->fre, g->fim, true);
    for (int i = 0; i < NFFT; i++) g->ola[i] += g->fre[i] * g->win[i];
    memcpy(y, g->ola, HOP * sizeof(float));
    memcpy(g->ola, g->ola + HOP, HOP * sizeof(float));
    memset(g->ola + HOP, 0, HOP * sizeof(float));
    PSTEP(ifft);

    g->frame++;
}

// ---- C ABI ----

va_gtcrn* va_gtcrn_create(void) {
    va_gtcrn* g = new (std::nothrow) va_gtcrn();
    if (!g) return nullptr;
    va_gtcrn_reset(g);
    for (int i = 0; i < NFFT; i++)
        g->win[i] = sqrtf(0.5f * (1.f - cosf(2.f * (float)M_PI * i / NFFT)));

    // GT 块权重表（tra GRU 转置：wih (48,8)→(8,48)，whh (48,16)→(16,48)）
    static const struct {
        const float *pc1_w, *pc1_b, *pc1_a, *dw_w, *dw_b, *dw_a, *pc2_w, *pc2_b;
        const float *wih, *whh, *bih, *bhh, *fc_w, *fc_b;
    } gsrc[2][3] = {
        {{gw_gte0_pc1_w, gw_gte0_pc1_b, gw_gte0_pc1_act, gw_gte0_dw_w, gw_gte0_dw_b,
          gw_gte0_dw_act, gw_gte0_pc2_w, gw_gte0_pc2_b, gw_gte0_tra_wih,
          gw_gte0_tra_whh, gw_gte0_tra_bih, gw_gte0_tra_bhh, gw_gte0_tra_fc_w,
          gw_gte0_tra_fc_b},
         {gw_gte1_pc1_w, gw_gte1_pc1_b, gw_gte1_pc1_act, gw_gte1_dw_w, gw_gte1_dw_b,
          gw_gte1_dw_act, gw_gte1_pc2_w, gw_gte1_pc2_b, gw_gte1_tra_wih,
          gw_gte1_tra_whh, gw_gte1_tra_bih, gw_gte1_tra_bhh, gw_gte1_tra_fc_w,
          gw_gte1_tra_fc_b},
         {gw_gte2_pc1_w, gw_gte2_pc1_b, gw_gte2_pc1_act, gw_gte2_dw_w, gw_gte2_dw_b,
          gw_gte2_dw_act, gw_gte2_pc2_w, gw_gte2_pc2_b, gw_gte2_tra_wih,
          gw_gte2_tra_whh, gw_gte2_tra_bih, gw_gte2_tra_bhh, gw_gte2_tra_fc_w,
          gw_gte2_tra_fc_b}},
        {{gw_gtd0_pc1_w, gw_gtd0_pc1_b, gw_gtd0_pc1_act, gw_gtd0_dw_w, gw_gtd0_dw_b,
          gw_gtd0_dw_act, gw_gtd0_pc2_w, gw_gtd0_pc2_b, gw_gtd0_tra_wih,
          gw_gtd0_tra_whh, gw_gtd0_tra_bih, gw_gtd0_tra_bhh, gw_gtd0_tra_fc_w,
          gw_gtd0_tra_fc_b},
         {gw_gtd1_pc1_w, gw_gtd1_pc1_b, gw_gtd1_pc1_act, gw_gtd1_dw_w, gw_gtd1_dw_b,
          gw_gtd1_dw_act, gw_gtd1_pc2_w, gw_gtd1_pc2_b, gw_gtd1_tra_wih,
          gw_gtd1_tra_whh, gw_gtd1_tra_bih, gw_gtd1_tra_bhh, gw_gtd1_tra_fc_w,
          gw_gtd1_tra_fc_b},
         {gw_gtd2_pc1_w, gw_gtd2_pc1_b, gw_gtd2_pc1_act, gw_gtd2_dw_w, gw_gtd2_dw_b,
          gw_gtd2_dw_act, gw_gtd2_pc2_w, gw_gtd2_pc2_b, gw_gtd2_tra_wih,
          gw_gtd2_tra_whh, gw_gtd2_tra_bih, gw_gtd2_tra_bhh, gw_gtd2_tra_fc_w,
          gw_gtd2_tra_fc_b}}};
    for (int side = 0; side < 2; side++)
        for (int j = 0; j < 3; j++) {
            const auto& q = gsrc[side][j];
            GTW& w = g->gtw[side][j];
            w.pc1_w = q.pc1_w; w.pc1_b = q.pc1_b; w.pc1_a = q.pc1_a;
            w.dw_w = q.dw_w; w.dw_b = q.dw_b; w.dw_a = q.dw_a;
            w.pc2_w = q.pc2_w; w.pc2_b = q.pc2_b;
            w.tra_wih = q.wih;
            w.tra_whh = q.whh;
            w.tra_bih = q.bih; w.tra_bhh = q.bhh;
            w.tra_fc_w = q.fc_w; w.tra_fc_b = q.fc_b;
        }

    // DPGRNN 权重表（intra (12,8)/(12,4)、inter (24,8)/(24,8) 转置）
    struct G4 { const float *wih, *whh, *bih, *bhh; };
    static const G4 dsrc[2][6] = {
        {{gw_dp1_intra1f_wih, gw_dp1_intra1f_whh, gw_dp1_intra1f_bih, gw_dp1_intra1f_bhh},
         {gw_dp1_intra1b_wih, gw_dp1_intra1b_whh, gw_dp1_intra1b_bih, gw_dp1_intra1b_bhh},
         {gw_dp1_intra2f_wih, gw_dp1_intra2f_whh, gw_dp1_intra2f_bih, gw_dp1_intra2f_bhh},
         {gw_dp1_intra2b_wih, gw_dp1_intra2b_whh, gw_dp1_intra2b_bih, gw_dp1_intra2b_bhh},
         {gw_dp1_inter1_wih, gw_dp1_inter1_whh, gw_dp1_inter1_bih, gw_dp1_inter1_bhh},
         {gw_dp1_inter2_wih, gw_dp1_inter2_whh, gw_dp1_inter2_bih, gw_dp1_inter2_bhh}},
        {{gw_dp2_intra1f_wih, gw_dp2_intra1f_whh, gw_dp2_intra1f_bih, gw_dp2_intra1f_bhh},
         {gw_dp2_intra1b_wih, gw_dp2_intra1b_whh, gw_dp2_intra1b_bih, gw_dp2_intra1b_bhh},
         {gw_dp2_intra2f_wih, gw_dp2_intra2f_whh, gw_dp2_intra2f_bih, gw_dp2_intra2f_bhh},
         {gw_dp2_intra2b_wih, gw_dp2_intra2b_whh, gw_dp2_intra2b_bih, gw_dp2_intra2b_bhh},
         {gw_dp2_inter1_wih, gw_dp2_inter1_whh, gw_dp2_inter1_bih, gw_dp2_inter1_bhh},
         {gw_dp2_inter2_wih, gw_dp2_inter2_whh, gw_dp2_inter2_bih, gw_dp2_inter2_bhh}}};
    for (int j = 0; j < 2; j++) {
        DPW& w = g->dpw[j];
        const G4* q = dsrc[j];
        w.i1f_wih = q[0].wih;
        w.i1f_whh = q[0].whh;
        w.i1f_bih = q[0].bih; w.i1f_bhh = q[0].bhh;
        w.i1b_wih = q[1].wih;
        w.i1b_whh = q[1].whh;
        w.i1b_bih = q[1].bih; w.i1b_bhh = q[1].bhh;
        w.i2f_wih = q[2].wih;
        w.i2f_whh = q[2].whh;
        w.i2f_bih = q[2].bih; w.i2f_bhh = q[2].bhh;
        w.i2b_wih = q[3].wih;
        w.i2b_whh = q[3].whh;
        w.i2b_bih = q[3].bih; w.i2b_bhh = q[3].bhh;
        w.e1_wih = q[4].wih;
        w.e1_whh = q[4].whh;
        w.e1_bih = q[4].bih; w.e1_bhh = q[4].bhh;
        w.e2_wih = q[5].wih;
        w.e2_whh = q[5].whh;
        w.e2_bih = q[5].bih; w.e2_bhh = q[5].bhh;
        w.ifc_w = j ? gw_dp2_intra_fc_w : gw_dp1_intra_fc_w;
        w.ifc_b = j ? gw_dp2_intra_fc_b : gw_dp1_intra_fc_b;
        w.iln_w = j ? gw_dp2_intra_ln_w : gw_dp1_intra_ln_w;
        w.iln_b = j ? gw_dp2_intra_ln_b : gw_dp1_intra_ln_b;
        w.rfc_w = j ? gw_dp2_inter_fc_w : gw_dp1_inter_fc_w;
        w.rfc_b = j ? gw_dp2_inter_fc_b : gw_dp1_inter_fc_b;
        w.rln_w = j ? gw_dp2_inter_ln_w : gw_dp1_inter_ln_w;
        w.rln_b = j ? gw_dp2_inter_ln_b : gw_dp1_inter_ln_b;
    }

    if (const char* d = getenv("VA_GTCRN_DUMP")) {
        snprintf(g->dump_dir, sizeof g->dump_dir, "%s", d);
        g->dump_frames = getenv("VA_GTCRN_DUMP_FRAMES")
                             ? atoi(getenv("VA_GTCRN_DUMP_FRAMES")) : 8;
    }
    return g;
}

void va_gtcrn_destroy(va_gtcrn* g) { delete g; }

void va_gtcrn_reset(va_gtcrn* g) {
    memset(g->hist, 0, sizeof g->hist);
    memset(g->ola, 0, sizeof g->ola);
    memset(g->ecc, 0, sizeof g->ecc);
    memset(g->dcc, 0, sizeof g->dcc);
    memset(g->eth, 0, sizeof g->eth);
    memset(g->dth, 0, sizeof g->dth);
    memset(g->ih, 0, sizeof g->ih);
    g->frame = 0;
}

int va_gtcrn_process(va_gtcrn* g, const float* x, float* y, int n) {
    if (!g || !x || !y || n <= 0 || n % HOP) return -1;
    for (int off = 0; off < n; off += HOP)
        gtcrn_frame(g, x + off, y + off);
    return n / HOP;
}

#ifdef VA_PROFILE
void va_gtcrn_prof_print(void) {
    if (!g_prof.n) return;
    double tot = g_prof.stft + g_prof.erb + g_prof.en + g_prof.gt + g_prof.dp +
                 g_prof.de + g_prof.ifft;
    printf("profile/帧 us: stft %.0f erb %.0f en %.0f gt %.0f dp %.0f de %.0f ifft %.0f"
           " | 合计 %.0f\n",
           g_prof.stft / g_prof.n * 1e6, g_prof.erb / g_prof.n * 1e6,
           g_prof.en / g_prof.n * 1e6, g_prof.gt / g_prof.n * 1e6,
           g_prof.dp / g_prof.n * 1e6, g_prof.de / g_prof.n * 1e6,
           g_prof.ifft / g_prof.n * 1e6, tot / g_prof.n * 1e6);
}
#endif
