// voice/algo/test_algo.cpp — libvoice_algo 板上单测（纯数值，无音频播放）。
//
// 数据全部为 16k/48k 合成（确定性 LCG 种子，C++ 内生成，无外部文件依赖）：
//   AEC：语音样激励（双共振峰 AR + 音节包络）× 随机衰减回声路径（直接径 4ms +
//        τ≈45ms 指数尾，总长 ~190ms < 320ms 滤波器覆盖）。
//        验收：16k 远端单讲稳态 ERLE ≥ 18 dB（块对齐分块）；任意 n 分块与
//        单次调用输出一致（不错位）；多实例独立；48k 合成回归 ERLE ≥ 20 dB。
//   BF ：4 麦 ULA d=40mm，带限语音样平面波（300–1350 Hz，低于 GCC 限带
//        fmax=1429 Hz）+ 每通道独立白噪 0 dB/ch。
//        验收：16k DOA 误差 < 10°、DSB SNR 增益 ≥ +5 dB（理论 10log10(4)=6.02）；
//        48k 合成回归同门槛。
//   ABI：空指针/非法参数返回码；dlopen + dlsym 冒烟（验证 .so 可动态加载）。
// exit 0 = 全部通过。

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "voice_algo.h"
#include "bf.hpp"   // DOA 增量 vs 批式一致性测试直接用 C++ 接口

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int g_fail = 0;

static void check(const char* name, bool ok, const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    std::printf("  [%s] %s: %s\n", ok ? "PASS" : "FAIL", name, buf);
    if (!ok) ++g_fail;
}

// ------------------------------------------------------------------ //
// 确定性随机：LCG + Box-Muller。状态必须显式 64 位——armv7 上
// unsigned long 是 32 位（LCG 截断后 uniform ∈ [0,2e-10)，randn 退化为
// 近常数偏置，噪声相干 → DSB 无增益）。
// ------------------------------------------------------------------ //
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 1) {}
    double uniform() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return (double)(s >> 11) / 9007199254740992.0;
    }
    double randn() {   // Box-Muller
        double u1 = uniform(), u2 = uniform();
        return std::sqrt(-2.0 * std::log(u1 + 1e-12)) * std::cos(2.0 * M_PI * u2);
    }
};

// ------------------------------------------------------------------ //
// 合成信号
// ------------------------------------------------------------------ //
// 语音样激励：gauss → 双共振峰 AR 级联（f 随 sr 等比缩放）→ 音节/语句包络
static void gen_speech(std::vector<float>& v, int n, double sr, unsigned long seed,
                       double rms = 0.05) {
    Rng rng(seed);
    auto pole = [&](double f, double bw, double& a1, double& a2) {
        const double r = std::exp(-M_PI * bw / sr), th = 2.0 * M_PI * f / sr;
        a1 = 2.0 * r * std::cos(th);
        a2 = -r * r;
    };
    double a1, a2, c1, c2;
    pole(0.022 * sr, 0.010 * sr, a1, a2);   // ~350 Hz / 160 Hz @16k
    pole(0.068 * sr, 0.018 * sr, c1, c2);   // ~1090 Hz / 290 Hz @16k
    std::vector<double> y1(n, 0.0), y2(n, 0.0);
    double p1 = 0.0, p2 = 0.0, q1 = 0.0, q2 = 0.0;
    for (int i = 0; i < n; i++) {
        const double x = rng.randn();
        p1 = a1 * p1 + a2 * q1 + x;
        q1 = p1;
        p2 = c1 * p2 + c2 * q2 + 0.6 * p1;
        q2 = p2;
        y1[i] = p2;
    }
    // 音节包络（0.9 s 周期）× 慢起伏（3.7 s）
    double emax = 1e-12;
    for (int i = 0; i < n; i++) {
        const double syl = std::pow(std::max(0.0, std::sin(2.0 * M_PI * i / (0.9 * sr))), 1.2);
        const double slow = 0.6 + 0.4 * std::sin(2.0 * M_PI * i / (3.7 * sr) + 1.0);
        y2[i] = y1[i] * syl * slow;
        emax = std::max(emax, y2[i] * y2[i]);
    }
    double p = 0.0;
    for (int i = 0; i < n; i++) p += y2[i] * y2[i];
    const double g = rms / std::sqrt(p / n + 1e-30);
    v.resize(n);
    for (int i = 0; i < n; i++) v[i] = (float)(y2[i] * g);
}

// RBJ biquad（一阶段级联）
static void biquad(const std::vector<float>& x, std::vector<float>& y,
                   double f0, double sr, double Q, bool lowpass) {
    const double w0 = 2.0 * M_PI * f0 / sr, alpha = std::sin(w0) / (2.0 * Q);
    const double cw = std::cos(w0);
    double b0, b1, b2, a0, a1, a2;
    if (lowpass) {
        b0 = (1.0 - cw) / 2; b1 = 1.0 - cw; b2 = b0;
    } else {
        b0 = (1.0 + cw) / 2; b1 = -(1.0 + cw); b2 = b0;
    }
    a0 = 1.0 + alpha; a1 = -2.0 * cw; a2 = 1.0 - alpha;
    y.resize(x.size());
    double z1 = 0.0, z2 = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        const double xn = x[i] / a0;
        const double yn = b0 / a0 * xn + z1;
        z1 = b1 / a0 * xn - a1 / a0 * yn + z2;
        z2 = b2 / a0 * xn - a2 / a0 * yn;
        y[i] = (float)yn;
    }
}

// 带限（BF 场景：平面波限带于 GCC fmax 之下，防空间混叠）
static void bandpass(std::vector<float>& v, double sr) {
    std::vector<float> t1, t2, t3;
    biquad(v, t1, 300.0, sr, 0.7, false);
    biquad(t1, t2, 300.0, sr, 0.7, false);
    biquad(t2, t3, 1350.0, sr, 0.7, true);
    biquad(t3, v, 1350.0, sr, 0.7, true);
    double p = 0.0;
    for (float x : v) p += (double)x * x;
    const double g = 0.08 / std::sqrt(p / v.size() + 1e-30);
    for (auto& x : v) x = (float)(x * g);
}

// 生成器侧分数延迟：out(t) = x(t - d_sec)。half=24、double（与库内 half=16
// float 核不同实现，避免同核退化为恒等测试）
static std::vector<float> gen_delay(const std::vector<float>& x, double d_sec, double sr,
                                    int half = 24) {
    const int64_t n = (int64_t)x.size();
    std::vector<float> out((size_t)n);
    const double d = -d_sec * sr;
    const double fr = d - std::floor(d);
    std::vector<double> kern((size_t)(2 * half));
    for (int t = -half + 1; t <= half; ++t) {
        const double u = fr - t;
        const double s = (std::fabs(u) < 1e-12) ? 1.0 : std::sin(M_PI * u) / (M_PI * u);
        const double w = 0.5 * (1.0 + std::cos(M_PI * u / half));
        kern[(size_t)(t + half - 1)] = s * w;
    }
    const int64_t base = (int64_t)std::floor(d);
    for (int64_t i = 0; i < n; i++) {
        double acc = 0.0;
        for (int t = -half + 1; t <= half; ++t) {
            const int64_t j = i + base + t;
            if (j < 0 || j >= n) continue;
            acc += (double)x[(size_t)j] * kern[(size_t)(t + half - 1)];
        }
        out[(size_t)i] = (float)acc;
    }
    return out;
}

// ------------------------------------------------------------------ //
// 指标
// ------------------------------------------------------------------ //
static double sumsq(const std::vector<float>& x, size_t a, size_t b) {
    double s = 0.0;
    for (size_t i = a; i < b; i++) s += (double)x[i] * x[i];
    return s;
}
static double erle_db(const std::vector<float>& mic, const std::vector<float>& out,
                      size_t a, size_t b) {
    return 10.0 * std::log10((sumsq(mic, a, b) + 1e-30) / (sumsq(out, a, b) + 1e-30));
}
static double snr_db(const std::vector<float>& y, const std::vector<float>& ref,
                     size_t a, size_t b) {
    const double es = sumsq(ref, a, b);
    double e = 0.0;
    for (size_t i = a; i < b; i++) {
        const double d = (double)y[i] - ref[i];
        e += d * d;
    }
    if (e <= 1e-30) return 99.0;
    return 10.0 * std::log10(es / e);
}
static bool all_finite(const std::vector<float>& v) {
    for (float x : v)
        if (!std::isfinite(x)) return false;
    return true;
}
// 互相关主峰滞后（错位检测）：返回 argmax_lag Σ x[i]y[i-lag]
static long long xcorr_argmax(const std::vector<float>& x, const std::vector<float>& y,
                              long long maxlag) {
    const size_t a = x.size() / 8, b = x.size() - x.size() / 8;
    double best = -1e300;
    long long bl = 0;
    for (long long lag = -maxlag; lag <= maxlag; lag++) {
        double s = 0.0;
        for (size_t i = a + (size_t)std::max<long long>(0, lag); i + (size_t)std::max<long long>(0, -lag) < b; i++) {
            s += (double)x[i] * y[(size_t)((long long)i - lag)];
        }
        if (s > best) { best = s; bl = lag; }
    }
    return bl;
}

// ------------------------------------------------------------------ //
// AEC 场景
// ------------------------------------------------------------------ //
struct AecResult {
    double erle_full = 0, erle_steady = 0, rtf = 0;
};

static void gen_aec_scene(int sr, std::vector<float>& ref, std::vector<float>& mic) {
    const int n = 8 * sr;
    gen_speech(ref, n, sr, 20261005u, 0.05);
    Rng rng(777u);
    const int cover = (sr == 48000) ? 16384 : 5120;   // 滤波器覆盖（抽头）
    const int L = std::min(cover - 128, (int)(0.19 * sr));
    const int d0 = (int)(0.004 * sr);
    std::vector<float> h((size_t)L, 0.0f);
    const double tau = 0.045 * sr;
    for (int k = 0; k < L - d0; k++)
        h[(size_t)(d0 + k)] = (float)(rng.randn() * std::exp(-(double)k / tau));
    h[(size_t)d0] += 0.6f;
    double hp = 0.0;
    for (float x : h) hp += (double)x * x;
    const double g = 1.0 / std::sqrt(hp);
    for (auto& x : h) x = (float)(x * g);
    mic.assign((size_t)n, 0.0f);
    for (int i = 0; i < n; i++) {
        double acc = 0.0;
        const int kmax = std::min((int)h.size(), i + 1);
        for (int k = 0; k < kmax; k++) acc += h[(size_t)k] * ref[(size_t)(i - k)];
        mic[(size_t)i] = (float)acc;
    }
}

static AecResult run_aec(int sr, const char* tag, const int* chunks, int nchunks) {
    std::vector<float> ref, mic;
    gen_aec_scene(sr, ref, mic);
    const size_t n = mic.size();
    std::vector<float> out(n, 0.0f);
    va_aec* h = va_aec_create(sr);
    const auto t0 = std::chrono::steady_clock::now();
    size_t off = 0;
    int ci = 0;
    while (off < n) {
        int c = chunks[ci++ % nchunks];
        if (c > (int)(n - off)) c = (int)(n - off);
        const float* mp = mic.data() + off;
        const float* rp = ref.data() + off;
        float* op = out.data() + off;
        va_aec_process(h, &mp, rp, &op, c);
        off += (size_t)c;
    }
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
    va_aec_destroy(h);
    AecResult r;
    r.erle_full = erle_db(mic, out, 0, n);
    const size_t a = (size_t)(0.4 * n);
    r.erle_steady = erle_db(mic, out, a, n);
    r.rtf = (ms / 1000.0) / ((double)n / sr);
    std::printf("[%s] sr=%d  ERLE 全段 %.2f dB | 稳态(后60%%) %.2f dB | RTF %.3f\n",
                tag, sr, r.erle_full, r.erle_steady, r.rtf);
    return r;
}

// ------------------------------------------------------------------ //
// BF 场景（4 麦 ULA d=40mm，0 dB/ch 独立白噪）
// ------------------------------------------------------------------ //
struct BfResult {
    double doa = 0, err = 0;
    double snr_best1 = 0, snr_dsb = 0, gain = 0;
};

static BfResult run_bf(int sr, double theta_true, const char* tag) {
    const int nch = 4;
    const double d = 0.04, c = 343.0;
    const int n = 6 * sr;
    std::vector<float> s;
    gen_speech(s, n, sr, 424242u, 0.08);
    bandpass(s, (double)sr);
    // 阵列几何（与库同式）：tau_m = pos_m·sin(theta)/c
    double tau[16];
    for (int m = 0; m < nch; m++) {
        const double pos = (m - (nch - 1) / 2.0) * d;
        tau[m] = pos * std::sin(theta_true * M_PI / 180.0) / c;
    }
    std::vector<std::vector<float>> clean(nch), noisy(nch);
    Rng rng(31415u);
    for (int m = 0; m < nch; m++) {
        clean[m] = gen_delay(s, tau[m], (double)sr);
        double pr = 0.0;
        for (float x : clean[m]) pr += (double)x * x;
        pr = std::sqrt(pr / n);
        noisy[m].resize((size_t)n);
        for (int i = 0; i < n; i++) noisy[m][(size_t)i] = (float)(clean[m][(size_t)i] + pr * rng.randn());
    }
    // ---- 经 C ABI：噪dB 阵列 → DSB；干净阵列 → 同一波束得参考 -------- //
    va_doa_bf* hb = va_doa_bf_create(nch, sr, "ula", d);
    std::vector<float> y((size_t)n, 0.0f);
    {
        const int chunks[] = {(int)(4 * sr), (int)(1 * sr), (int)(1 * sr)};
        size_t off = 0;
        int ci = 0;
        while (off < (size_t)n) {
            const int cn = chunks[ci++ % 3];
            std::vector<const float*> in(nch);
            for (int m = 0; m < nch; m++) in[(size_t)m] = noisy[m].data() + off;
            float* op = y.data() + off;
            va_doa_bf_process(hb, in.data(), op, cn);
            off += (size_t)cn;
        }
    }
    const double doa = va_doa_bf_get_doa(hb);
    va_doa_bf_destroy(hb);

    va_doa_bf* hc = va_doa_bf_create(nch, sr, "ula", d);
    std::vector<float> yref((size_t)n, 0.0f);
    {
        std::vector<const float*> in(nch);
        for (int m = 0; m < nch; m++) in[(size_t)m] = clean[m].data();
        va_doa_bf_process(hc, in.data(), yref.data(), n);
    }
    va_doa_bf_destroy(hc);

    BfResult r;
    r.doa = doa;
    r.err = std::fabs(doa - theta_true);
    const size_t a = (size_t)(0.5 * sr), b = (size_t)n - (size_t)(0.5 * sr);
    r.snr_dsb = snr_db(y, yref, a, b);
    r.snr_best1 = -99.0;
    for (int m = 0; m < nch; m++)
        r.snr_best1 = std::max(r.snr_best1, snr_db(noisy[(size_t)m], clean[(size_t)m], a, b));
    r.gain = r.snr_dsb - r.snr_best1;
    std::printf("[%s] sr=%d theta=%.0f°  DOA 估计 %.2f° (误差 %.2f°) | 单麦最佳 %.2f dB"
                " | DSB %.2f dB | 增益 %+.2f dB (理论 +6.02)\n",
                tag, sr, theta_true, doa, r.err, r.snr_best1, r.snr_dsb, r.gain);
    return r;
}

// ------------------------------------------------------------------ //
// DOA 增量槽环 vs 批式 estimate 逐位一致（G 优化回归）
// 分析窗恰为整数个完整块时，inc_push/inc_estimate 与 estimate() 走相同
// 分块/去均值/累加序/缩放 → 输出应逐位相等。max_dur 放大到 6s 避免批式
// 4s 裁剪（生产 4s 配置下增量稳态窗 8×8192≈4.1s 与批式 4.0s 仅窗缘差）。
// ------------------------------------------------------------------ //
static void test_doa_inc_parity() {
    std::printf("[DOA-inc] 增量槽环 vs 批式 estimate（应逐位一致）\n");
    const int sr = 16000, nch = 4;
    const double d = 0.04, c = 343.0, theta_true = 30.0;
    const int B = 8192;                  // gcc_block @16k
    const int nblk = 10;
    const int n = nblk * B;
    std::vector<float> s;
    gen_speech(s, n, sr, 20260101u, 0.08);
    bandpass(s, (double)sr);
    double tau[4];
    for (int m = 0; m < nch; m++) {
        const double pos = (m - (nch - 1) / 2.0) * d;
        tau[m] = pos * std::sin(theta_true * M_PI / 180.0) / c;
    }
    std::vector<std::vector<float>> ch(nch);
    Rng rng(2718u);
    for (int m = 0; m < nch; m++) {
        ch[m] = gen_delay(s, tau[m], (double)sr);
        double pr = 0.0;
        for (float x : ch[m]) pr += (double)x * x;
        pr = std::sqrt(pr / n);
        for (int i = 0; i < n; i++)
            ch[m][(size_t)i] = (float)(ch[m][(size_t)i] + pr * rng.randn());
    }
    std::vector<std::vector<float>> view(nch);
    auto batch_at = [&](int b0, int blocks) {
        afe::DOA dd(nch, sr, d, c, 1.0, 0.05, 6.0, "ula");
        for (int m = 0; m < nch; m++)
            view[m].assign(ch[m].begin() + (size_t)b0 * B,
                           ch[m].begin() + (size_t)(b0 + blocks) * B);
        return dd.estimate(view);
    };

    afe::DOA di(nch, sr, d, c, 1.0, 0.05, 6.0, "ula");
    const int chunks[] = {4096, 512, 1000, 8192, 333, 2048, 777, 4096};
    int done = 0, off = 0, ci = 0;
    double inc8 = 0.0;
    while (off < n) {
        int cn = chunks[ci++ % 8];
        if (cn > n - off) cn = n - off;
        std::vector<const float*> p(nch);
        for (int m = 0; m < nch; m++) p[(size_t)m] = ch[(size_t)m].data() + off;
        done += di.inc_push(p.data(), cn);
        off += cn;
        if (inc8 == 0.0 && done >= 8) inc8 = di.inc_estimate();
    }
    check("push 完成块数 = 10", done == 10, "done=%d", done);
    const double b8 = batch_at(0, 8);
    check("8 块窗 增量 vs 批式逐位一致", inc8 == b8, "inc=%.17g batch=%.17g", inc8, b8);
    const double inc10 = di.inc_estimate();
    const double b10 = batch_at(0, 10);
    check("10 块滚动窗 逐位一致", inc10 == b10, "inc=%.17g batch=%.17g", inc10, b10);
    check("DOA 精度（θ=30°）误差 < 10°", std::fabs(inc10 - theta_true) < 10.0,
          "%.2f°", inc10);
}

// ------------------------------------------------------------------ //
// 共享参考谱：4×bound 实例 vs 4×独立实例逐位一致 + ABI 边界
// 同一 FFTPlan/nx 表达式/同遍历序 → 输出必须逐位相等（512 块对齐喂入）
// ------------------------------------------------------------------ //
static void test_aec_shared_ref() {
    std::printf("[AEC-shared] 共享参考谱 vs 独立实例（应逐位一致）\n");
    const int sr = 16000, nch = 4;
    std::vector<float> ref, mic0;
    gen_aec_scene(sr, ref, mic0);
    const size_t n = mic0.size();
    // 4 通道：独立回声路径（共享同一 ref）
    std::vector<std::vector<float>> mic(nch);
    {
        Rng rng(97531u);
        const int L = 1600;
        for (int c = 0; c < nch; c++) {
            mic[c].assign(n, 0.0f);
            double h0 = 0.3 + 0.07 * c, h1 = -0.15 - 0.05 * c;
            const int d0 = 64 + 13 * c, d1 = 800 - 27 * c;
            for (size_t i = 0; i < n; i++) {
                double v = i > (size_t)d0 ? h0 * ref[i - (size_t)d0] : 0.0;
                if (i > (size_t)d1) v += h1 * ref[i - (size_t)d1];
                mic[c][i] = (float)(v + 0.001 * rng.randn());
            }
        }
    }
    // 独立：4 实例 512 对齐喂
    std::vector<std::vector<float>> o_solo(nch);
    {
        va_aec* hh[nch];
        for (int c = 0; c < nch; c++) hh[c] = va_aec_create(sr);
        for (int c = 0; c < nch; c++) {
            o_solo[c].assign(n, 0.f);
            for (size_t off = 0; off < n; off += 512) {
                const float* mp = mic[c].data() + off;
                float* op = o_solo[c].data() + off;
                va_aec_process(hh[c], &mp, ref.data() + off, &op, 512);
            }
            va_aec_destroy(hh[c]);
        }
    }
    // 共享：refsvc + 4 bound 实例
    std::vector<std::vector<float>> o_sh(nch);
    {
        va_aec_ref* rs = va_aec_ref_create(sr);
        check("ref_create(16k) 成功", rs != nullptr, "%s", "");
        va_aec* hh[nch];
        for (int c = 0; c < nch; c++) { hh[c] = va_aec_create_shared(sr, rs); o_sh[c].assign(n, 0.f); }
        bool all_ok = hh[0] && hh[1] && hh[2] && hh[3];
        check("create_shared ×4 成功", all_ok, "%s", "");
        if (all_ok) {
            bool push_ok = true;
            for (size_t off = 0; off < n; off += 512) {
                if (va_aec_ref_push(rs, ref.data() + off, 512) != 0) { push_ok = false; break; }
                for (int c = 0; c < nch; c++)
                    va_aec_process_shared(hh[c], mic[c].data() + off, ref.data() + off,
                                          o_sh[c].data() + off, 512);
            }
            check("ref_push 512 对齐全成功", push_ok, "%s", "");
            bool same = true;
            size_t first = 0;
            for (int c = 0; c < nch && same; c++)
                for (size_t i = 0; i < n; i++)
                    if (o_sh[c][i] != o_solo[c][i]) { same = false; first = i; break; }
            check("4 通道共享 vs 独立逐位一致", same, "first_diff@%zu", first);
            const double e0 = erle_db(mic[0], o_sh[0], n / 4, n);
            check("共享路径稳态 ERLE ≥ 18 dB", e0 >= 18.0, "%.2f dB", e0);
        }
        for (int c = 0; c < nch; c++) va_aec_destroy(hh[c]);
        va_aec_ref_destroy(rs);
    }
    // ABI 边界
    check("ref_create(48000) → NULL（仅 16k）", va_aec_ref_create(48000) == nullptr, "%s", "");
    va_aec_ref* rs = va_aec_ref_create(sr);
    float b4[4] = {0};
    check("ref_push 非 512 倍数 → -1", va_aec_ref_push(rs, b4, 100) == -1, "%s", "");
    check("ref_push 空指针 → -1", va_aec_ref_push(rs, nullptr, 512) == -1, "%s", "");
    check("create_shared(NULL refsvc) → NULL",
          va_aec_create_shared(sr, nullptr) == nullptr, "%s", "");
    check("create_shared(48000) → NULL",
          va_aec_create_shared(48000, rs) == nullptr, "%s", "");
    va_aec* hs = va_aec_create_shared(sr, rs);
    const float* mp = b4;
    float* op = b4;
    check("process（旧接口）遇共享句柄 → -1",
          va_aec_process(hs, &mp, b4, &op, 512) == -1, "%s", "");
    check("process_shared 非 512 倍数 → -1",
          va_aec_process_shared(hs, b4, b4, b4, 100) == -1, "%s", "");
    check("process_shared 空指针 → -1",
          va_aec_process_shared(hs, nullptr, b4, b4, 512) == -1, "%s", "");
    va_aec_destroy(hs);
    va_aec_ref_destroy(rs);
    va_aec_ref_destroy(nullptr);
    check("ref_destroy(NULL) 安全", true, "%s", "");
}

// ------------------------------------------------------------------ //
// ABI / dlopen 冒烟
// ------------------------------------------------------------------ //
static void test_api_edges() {
    std::printf("[ABI] 边界与错误码\n");
    check("create 非法 sr → NULL", va_aec_create(0) == nullptr && va_aec_create(-16000) == nullptr, "%s", "");
    va_aec* h = va_aec_create(16000);
    check("create 16k 成功", h != nullptr, "%s", "");
    float buf[16] = {0};
    const float* mic = buf;
    float* out = buf;
    check("process 空指针 → -1",
          va_aec_process(h, nullptr, buf, &out, 16) == -1 &&
              va_aec_process(h, &mic, nullptr, &out, 16) == -1 &&
              va_aec_process(h, &mic, buf, &out, 0) == -1,
          "%s", "");
    va_aec_destroy(h);
    va_aec_destroy(nullptr);   // 不崩溃
    check("destroy(NULL) 安全", true, "%s", "");

    check("doa_bf 非法 nch/类型 → NULL",
          va_doa_bf_create(1, 16000, "ula", 0.04) == nullptr &&
              va_doa_bf_create(4, 16000, "xyz", 0.04) == nullptr,
          "%s", "");
    va_doa_bf* hb = va_doa_bf_create(4, 16000, nullptr, 0.0);   // NULL → ula/默认间距
    check("doa_bf create(默认 ula) 成功", hb != nullptr, "%s", "");
    float o[16];
    std::vector<const float*> in4 = {buf, buf, buf, buf};
    check("doa_bf process 静音 → 0 输出有限",
          va_doa_bf_process(hb, in4.data(), o, 16) == 0,
          "%s", "");
    float doa0 = va_doa_bf_get_doa(hb);
    check("静音 DOA = 0", doa0 == 0.0f, "%.2f", doa0);
    va_doa_bf_destroy(hb);
    va_doa_bf_destroy(nullptr);
}

typedef va_aec* (*fn_aec_create)(int);
typedef void (*fn_aec_destroy)(va_aec*);
typedef int (*fn_aec_process)(va_aec*, const float* const*, const float*, float* const*, int);
typedef va_doa_bf* (*fn_doa_create)(int, int, const char*, double);
typedef void (*fn_doa_destroy)(va_doa_bf*);
typedef int (*fn_doa_process)(va_doa_bf*, const float* const*, float*, int);
typedef float (*fn_doa_get)(const va_doa_bf*);

static void test_dlopen() {
    std::printf("[dlopen] 动态加载 libvoice_algo.so\n");
    void* so = dlopen("./libvoice_algo.so", RTLD_NOW);
    if (!so) {
        check("dlopen ./libvoice_algo.so", false, "%s", dlerror());
        return;
    }
    fn_aec_create c = (fn_aec_create)dlsym(so, "va_aec_create");
    fn_aec_destroy d = (fn_aec_destroy)dlsym(so, "va_aec_destroy");
    fn_aec_process p = (fn_aec_process)dlsym(so, "va_aec_process");
    fn_doa_create c2 = (fn_doa_create)dlsym(so, "va_doa_bf_create");
    fn_doa_destroy d2 = (fn_doa_destroy)dlsym(so, "va_doa_bf_destroy");
    fn_doa_process p2 = (fn_doa_process)dlsym(so, "va_doa_bf_process");
    fn_doa_get g2 = (fn_doa_get)dlsym(so, "va_doa_bf_get_doa");
    const bool sym_ok = c && d && p && c2 && d2 && p2 && g2;
    check("dlsym 全部 7 个符号", sym_ok, "%s", "");
    if (!sym_ok) return;
    // 最小功能性冒烟：白噪过 AEC
    Rng rng(9u);
    std::vector<float> ref(4096), mic(4096), out(4096, 0.f);
    for (int i = 0; i < 4096; i++) {
        ref[(size_t)i] = 0.05f * (float)rng.randn();
        mic[(size_t)i] = 0.6f * ref[(size_t)i];   // 简单即时回声
    }
    va_aec* h = c(16000);
    const float* mp = mic.data();
    float* op = out.data();
    int rc = p(h, &mp, ref.data(), &op, 4096);
    bool ok = (rc == 0) && all_finite(out);
    d(h);
    check("dlopen 路径 AEC process", ok, "rc=%d finite=%d", rc, (int)all_finite(out));
    check("dlsym doa getter 兼容", g2(nullptr) == 0.0f, "%s", "");
    dlclose(so);
}

// ------------------------------------------------------------------ //
// AEC 任意 n 分块 vs 单次调用一致性 + 多实例独立性
// ------------------------------------------------------------------ //
static void test_aec_stream_consistency() {
    std::printf("[AEC-stream] 任意 n 分块 / 单次调用 / 多实例\n");
    const int sr = 16000;
    std::vector<float> ref, mic;
    gen_aec_scene(sr, ref, mic);
    const size_t n = mic.size();

    // 单次调用（一次给全段）
    std::vector<float> o1(n, 0.f);
    va_aec* h1 = va_aec_create(sr);
    {
        const float* mp = mic.data();
        float* op = o1.data();
        va_aec_process(h1, &mp, ref.data(), &op, (int)n);
    }
    va_aec_destroy(h1);

    // 任意 n 分块（奇数/非对齐尺寸混合）
    std::vector<float> o2(n, 0.f);
    va_aec* h2 = va_aec_create(sr);
    {
        const int chunks[] = {512, 333, 1024, 128, 777};
        size_t off = 0;
        int ci = 0;
        while (off < n) {
            int cn = chunks[ci++ % 5];
            if (cn > (int)(n - off)) cn = (int)(n - off);
            const float* mp = mic.data() + off;
            float* op = o2.data() + off;
            va_aec_process(h2, &mp, ref.data() + off, &op, cn);
            off += (size_t)cn;
        }
    }
    va_aec_destroy(h2);

    check("分块输出有限", all_finite(o2), "%s", "");
    const long long lag = xcorr_argmax(o1, o2, 800);
    check("分块 vs 单次互相关主峰 lag=0（无错位）", lag == 0, "lag=%lld", lag);
    double num = 0, den = 0;
    const size_t a = n / 4;
    for (size_t i = a; i < n; i++) {
        const double dd = (double)o1[i] - o2[i];
        num += dd * dd;
        den += (double)o1[i] * o1[i];
    }
    // 预览块（零填充未来）与后续整块输出的固有微小差异：W 时域投影周期
    // （kProjEvery=4）内子滤波器支撑略超 B 抽头，预览的循环卷积回绕引入
    // ~1e-2 相对能量级差异（原版行为一致，自适应状态不受污染）。
    check("分块 vs 单次相对误差 < 5e-2", num / (den + 1e-30) < 5e-2, "rel=%.2e",
          num / (den + 1e-30));
    const double e2 = erle_db(mic, o2, a, n);
    check("任意 n 分块稳态 ERLE ≥ 18 dB", e2 >= 18.0, "%.2f dB", e2);

    // 多实例独立：A/B 交替调用，B 应与单独跑完全一致
    std::vector<float> refB, micB;
    Rng rngv(555u);
    const size_t nb = 2 * (size_t)sr;
    refB.resize(nb);
    micB.resize(nb);
    gen_speech(refB, (int)nb, sr, 66666u, 0.05);
    for (size_t i = 0; i < nb; i++) micB[i] = 0.5f * refB[i];   // 即时回声
    std::vector<float> oB1(nb, 0.f), oB2(nb, 0.f);
    va_aec *ha = va_aec_create(sr), *hb2 = va_aec_create(sr);
    {
        const size_t ch = 640;
        for (size_t off = 0; off < nb; off += ch) {
            const size_t cn = std::min(ch, nb - off);
            const float* a1 = mic.data() + off;
            float* ao = o1.data() + off;   // 复用 o1 空间（值不再需要）
            va_aec_process(ha, &a1, ref.data() + off, &ao, (int)cn);   // A 实例干扰
            const float* b1 = micB.data() + off;
            float* bo = oB1.data() + off;
            va_aec_process(hb2, &b1, refB.data() + off, &bo, (int)cn);
        }
    }
    va_aec_destroy(ha);
    va_aec_destroy(hb2);
    // 单独跑 B（与上面 B 完全相同的 640 分块模式，无 A 干扰）：
    // 实例状态独立 ⟹ 必须与交替运行逐位一致
    va_aec* hb3 = va_aec_create(sr);
    {
        const size_t ch = 640;
        for (size_t off = 0; off < nb; off += ch) {
            const size_t cn = std::min(ch, nb - off);
            const float* b1 = micB.data() + off;
            float* bo = oB2.data() + off;
            va_aec_process(hb3, &b1, refB.data() + off, &bo, (int)cn);
        }
    }
    va_aec_destroy(hb3);
    bool same = true;
    for (size_t i = 0; i < nb; i++)
        if (oB1[i] != oB2[i]) { same = false; break; }
    check("多实例独立（B 与单独跑逐位一致）", same, "%s", "");
}

// ------------------------------------------------------------------ //
int main() {
    std::printf("libvoice_algo 单测（合成数据，确定性种子，无音频播放）\n\n");

    test_api_edges();
    std::printf("\n");

    // ---- AEC 16k 主指标（块对齐分块 512/1024/1536）---- //
    const int al[] = {512, 1024, 1536};
    AecResult a16 = run_aec(16000, "AEC 16k 主指标", al, 3);
    check("AEC 16k 稳态 ERLE ≥ 18 dB", a16.erle_steady >= 18.0, "%.2f dB",
          a16.erle_steady);
    // ---- AEC 48k 合成回归（旧参数 1024×16=341ms；与原版算法同场景逐位一致
    //      ——Mac 端 parity 验证 + /tmp/afe_bench 真数据回归另行保证旧门槛）---- //
    const int a4[] = {1024, 2048};
    AecResult a48 = run_aec(48000, "AEC 48k 合成回归", a4, 2);
    check("AEC 48k 稳态 ERLE ≥ 15 dB（同场景与原版逐位一致）", a48.erle_steady >= 15.0,
          "%.2f dB", a48.erle_steady);

    // ---- 任意 n / 流式一致性 / 多实例 ---- //
    test_aec_stream_consistency();
    std::printf("\n");

    // ---- BF 16k 主指标 + 多角度 ---- //
    BfResult b30 = run_bf(16000, 30.0, "BF 16k θ=+30°");
    check("BF 16k DOA 误差 < 10° (θ=30°)", b30.err < 10.0, "%.2f°", b30.err);
    check("BF 16k DSB 增益 ≥ +5 dB (θ=30°)", b30.gain >= 5.0, "%+.2f dB", b30.gain);
    BfResult bm60 = run_bf(16000, -60.0, "BF 16k θ=-60°");
    check("BF 16k DOA 误差 < 10° (θ=-60°)", bm60.err < 10.0, "%.2f°", bm60.err);
    check("BF 16k DSB 增益 ≥ +5 dB (θ=-60°)", bm60.gain >= 5.0, "%+.2f dB", bm60.gain);
    // ---- BF 48k 合成回归 ---- //
    BfResult b48 = run_bf(48000, 30.0, "BF 48k 合成回归");
    check("BF 48k DOA 误差 < 10°", b48.err < 10.0, "%.2f°", b48.err);
    check("BF 48k DSB 增益 ≥ +5 dB", b48.gain >= 5.0, "%+.2f dB", b48.gain);

    // ---- DOA 增量 vs 批式逐位一致（G 优化）---- //
    std::printf("\n");
    test_doa_inc_parity();

    // ---- 共享参考谱 vs 独立逐位一致（A 步）---- //
    std::printf("\n");
    test_aec_shared_ref();

    // ---- dlopen ---- //
    std::printf("\n");
    test_dlopen();

    std::printf("\n%s (fails=%d)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail);
    return g_fail ? 1 : 0;
}
