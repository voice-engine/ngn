// voice/algo/aec_kf.cpp — PBFDKF 实现（与 afecpp 同步，含 FIXES_AEC 修复）
//
// 详见 aec_kf.hpp 头注释与 AEC_KF_REPORT.md。可调参数集中在下方匿名命名
// 空间的 constexpr，量纲与作用在报告"参数"一节给出推导。

#include "aec_kf.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

// ---- 可调参数（块率 = 48000/1024 ≈ 46.9 块/s，1 块 ≈ 21.3ms）----
// PSD 统计：单块谱 |E|²/Σ|X|² 是 χ²₂（方差=均值），逐 bin 直接使用会数值爆
// 炸，故 pE 与 Σ|X|² 用同一 5-tap 频域核平滑（比值无偏）+ 时间 EMA。
#ifndef KF_KGAMMAINIT
#define KF_KGAMMAINIT 2.0f
#endif
constexpr float kGammaInit = KF_KGAMMAINIT;    // P0 = γ·R/Σ|X|²：初始等效 NLMS 步长 γ/(1+γ)≈0.67
#ifndef KF_KWARMUPBLOCKS
#define KF_KWARMUPBLOCKS 4
#endif
constexpr int   kWarmupBlocks = KF_KWARMUPBLOCKS;     // P 初始化预热块数（统计充分后再放行更新）
#ifndef KF_KCQ
#define KF_KCQ 2e-3f
#endif
constexpr float kCq = KF_KCQ;   // Q = c_q·R/Σ|X|²：稳态等效步长（跟踪/失配折中）。
                                 // 5e-4→2e-3（FIXES_AEC 缺陷3）：ppm 级参考时变
                                 // （分数重采样/时钟失配）下随机游走模型需更快跟踪；
                                 // S1/S2/S3 基准与双讲保护实测不受损（见报告）。
#ifndef KF_KRUP
#define KF_KRUP 0.60f
#endif
constexpr float kRUp = KF_KRUP;
#ifndef KF_KRDN
#define KF_KRDN 0.25f
#endif
constexpr float kRDn = KF_KRDN;   // R 下降平滑系数（~4 块；需跟随收敛残差下降）
constexpr float kStotSmooth = 0.70f;   // Σ|X_k|² 平滑（Q 分母去抖）
constexpr float kStotRegRel = 0.05f;   // Q 分母正则（相对均值，死 bin 防爆）
constexpr float kStotInitFloor = 1e-3f; // P 初始化分母下限（相对均值）
#ifndef KF_KPFLOORREL
#define KF_KPFLOORREL 1e-3f
#endif
constexpr float kPFloorRel = KF_KPFLOORREL;   // P̄ 下限 = rel·R/Σ|X|²（防塌缩，保证可再跟踪）
constexpr float kPCapRel    = 100.0f;  // P̄ 上限 = rel·R/Σ|X|²（防发散）
#ifndef KF_KCONSISTTHR
#define KF_KCONSISTTHR 1e30f
#endif
constexpr float kConsistThr = KF_KCONSISTTHR;    // 新息一致性膨胀阈值（平滑残差/预测方差 S）
#ifndef KF_KCONSISTMAX
#define KF_KCONSISTMAX 4.0f
#endif
constexpr float kConsistMax = KF_KCONSISTMAX;    // 单块协方差膨胀上限
#ifndef KF_KMUMAX
#define KF_KMUMAX 0.5f
#endif
constexpr float kMuMax = KF_KMUMAX;
#ifndef KF_KOUTLIERC
#define KF_KOUTLIERC 1.5f
#endif
constexpr float kOutlierC = KF_KOUTLIERC;    // 新息离群抑制系数（S ≥ c·|E|²，未平滑）
#ifndef KF_KGALPHA
#define KF_KGALPHA 0.10f
#endif
constexpr float kGAlpha = KF_KGALPHA;   // 分块份额 g 的 EMA 速率（~10 块学成）
#ifndef KF_KGUNIFORM
#define KF_KGUNIFORM 0.30f
#endif
constexpr float kGUniform = KF_KGUNIFORM;   // g 混入均匀份额比例（保证再跟踪/防过拟合）
#ifndef KF_KFGEMA
#define KF_KFGEMA 0.95f
#endif
constexpr float kFgEma = KF_KFGEMA;   // 双路径残差能量 EMA（~20 块窗口）
#ifndef KF_KFGCOPYRATIO
#define KF_KFGCOPYRATIO 0.95f
#endif
constexpr float kFgCopyRatio = KF_KFGCOPYRATIO;  // 背景 E_bg < ratio·E_fg 时复制到前景
#ifndef KF_KFGCOPYHOLD
#define KF_KFGCOPYHOLD 2
#endif
constexpr int   kFgCopyHold = KF_KFGCOPYHOLD;       // 复制后冷却块数（防抖动）
#ifndef KF_FG_BLEND
#define KF_FG_BLEND 0.05f
#endif
constexpr float kFgBlend = KF_FG_BLEND;   // 前景慢 EMA 系数（τ≈50 块，滤除背景失配噪声）
#ifndef KF_KGRHO
#define KF_KGRHO 0.65f
#endif
constexpr float kGRho = KF_KGRHO;         // 分块先验衰减率（w̄_k ∝ ρ^k，房间 IR 物理形态）
#ifndef KF_KGLATECAP
#define KF_KGLATECAP 0.005f
#endif
constexpr float kGLateCap = KF_KGLATECAP; // 晚期分块 g 硬下限占比（近端同源内容防护）
constexpr double kDivergeGain = 100.0; // 残差功率 > gain·(mic+ref) 功率 → 复位
#ifndef KF_KPROJEVERY
#define KF_KPROJEVERY 4
#endif
constexpr int   kProjEvery = KF_KPROJEVERY;  // W 时域约束周期（块）

// ---- FIXES_AEC：残差-参考相干统计 / 欠消检测 / 后置维纳滤波 / 软先验 ----
// 检测统计为复数互谱 EMA（α=0.6，τ≈2.5 块）：|EMA(E·X0^*)|² 无偏（独立信号
// 平均→0；幅度 EMA 有 ~1/N_eff 正偏置，不可用——实测会把所有场景抬到 0.6+）。
#ifndef KF_POSTEMA
#define KF_POSTEMA 0.60f
#endif
constexpr float kPostEma = KF_POSTEMA;
// 后置滤波模式门（全局相干分数 cfrac 双门限滞回）。实测（α=0.6）：
//   双讲（独立近端 1:1）0.17-0.39 / S2 近端同源 0.00-0.22 → 门关（直通，零损伤）
//   谐波滞后未消 0.29-0.78 / 单讲收敛后视收敛度 0.05-0.82 → 门开（维纳后置）
#ifndef KF_PFONCFRAC
#define KF_PFONCFRAC 0.45f
#endif
constexpr float kPfOnCfrac = KF_PFONCFRAC;
#ifndef KF_PFOFFCFRAC
#define KF_PFOFFCFRAC 0.35f
#endif
constexpr float kPfOffCfrac = KF_PFOFFCFRAC;
#ifndef KF_PFPERSIST
#define KF_PFPERSIST 6
#endif
constexpr int kPfPersist = KF_PFPERSIST;         // 后置滤波开去抖块数
#ifndef KF_POSTC0
#define KF_POSTC0 0.50f
#endif
constexpr float kPostC0 = KF_POSTC0;         // 相干占比硬膝（膝下直通）
#ifndef KF_POSTKAPPA
#define KF_POSTKAPPA 4.0f
#endif
constexpr float kPostKappa = KF_POSTKAPPA;   // 过减因子
#ifndef KF_POSTGMIN
#define KF_POSTGMIN 0.10f
#endif
constexpr float kPostGmin = KF_POSTGMIN;     // 增益下限（-20 dB）
#ifndef KF_POSTATK
#define KF_POSTATK 0.30f
#endif
constexpr float kPostAtk = KF_POSTATK;       // 增益下降速率（快降 ~2 块）
#ifndef KF_POSTREL
#define KF_POSTREL 0.85f
#endif
constexpr float kPostRel = KF_POSTREL;       // 增益回升速率（慢升 ~7 块）
#ifndef KF_POSTWARM
#define KF_POSTWARM 4
#endif
constexpr int   kPostWarm = KF_POSTWARM;     // 预热块数（统计积累前直通）
// 欠消检测（跟踪模式：Q 自适应）。触发 = cfrac 高（残差以相干回声为主）
// 且残差/参考功率比超阈（欠消 > -11dB）持续 kDetPersist 块。
#ifndef KF_DETCOHHI
#define KF_DETCOHHI 0.60f
#endif
constexpr float kDetCohHi = KF_DETCOHHI;
#ifndef KF_DETCOHLO
#define KF_DETCOHLO 0.40f
#endif
constexpr float kDetCohLo = KF_DETCOHLO;
#ifndef KF_DETRELTHR
#define KF_DETRELTHR 0.08f
#endif
constexpr float kDetRelThr = KF_DETRELTHR;
#ifndef KF_DETPERSIST
#define KF_DETPERSIST 3
#endif
constexpr int   kDetPersist = KF_DETPERSIST;
#ifndef KF_TRACKQBOOST
#define KF_TRACKQBOOST 15.0f
#endif
constexpr float kTrackQBoost = KF_TRACKQBOOST;  // 跟踪模式 Q 放大倍数
// 证据自适应晚期块上限：块 k 的 W 能量份额 EMA 超过先验 w̄_k 的比率直接放大
// 其 g 上限（至多 kCapRelaxMax 倍，上限 1.0）。
#ifndef KF_CAPRELAXMAX
#define KF_CAPRELAXMAX 4.0f
#endif
constexpr float kCapRelaxMax = KF_CAPRELAXMAX;

inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// 5-tap 频域平滑（核 [1,4,6,4,1]/16，环形边界）——实谱对称，环形无误
inline void smooth5(const float* x, float* y, int M) {
    for (int m = 0; m < M; m++) {
        const float s = 6.0f * x[m]
                      + 4.0f * (x[(m + 1) % M] + x[(m + M - 1) % M])
                      + (x[(m + 2) % M] + x[(m + M - 2) % M]);
        y[m] = 0.0625f * s;
    }
}

// 5-tap 平滑（Hermitian 半谱版，bins [0, H]）：边界按镜像反射（reflect(i) =
// |i| 与 2H-i）——全谱环形边界在 DC/Nyquist 处的邻居恰是镜像 bin，取值相同，
// 故与全谱版逐位等价。in-place 安全（升序写、只读未写侧）。
inline void smooth5h(const float* x, float* y, int H) {
    for (int m = 0; m <= H; m++) {
        const int im1 = m >= 1 ? m - 1 : 1;
        const int im2 = m >= 2 ? m - 2 : 2 - m;
        const int ip1 = m <= H - 1 ? m + 1 : H - 1;      // reflect(H+1)=H-1
        const int ip2 = m <= H - 2 ? m + 2 : 2 * H - m - 2;  // reflect(H+2)=H-2
        const float s = 6.0f * x[m]
                      + 4.0f * (x[ip1] + x[im1])
                      + (x[ip2] + x[im2]);
        y[m] = 0.0625f * s;
    }
}

}  // namespace

// ------------------------------------------------------------------ //
// AecFFTPlan：迭代 radix-2。每级独立连续 twiddle 表（stage_tw[s][j] =
// exp(-2πi j/2^(s+1))，内层循环顺序访问可向量化）；inverse 为编译期模板
// 参数消除内层分支；逆变换含 1/n 缩放。
// ------------------------------------------------------------------ //
void AecFFTPlan::init(int n) {
    this->n = n;
    int lg = 0;
    while ((1 << lg) < n) lg++;
    rev.assign(n, 0);
    for (int i = 1; i < n; i++) rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (lg - 1));
    stage_half.clear();
    int total = 0;
    for (int len = 2; len <= n; len <<= 1) { stage_half.push_back(len >> 1); total += len >> 1; }
    twr.resize(total); twi.resize(total);
    twr_inv.resize(total); twi_inv.resize(total);
    int off = 0;
    for (int len = 2; len <= n; len <<= 1) {
        const int half = len >> 1;
        for (int j = 0; j < half; j++) {
            const double ang = -2.0 * M_PI * j / len;
            twr[off + j] = (float)std::cos(ang);
            twi[off + j] = (float)std::sin(ang);
            twr_inv[off + j] = twr[off + j];
            twi_inv[off + j] = -twi[off + j];
        }
        off += half;
    }
}

template <bool kInverse>
inline void fft_core(const int n, const std::vector<int>& rev,
                     const float* twr, const float* twi,   // 每级表已按方向共轭好
                     const int* stage_half, std::complex<float>* a) {
    for (int i = 0; i < n; i++)
        if (i < rev[i]) std::swap(a[i], a[rev[i]]);
    float* x = reinterpret_cast<float*>(a);
    int off = 0;   // 每级 twiddle 表的起始偏移（表首含 len=2 的全 1，可跳过乘法）
    int s = 0;
    for (int len = 2; len <= n; len <<= 1, s++) {
        const int half = stage_half[s];
        if (len == 2) {                     // w=1：纯加减（仍推进表偏移）
            for (int i = 0; i < n; i += 2) {
                const float ar = x[2 * i], ai = x[2 * i + 1];
                const float br = x[2 * i + 2], bi = x[2 * i + 3];
                x[2 * i] = ar + br; x[2 * i + 1] = ai + bi;
                x[2 * i + 2] = ar - br; x[2 * i + 3] = ai - bi;
            }
            off += half;
            continue;
        }
        const float* wr = twr + off;
        const float* wi = twi + off;
        for (int i = 0; i < n; i += len) {
            float* p = x + 2 * i;
            float* q = p + 2 * half;
            for (int j = 0; j < half; j++) {
                const float cr = wr[j], ci = wi[j];
                const float ar = p[2 * j], ai = p[2 * j + 1];
                const float br = q[2 * j], bi = q[2 * j + 1];
                const float vr = br * cr - bi * ci;
                const float vi = br * ci + bi * cr;
                p[2 * j] = ar + vr; p[2 * j + 1] = ai + vi;
                q[2 * j] = ar - vr; q[2 * j + 1] = ai - vi;
            }
        }
        off += half;
    }
}

void AecFFTPlan::run(std::complex<float>* a, bool inverse) const {
    if (inverse) {
        fft_core<true>(n, rev, twr_inv.data(), twi_inv.data(), stage_half.data(), a);
        const float s = 1.0f / n;
        for (int i = 0; i < n; i++) a[i] *= s;
    } else {
        fft_core<false>(n, rev, twr.data(), twi.data(), stage_half.data(), a);
    }
}

// ------------------------------------------------------------------ //
// RefEngine：参考谱服务。push = 原 process_block_ 段1 的 herm 路径原样
// 搬移（滑窗→FFT→半谱入环），另存每槽 |X|² 表（nx 表达式与独立模式
// pass1 内联计算逐字相同）。
// ------------------------------------------------------------------ //
void RefEngine::init(int B, int K) {
    B_ = B < 64 ? 64 : B;
    if (B_ & (B_ - 1)) {
        int p = 1;
        while (p < B_) p <<= 1;
        B_ = p;
    }
    K_ = K < 1 ? 1 : K;
    M_ = 2 * B_;
    nb_ = M_ / 2 + 1;
    fft_.init(M_);
    x_time_.assign(M_, 0.0f);
    Xring_.assign((size_t)K_ * nb_, std::complex<float>());
    nxring_.assign((size_t)K_ * nb_, 0.0f);
    fftbuf_.assign(M_, std::complex<float>());
    xpos_ = 0;
}

void RefEngine::push(const float* ref) {
    const int B = B_, M = M_, nb = nb_;
    std::copy(x_time_.begin() + B, x_time_.end(), x_time_.begin());
    for (int i = 0; i < B; i++) x_time_[B + i] = ref[i];
    for (int i = 0; i < M; i++) fftbuf_[i] = std::complex<float>(x_time_[i], 0.0f);
    fft_.run(fftbuf_.data(), false);
    std::complex<float>* xs = &Xring_[(size_t)xpos_ * nb];
    float* nxs = &nxring_[(size_t)xpos_ * nb];
    for (int m = 0; m < nb; m++) {
        const float xr = fftbuf_[m].real(), xi = fftbuf_[m].imag();
        xs[m] = fftbuf_[m];
        nxs[m] = xr * xr + xi * xi;
    }
    xpos_ = (xpos_ + 1) % K_;
}

void RefEngine::stot(float* out) const {
    // push 后 xpos_ = 下一写入位；最新槽 = (xpos_-1)，k 块前 = (xpos_-1-k)
    std::fill(out, out + nb_, 0.0f);
    for (int k = 0; k < K_; k++) {
        const float* nxs = &nxring_[(size_t)((xpos_ - 1 - k + 2 * K_) % K_) * nb_];
        for (int m = 0; m < nb_; m++) out[m] += nxs[m];
    }
}

// ------------------------------------------------------------------ //
// 构造
// ------------------------------------------------------------------ //
PBFDKF::PBFDKF(int frame, int blocks, int sr, bool half)
    : sr_(sr), herm_(half) {
    B_ = frame < 64 ? 64 : frame;
    if (B_ & (B_ - 1)) {            // radix-2 要求 2 的幂
        int p = 1;
        while (p < B_) p <<= 1;
        B_ = p;
    }
    K_ = blocks < 1 ? 1 : blocks;
    M_ = 2 * B_;
    nb_ = herm_ ? M_ / 2 + 1 : M_;   // Hermitian 半谱：bins [0, M/2]

    fft_.init(M_);
    if (herm_) {
        own_ref_.init(B_, K_);   // herm：x_time/X 环/nx 表由 RefEngine 承载
    } else {
        x_time_.assign(M_, 0.0f);
        X_.assign(K_, std::vector<std::complex<float>>(nb_));
    }
    W_.assign(K_, std::vector<std::complex<float>>(nb_));
    Wfg_.assign(K_, std::vector<std::complex<float>>(nb_));
    gprofile_.resize(K_);
    gcapk_.resize(K_);
    {
        double ssum = 0.0;
        for (int k = 0; k < K_; k++) ssum += std::pow((double)kGRho, k);
        for (int k = 0; k < K_; k++) {
            gprofile_[k] = (float)(std::pow((double)kGRho, k) / ssum);   // w̄_k
            gcapk_[k] = kGLateCap + (1.0f - kGLateCap) * gprofile_[k] * K_;  // 相对均匀的提升上限
        }
        // 早期分块不设实际上限（真实路径可能集中在前几块）
        for (int k = 0; k < 4 && k < K_; k++) gcapk_[k] = 1.0f;
    }
    wshare_ema_.assign(K_, 0.0);
    wsum_f_.assign(K_, 0.0f);
    gcap_use_ = gcapk_;
    G_.assign(K_, std::vector<float>(nb_));
    for (int k = 0; k < K_; k++) std::fill(G_[k].begin(), G_[k].end(), gprofile_[k]);
    Pbar_.assign(nb_, 0.0f);
    R_.assign(nb_, 0.0f);
    Stot_sm_.assign(nb_, 0.0f);
    inited_.assign(nb_, 0);

    fftbuf_.resize(M_);
    Y_.resize(nb_);
    Yfg_.resize(nb_);
    E_.resize(nb_);
    e_blk_.resize(B_);
    efg_blk_.resize(B_);
    stot_.resize(nb_);
    pef_.resize(nb_);
    stf_.resize(nb_);
    qv_.resize(nb_);
    Sm_.resize(nb_);
    invSm_.resize(nb_);
    pfl_.resize(nb_);
    pcp_.resize(nb_);
    gex_.resize(nb_);
    g2ex_.resize(nb_);
    invg_.resize(nb_);
    gnew_.resize(nb_);
    tmic_.assign(B_, 0.0f);
    tref_.assign(B_, 0.0f);
    tout_.resize(B_);
}

// ------------------------------------------------------------------ //
// 单块处理（核心）：overlap-save + 逐 bin 比例化卡尔曼
// herm_ 模式：所有 per-bin 循环/存储只覆盖 [0, M/2]（实谱 Hermitian 对称，
// W 更新项保持对称性，投影定期强制）；FFT 出入口做全谱 ↔ 半谱转换。
// ------------------------------------------------------------------ //
double PBFDKF::sum_bins_(const float* x) const {
    // 与全 M bin 顺序和语义一致的标量和（herm：镜像对计双）
    if (!herm_) {
        double s = 0.0;
        for (int m = 0; m < M_; m++) s += x[m];
        return s;
    }
    const int H = M_ / 2;
    double s = x[0] + x[H];
    for (int m = 1; m < H; m++) s += 2.0 * x[m];
    return s;
}

void PBFDKF::herm_expand_(const std::vector<std::complex<float>>& half) {
    for (int m = nb_; m < M_; m++) fftbuf_[m] = std::conj(half[M_ - m]);
}

void PBFDKF::process_block_(const float* mic, const float* ref, float* out) {
    const int B = B_, M = M_, K = K_, nb = nb_;
    stats_.blocks++;

    // -- 0) 参考静音门控：ref 块严格全零（永续播放架构无写者期）→ 走轻量
    //    等价路径。X=0 ⟹ 回声谱 Y/Yfg=0（pass1 MAC 结果全零）、卡尔曼更新
    //    项=0（pass2 恒等）、W/份额不变（g 学习/投影恒等）——跳过即逐位等价；
    //    统计（R 跟近端、P̄ 按 Q 增长）与发散保护必须照常（数学行为不变）。
    //    待机期 AEC 计算量降至 ~1/3（实机占空比收益）。
    double pr_gate = 0.0;
    for (int i = 0; i < B; i++) pr_gate += (double)ref[i] * ref[i];
    // hold 计数：连续静音 ≥20 块（0.64s）才进入门控——语音音节间隙（0.9s 周期
    // 包络过零，典型 6-14 块）不触发，语音期与无门控逐位一致（间隙跳更新会改
    // 收敛轨迹，实测 ERLE -1.7dB）；真待机（永续架构无写者）秒级静音正常进入
    if (pr_gate <= 0.0) { if (sil_hold_ < 1000) sil_hold_++; }
    else sil_hold_ = 0;
    const bool ref_silent = sil_hold_ >= 20;

    // -- 1) 参考谱：herm 走 RefEngine（独立=内置 push；共享=外部已 push）-- //
    //    legacy 路径保持原样（x_time 滑窗 → FFT → X_ 环，48k 逐位基线）
    const RefEngine* RE = nullptr;
    if (herm_) {
        if (!shared_) own_ref_.push(ref);
        RE = shared_ ? shared_ : &own_ref_;
    } else {
        std::copy(x_time_.begin() + B, x_time_.end(), x_time_.begin());
        for (int i = 0; i < B; i++) x_time_[B + i] = ref[i];
        for (int i = 0; i < M; i++) fftbuf_[i] = std::complex<float>(x_time_[i], 0.0f);
        fft_.run(fftbuf_.data(), false);
        std::copy(fftbuf_.begin(), fftbuf_.begin() + nb, X_[xpos_].begin());
        xpos_ = (xpos_ + 1) % K;
    }

    // -- 2) 回声估计（背景 Y 与前景 Yfg 同环遍历），Σ|X_k|² 与加权谱 ---- //
    //    gex=Σg_k|X_k|²、g2ex=Σg_k²|X_k|²（pass1 要用）一并累积，单次遍历。
    //    herm：|X_k|² 查 RefEngine 预计算表（与独立模式内联 nx 同值同序，
    //    共享/独立两路逐位一致）；legacy：内联计算（逐位基线，不动）。
    if (ref_silent) {
        // 静音门控：X_k 全零 ⟹ 上述累积量全零——fill 即为精确结果，跳过 MAC
        std::fill(Y_.begin(), Y_.end(), std::complex<float>());
        std::fill(Yfg_.begin(), Yfg_.end(), std::complex<float>());
        std::fill(stot_.begin(), stot_.end(), 0.0f);
        std::fill(gex_.begin(), gex_.end(), 0.0f);
        std::fill(g2ex_.begin(), g2ex_.end(), 0.0f);
        goto pass1_done;
    }
    std::fill(Y_.begin(), Y_.end(), std::complex<float>());
    std::fill(Yfg_.begin(), Yfg_.end(), std::complex<float>());
    std::fill(stot_.begin(), stot_.end(), 0.0f);
    std::fill(gex_.begin(), gex_.end(), 0.0f);
    std::fill(g2ex_.begin(), g2ex_.end(), 0.0f);
    if (herm_) {
        RE->stot(stot_.data());
        const int sp1 = (RE->xpos() - 1 + K) % K;   // 最新槽
        for (int k = 0; k < K; k++) {
            const int slot = (sp1 - k + K) % K;
            const std::complex<float>* __restrict xk = RE->X(slot);
            const float* __restrict nxs = RE->nx(slot);
            const std::complex<float>* __restrict wk = W_[k].data();
            const std::complex<float>* __restrict wfk = Wfg_[k].data();
            const float* __restrict gk = G_[k].data();
            std::complex<float>* __restrict y = Y_.data();
            std::complex<float>* __restrict yf = Yfg_.data();
            float* __restrict gx = gex_.data();
            float* __restrict g2x = g2ex_.data();
            for (int m = 0; m < nb; m++) {
                const float xr = xk[m].real(), xi = xk[m].imag();
                const float ar = wk[m].real(), ai = wk[m].imag();
                const float br = wfk[m].real(), bi = wfk[m].imag();
                y[m]  = std::complex<float>(y[m].real()  + (xr * ar - xi * ai),
                                            y[m].imag()  + (xr * ai + xi * ar));
                yf[m] = std::complex<float>(yf[m].real() + (xr * br - xi * bi),
                                            yf[m].imag() + (xr * bi + xi * br));
                const float nx = nxs[m];
                const float g = gk[m];
                gx[m] += g * nx;
                g2x[m] += g * g * nx;
            }
        }
    } else {
        for (int k = 0; k < K; k++) {
            const std::complex<float>* __restrict xk = X_[(xpos_ - 1 - k + 2 * K) % K].data();
            const std::complex<float>* __restrict wk = W_[k].data();
            const std::complex<float>* __restrict wfk = Wfg_[k].data();
            const float* __restrict gk = G_[k].data();
            std::complex<float>* __restrict y = Y_.data();
            std::complex<float>* __restrict yf = Yfg_.data();
            float* __restrict st = stot_.data();
            float* __restrict gx = gex_.data();
            float* __restrict g2x = g2ex_.data();
            for (int m = 0; m < nb; m++) {
                const float xr = xk[m].real(), xi = xk[m].imag();
                const float ar = wk[m].real(), ai = wk[m].imag();
                const float br = wfk[m].real(), bi = wfk[m].imag();
                y[m]  = std::complex<float>(y[m].real()  + (xr * ar - xi * ai),
                                            y[m].imag()  + (xr * ai + xi * ar));
                yf[m] = std::complex<float>(yf[m].real() + (xr * br - xi * bi),
                                            yf[m].imag() + (xr * bi + xi * br));
                const float nx = xr * xr + xi * xi;
                st[m] += nx;
                const float g = gk[m];
                gx[m] += g * nx;
                g2x[m] += g * g * nx;
            }
        }
    }
    pass1_done:;
    // 两路回声谱打包为一路复信号（实=Y 虚=Yfg），单次逆变换同时得两路实输出。
    // herm：Y/Yfg 只在 [0,H] 累加，全谱由镜像重建（Yre 偶/Yim 奇）：
    //   pack[m]   = (Yre - YfgIm) + i(Yim + YfgRe)
    //   pack[M-m] = (Yre + YfgIm) + i(-Yim + YfgRe)
    if (herm_) {
        const int H = M / 2;
        for (int m = 0; m <= H; m++) {
            const float yr = Y_[m].real(), yi = Y_[m].imag();
            const float fr = Yfg_[m].real(), fi = Yfg_[m].imag();
            fftbuf_[m] = std::complex<float>(yr - fi, yi + fr);
            if (m > 0)
                fftbuf_[M - m] = std::complex<float>(yr + fi, -yi + fr);
        }
        fft_.run(fftbuf_.data(), true);
    } else {
        for (int m = 0; m < M; m++)
            Y_[m] = std::complex<float>(Y_[m].real() - Yfg_[m].imag(),
                                        Y_[m].imag() + Yfg_[m].real());
        fft_.run(Y_.data(), true);
    }
    const std::complex<float>* const yspec = herm_ ? fftbuf_.data() : Y_.data();
    double peb = 0.0, pefg = 0.0, pd_ = 0.0, pr_ = 0.0;
    for (int i = 0; i < B; i++) {
        const float e = mic[i] - yspec[B + i].real();       // 背景（自适应用）
        const float ef = mic[i] - yspec[B + i].imag();      // 前景（输出用）
        e_blk_[i] = e;
        efg_blk_[i] = ef;
        out[i] = ef;
        peb += (double)e * e;
        pefg += (double)ef * ef;
        pd_ += (double)mic[i] * mic[i];
        pr_ += (double)ref[i] * ref[i];
    }

    // -- 2b) 双路径切换：背景 EMA 残差明显优于前景时整体复制 ------------ //
    //    近端语音在两路残差中同现，能量比较中抵消，比的是残余回声+失配；
    //    复制后冷却防抖动。输出始终来自前景 = 历史最优滤波器。
    {
        Ebg_ = kFgEma * Ebg_ + (1.0 - kFgEma) * peb;
        Efg_ = kFgEma * Efg_ + (1.0 - kFgEma) * pefg;
        if (copy_hold_ > 0) copy_hold_--;
        double pr = 0.0;
        for (int i = 0; i < B; i++) pr += (double)ref[i] * ref[i];
        // 慢 EMA 前景跟踪：双端时背景残差含近端分量，能量比较不敏感，
        // 慢平均让持续收敛的路径成分通过、零均值失配噪声被低通掉
        if (pr > 1e-12) {
            for (int k = 0; k < K; k++) {
                std::complex<float>* wfk = Wfg_[k].data();
                const std::complex<float>* wk = W_[k].data();
                for (int m = 0; m < nb; m++) wfk[m] += kFgBlend * (wk[m] - wfk[m]);
            }
        }
        if (copy_hold_ == 0 && pr > 1e-12 && Ebg_ < kFgCopyRatio * Efg_) {
            for (int k = 0; k < K; k++) Wfg_[k] = W_[k];
            Ebg_ = Efg_ = 0.5 * (Ebg_ + Efg_);
            copy_hold_ = kFgCopyHold;
            stats_.fg_copies++;
            for (int i = 0; i < B; i++) {              // 新前景 = 背景：输出即背景残差
                efg_blk_[i] = e_blk_[i];
                out[i] = e_blk_[i];
            }
        }
    }

    // -- 3) 残差谱 E = FFT([0_B | e]）（背景残差，供自适应）-------------- //
    std::fill(fftbuf_.begin(), fftbuf_.begin() + B, std::complex<float>());
    for (int i = 0; i < B; i++) fftbuf_[B + i] = std::complex<float>(e_blk_[i], 0.0f);
    fft_.run(fftbuf_.data(), false);
    std::copy(fftbuf_.begin(), fftbuf_.begin() + nb, E_.begin());

    // -- 4) 统计量更新 -------------------------------------------------- //
    // pE 与 stot 用同一 5-tap 核做频域平滑（χ²₂ 去抖，比值无偏）：
    //   R      = 残差 PSD 非对称 EMA（测量噪声 = 近端+底噪+失配）
    //   Stot_sm= Σ|X_k|² 的时间 EMA（Q 的分母、P̄ 的尺度基准）
    // 预热 kWarmupBlocks 块后，用平滑统计初始化 P0 = γ·R/Σ|X|²
    //（初始等效 NLMS 步长 γ/(1+γ)，分母带相对下限防死 bin 爆炸）
    for (int m = 0; m < nb; m++) {
        pef_[m] = std::norm(E_[m]);
        stf_[m] = stot_[m];
    }
    if (herm_) {
        smooth5h(pef_.data(), pef_.data(), M / 2);
        smooth5h(stf_.data(), stf_.data(), M / 2);
    } else {
        smooth5(pef_.data(), pef_.data(), M);
        smooth5(stf_.data(), stf_.data(), M);
    }
    double smean_d = sum_bins_(stf_.data()) / M;
    if (smean_d > 1e-12) act_blocks_++;
    const float smean = (float)smean_d;
    const float stot_thr = kStotInitFloor * smean + 1e-12f;
    for (int m = 0; m < nb; m++) {
        if (!inited_[m]) {
            R_[m] = pef_[m];                    // 预热期持续刷新，供 P 初始化
            if (act_blocks_ >= kWarmupBlocks && stf_[m] > stot_thr) {
                Pbar_[m] = K * kGammaInit * R_[m] / (stf_[m] > stot_thr ? stf_[m] : stot_thr);
                inited_[m] = 1;
            }
        } else {
            const float a = pef_[m] > R_[m] ? kRUp : kRDn;
            R_[m] += a * (pef_[m] - R_[m]);
        }
        Stot_sm_[m] = kStotSmooth * Stot_sm_[m] + (1.0f - kStotSmooth) * stf_[m];
    }

    // -- 4b) 残差-参考互谱统计（FIXES_AEC）：后置滤波 + 欠消检测 ------- //
    //    复数互谱 EMA（无偏）：coh[m]=|EMA(E·X0^*)|²/EMA|X0|² 为残差中与
    //    参考相干的功率（回声成分）；c[m]=coh/Cee 为每 bin 相干占比，
    //    cfrac 为其全局和。独立近端语音 c 低（不相干），回声主导 c 高。
    {
        if (CexR_.empty()) {
            CexR_.assign(nb, 0.0f); CexI_.assign(nb, 0.0f);
            Cee_.assign(nb, 0.0f); Cxx_.assign(nb, 0.0f);
            Gpost_.assign(nb, 1.0f); GpostT_.assign(nb, 1.0f); Gsm_.assign(nb, 0.0f);
            Cbin_.assign(nb, 0.0f);
            Efg_spec_.resize(M);
        }
        const std::complex<float>* x0 =
            herm_ ? RE->X((RE->xpos() - 1 + K) % K)
                  : X_[(xpos_ - 1 + K) % K].data();
        // 裸 float 复数运算（std::complex 乘法有 NaN 语义分支，不向量化）
        const float ea = kPostEma, eb = 1.0f - kPostEma;
        double coh_sum = 0, pee_sum = 0, pxx_sum = 0;
        for (int m = 0; m < nb; m++) {
            const float xr = x0[m].real(), xi = x0[m].imag();
            const float er = E_[m].real(), ei = E_[m].imag();
            // EMA(X0·E^*)：实= xr·er+xi·ei，虚= xi·er−xr·ei
            CexR_[m] = ea * CexR_[m] + eb * (xr * er + xi * ei);
            CexI_[m] = ea * CexI_[m] + eb * (xi * er - xr * ei);
            Cee_[m] = ea * Cee_[m] + eb * (er * er + ei * ei);
            Cxx_[m] = ea * Cxx_[m] + eb * (xr * xr + xi * xi);
            const float coh = (CexR_[m] * CexR_[m] + CexI_[m] * CexI_[m]) /
                              (Cxx_[m] + 1e-12f);
            coh_sum += coh;
            pee_sum += Cee_[m];
            pxx_sum += Cxx_[m];
            float c = coh / (Cee_[m] + 1e-30f);
            if (c > 1.0f) c = 1.0f;
            Cbin_[m] = c;
        }
        cfrac_sm_ = 0.7 * cfrac_sm_ + 0.3 * (coh_sum / (pee_sum + 1e-30));
        // 后置滤波模式门（去抖）：进入需相干分数持续 kPfPersist 块超阈（排除
        // 短暂相干起伏——AR 类语音素材 2-3 块的相干脉冲会反复开关后置、使
        // 流式预览输出与整块处理发散）；退出滞回即可。
        if (pf_on_) {
            if (cfrac_sm_ < kPfOffCfrac) pf_on_ = false;
        } else if (cfrac_sm_ > kPfOnCfrac) {
            if (++pf_hold_ >= kPfPersist) { pf_on_ = true; pf_hold_ = 0; }
        } else {
            pf_hold_ = 0;
        }
        // 欠消检测（跟踪模式 → Q 放大）：残差以相干回声为主且欠消量显著
        const double rel = pee_sum / (pxx_sum + 1e-30);
        const bool active = pr_ > 1e-12;
        if (active && cfrac_sm_ > kDetCohHi && rel > kDetRelThr) {
            if (det_hold_ < 100000) det_hold_++;
        } else if (!active || cfrac_sm_ < kDetCohLo || rel < kDetRelThr * 0.3) {
            det_hold_ = 0;
        }
        track_mode_ = det_hold_ >= kDetPersist;
        // 每 bin 后置增益：门开时硬膝维纳（c ≤ 膝直通，保护边界 bin 的近端）
        if (herm_)
            smooth5h(Cbin_.data(), Gsm_.data(), M / 2);
        else
            smooth5(Cbin_.data(), Gsm_.data(), M);
        for (int m = 0; m < nb; m++) {
            const float c = Gsm_[m];
            float g = 1.0f;
            if (pf_on_ && c > kPostC0) {
                g = (1.0f - c) / (1.0f - c + kPostKappa * c + 1e-30f);
                if (g < kPostGmin) g = kPostGmin;
            }
            Gpost_[m] = g;
        }
        // 时间平滑（快降/慢升）：门关时增益经慢升通道回 1；跟踪最大增益，
        // 门关且已回满时后置 FFT 整段跳过（双讲/静音段零开销）
        float gmax = 0.0f;
        for (int m = 0; m < nb; m++) {
            float g = Gpost_[m];
            const float gp = GpostT_[m];
            g = g < gp ? (kPostAtk * gp + (1 - kPostAtk) * g)
                       : (kPostRel * gp + (1 - kPostRel) * g);
            GpostT_[m] = g;
            if (g > gmax) gmax = g;
        }
        pf_idle_ = !pf_on_ && gmax > 0.9995f;
    }

    // -- 5) 过程噪声 Q 向量 + P̄ 上下限（并入 R 更新循环之后单次遍历）--- //
    double qsmean = sum_bins_(Stot_sm_.data()) / M;
    const float qreg = kStotRegRel * (float)qsmean + 1e-9f;
    const float qmul = track_mode_ ? kTrackQBoost : 1.0f;   // 欠消跟踪模式 Q 放大
    const float kc = K * kCq, kfl = K * kPFloorRel, kcp = K * kPCapRel;
    for (int m = 0; m < nb; m++) {
        const float den = 1.0f / (Stot_sm_[m] + qreg);
        const float rm = R_[m] * den;
        qv_[m] = kc * qmul * rm;
        pfl_[m] = kfl * rm + 1e-15f;
        pcp_[m] = kcp * rm + 1e-9f;
    }

    // -- 6) 卡尔曼 pass1：加权激励谱 Σg_k|X_k|² → S = R + P̄·Σg_k|X_k|² -- //
    //    新息一致性膨胀（Sage-Husa 强跟踪的乘法形式）：平滑残差 pE 显著
    //    超过预测新息方差 S（W 失配远大于 P̄ 所述：谱突变/回声再出现/路径
    //    变化）时，按超出比例放大 P̄，一块内回到真实误差量级；R 的快升使
    //    双端场景该膨胀仅在近端起始 1-2 块内短暂出现（|E|²≈S 后自动关闭）。
    //    膨胀后的增益仍受 μmax 上限约束（只改修正的方向分配，不改幅度界）。
    for (int m = 0; m < nb; m++) {
        const float s0 = R_[m] + Pbar_[m] * gex_[m] + 1e-12f;
        float scale = 1.0f;                       // 不缩：P̄ 需在静音段保持记忆
        const float ratio = pef_[m] / s0;
        if (ratio > kConsistThr) scale = 0.7f * ratio;   // 残差超预测 → 膨胀
        if (scale > kConsistMax) scale = kConsistMax;
        if (scale > 1.0f) stats_.q_boosts++;
        // 预测：P̄ ← scale·P̄ + Q；S = R + P̄⁻·Σg|X|²（P̄⁻ 为先验）
        Pbar_[m] = Pbar_[m] * scale + qv_[m];
        const float sigp = Pbar_[m] * gex_[m];
        float s = R_[m] + sigp + 1e-12f;
        // 增益上限：μ_total = SigP/S ≤ kMuMax ⟺ S ≥ SigP/μmax
        const float smin = sigp / kMuMax;
        if (s < smin) s = smin;
        // 新息离群抑制（稳健卡尔曼）：当前残差（未平滑谱）显著超过预测方
        // 差 S 时，视为测量噪声突增（近端语音起始/噪声突发）而非状态误差
        // ——瞬时抬高本块 S 抑制增益，待 R 跟上后自动恢复；稳态下该钳位
        // 还等效压低平均步长（更深收敛）。对回声再出现仅延迟 1-2 块。
        const float sout = kOutlierC * std::norm(E_[m]);
        if (s < sout) s = sout;
        Sm_[m] = s;
        invSm_[m] = 1.0f / s;
    }

    if (!ref_silent) {
    // -- 7) 卡尔曼 pass2：W_k += P̄·g_k·X_k^H/S·E；P̄ ← (1-μ_total)·P̄ ---- //
    //    同时累积 W 能量（g 的学习数据 + 块份额 EMA，供证据自适应上限）
    for (int m = 0; m < nb; m++) gnew_[m] = 0.0f;   // Σ_k|W_k[m]|²
    for (int k = 0; k < K; k++) wsum_f_[k] = 0.0f;
    for (int k = 0; k < K; k++) {
        const float* __restrict gk = G_[k].data();
        std::complex<float>* __restrict wk = W_[k].data();
        const std::complex<float>* __restrict xk =
            herm_ ? RE->X((RE->xpos() - 1 - k + 2 * K) % K)
                  : X_[(xpos_ - 1 - k + 2 * K) % K].data();
        const float* __restrict pb = Pbar_.data();
        const float* __restrict is = invSm_.data();
        const std::complex<float>* __restrict ev = E_.data();
        float* __restrict gn = gnew_.data();
        for (int m = 0; m < nb; m++) {
            const float c = pb[m] * gk[m] * is[m];
            const float gr = c * xk[m].real(), gi = -c * xk[m].imag();
            const float er = ev[m].real(), ei = ev[m].imag();
            const float nr = wk[m].real() + (gr * er - gi * ei);
            const float ni = wk[m].imag() + (gr * ei + gi * er);
            wk[m] = std::complex<float>(nr, ni);
            gn[m] += nr * nr + ni * ni;
            wsum_f_[k] += nr * nr + ni * ni;          // float 累积（不破坏向量化）
        }
    }
    for (int k = 0; k < K; k++)
        wshare_ema_[k] = 0.95 * wshare_ema_[k] + 0.05 * (double)wsum_f_[k];
    }  // 静音门控①：X=0 ⟹ 更新项=0、gnew/wsum 残留值仍精确（W 未变），跳过
             // 主循环；下方清零也一并跳（防 gnew 清零破坏段 8 的 share 语义）

    for (int m = 0; m < nb; m++) {
        float drop = Pbar_[m] * g2ex_[m] * invSm_[m];  // 总协方差下降（Σg²|X|²）
        if (drop > 1.0f) drop = 1.0f;
        Pbar_[m] = clampf(Pbar_[m] * (1.0f - drop), pfl_[m], pcp_[m]);
    }

    // -- 8) 分块份额 g 学习：|W_k|² 分布的 EMA + 衰减先验混合 + 上限 ------ //
    //    房间冲激响应随延迟指数衰减：晚期分块的协方差份额被先验 w̄_k ∝ ρ^k
    //    与上限压低。这同时是双端保护的物理形态——真实回声集中在早块，
    //    参考信号的长时间延迟复现（近端"同内容语音"）不会被判为回声路径
    //    而被吸收（P_k = P̄·g_k，g 上限直接限制该块的学习能力）。
    //    上限为证据自适应（FIXES_AEC 缺陷2）：块 k 的 W 能量份额持续超过
    //    先验预期时按比率放宽——真实晚期回声（如 ≥50ms 纯延迟路径）获得
    //    学习能力；存在强早期路径时晚期块相对份额低（分母被早期路径占据），
    //    上限不动，近端同源内容防护（S2）原样保留。
    for (int m = 0; m < nb; m++) invg_[m] = 1.0f / (gnew_[m] + 1e-30f);  // |W|² 倒数
    std::fill(gnew_.begin(), gnew_.end(), 0.0f);                        // 复用为 Σg
    {
        double tot = 0.0;
        for (int k = 0; k < K; k++) tot += wshare_ema_[k];
        if (tot > 1e-30) {
            for (int k = 0; k < K; k++) {
                const double ratio = (wshare_ema_[k] / tot) / gprofile_[k];
                float relax = (float)ratio;
                if (relax < 1.0f) relax = 1.0f;
                if (relax > kCapRelaxMax) relax = kCapRelaxMax;
                gcap_use_[k] = gcapk_[k] * relax;
                if (gcap_use_[k] > 1.0f) gcap_use_[k] = 1.0f;
            }
        } else {
            for (int k = 0; k < K; k++) gcap_use_[k] = gcapk_[k];
        }
    }
    for (int k = 0; k < K; k++) {
        float* gk = G_[k].data();
        const std::complex<float>* wk = W_[k].data();
        const float prior = gprofile_[k];
        const float gcap = gcap_use_[k];
        for (int m = 0; m < nb; m++) {
            const float share = std::norm(wk[m]) * invg_[m];
            float g = (1.0f - kGAlpha) * gk[m] + kGAlpha * share;
            g = (1.0f - kGUniform) * g + kGUniform * prior;
            gk[m] = g > gcap ? gcap : g;
            gnew_[m] += gk[m];
        }
    }
    for (int m = 0; m < nb; m++) invg_[m] = gnew_[m] > 1e-30f ? 1.0f / gnew_[m] : 0.0f;
    for (int k = 0; k < K; k++) {
        float* gk = G_[k].data();
        const float* inv = invg_.data();
        for (int m = 0; m < nb; m++) gk[m] *= inv[m];
    }


    // -- 9) W 时域约束：子滤波器冲激响应限制在前 B 抽头 ------------------ //
    //    herm_ 快路径：错峰投影——每块只做 per 个连续轮转分区（per=2），
    //    每个分区仍每 K/per=5 块投影一次（原周期 4 块，绕回污染存活期同量
    //    级），摊销 2·per=4 FFT/块（原 2K/4=5）。per=1（周期 10 块）实测
    //    稳态 ERLE 掉 0.8dB，不取。48k 全谱路径保持原调度（逐位基线）。
    {
        const auto project_k = [&](int k) {
            if (herm_) {
                for (int m = 0; m < nb; m++) fftbuf_[m] = W_[k][m];
                herm_expand_(W_[k]);
            } else {
                fftbuf_ = W_[k];
            }
            fft_.run(fftbuf_.data(), true);
            for (int i = 0; i < B; i++) fftbuf_[i] = std::complex<float>(fftbuf_[i].real(), 0.0f);
            for (int i = B; i < M; i++) fftbuf_[i] = std::complex<float>(0.0f, 0.0f);
            fft_.run(fftbuf_.data(), false);
            if (herm_)
                for (int m = 0; m < nb; m++) W_[k][m] = fftbuf_[m];
            else
                W_[k] = fftbuf_;
        };
        if (herm_) {
            const int per = 2;  // 16k: 每块 2 分区轮转
            const int base = (int)(((stats_.blocks - 1) * (long)per) % (long)K);
            for (int j = 0; j < per && j < K; j++) project_k((base + j) % K);
        } else if (stats_.blocks % kProjEvery == 0) {
            for (int k = 0; k < K; k++) project_k(k);
        }
    }



    // -- 10) 发散保护（极端异常时复位自适应状态）----------------------- //
    {
        const double pe = peb, pd = pd_, pr = pr_;
        if (pr > 1e-12 && pe > kDivergeGain * (pd + pr)) {
            for (int k = 0; k < K; k++) {
                std::fill(W_[k].begin(), W_[k].end(), std::complex<float>());
                std::fill(G_[k].begin(), G_[k].end(), gprofile_[k]);
            }
            std::fill(Pbar_.begin(), Pbar_.end(), 0.0f);
            std::fill(R_.begin(), R_.end(), 0.0f);
            std::fill(Stot_sm_.begin(), Stot_sm_.end(), 0.0f);
            std::fill(inited_.begin(), inited_.end(), 0);
            act_blocks_ = 0;
            stats_.resets++;
        }
    }

    // -- 11) 后置维纳滤波（FIXES_AEC）：前景残差谱 × 平滑增益，重叠保留 -- //
    //    Efg 谱 [0_B|e] × G[m] 后逆变换取后 B 样本（G 经谱平滑，等效时域
    //    滤波器支撑远小于 B，无回绕污染）。预览块（run_tail_）用冻结的
    //    预块增益：零填充块的相干统计与将来真实整块不同，若用预览块自身
    //    统计的增益，发出的样本与单次调用路径发散（实测 rel 6e-3→2.9e-1）。
    if (post_cnt_ >= kPostWarm && !(preview_mode_ ? pf_idle_prev_ : pf_idle_)) {
        const float* guse = preview_mode_ ? GpostPrev_.data() : GpostT_.data();
        std::fill(Efg_spec_.begin(), Efg_spec_.begin() + B, std::complex<float>());
        for (int i = 0; i < B; i++) Efg_spec_[B + i] = std::complex<float>(efg_blk_[i], 0.0f);
        fft_.run(Efg_spec_.data(), false);
        if (herm_) {
            // 实对称增益 × Hermitian 谱 → Hermitian：半谱相乘后镜像共轭填充
            for (int m = 0; m < nb; m++) Efg_spec_[m] *= guse[m];
            for (int m = nb; m < M; m++)
                Efg_spec_[m] = std::conj(Efg_spec_[M - m]);
        } else {
            for (int m = 0; m < M; m++) Efg_spec_[m] *= guse[m];
        }
        fft_.run(Efg_spec_.data(), true);
        for (int i = 0; i < B; i++) out[i] = Efg_spec_[B + i].real();
    }
    post_cnt_++;
    // xpos_ 已在段 1 随参考谱入环前进（herm 经 RefEngine.push、legacy 就地）
}

// ------------------------------------------------------------------ //
// 块尾不足 B 的"预览"处理：快照状态 → 零填充处理 → 取前 r 个输出 → 回滚
// ------------------------------------------------------------------ //
void PBFDKF::run_tail_(const float* mic, const float* ref, int r, float* out) {
    if (r <= 0) return;
    if (r > B_) r = B_;

    // 快照自适应状态（含 FIXES_AEC 新增的后置滤波/检测/软先验状态）
    auto x_time = x_time_;
    auto X = X_;
    auto ownref = own_ref_;   // herm 独立模式的参考谱状态（共享模式无预览）
    auto W = W_;
    auto Wfg = Wfg_;
    auto G = G_;
    auto Pbar = Pbar_;
    auto R = R_;
    auto Stot = Stot_sm_;
    auto inited = inited_;
    auto CexR = CexR_;
    auto CexI = CexI_;
    auto Cee = Cee_;
    auto Cxx = Cxx_;
    auto GpostT = GpostT_;
    auto wshare = wshare_ema_;
    auto gcap_use = gcap_use_;
    const int xpos = xpos_;
    const int act = act_blocks_;
    const int hold = copy_hold_;
    const double ebg = Ebg_, efg = Efg_;
    const double cfrac = cfrac_sm_;
    const bool pf_on = pf_on_;
    const int det_hold = det_hold_;
    const bool track_mode = track_mode_;
    const int post_cnt = post_cnt_;
    const bool pf_idle = pf_idle_;
    const int pf_hold = pf_hold_;
    const Stats stats = stats_;   // 预览不计入块统计——stats_.blocks 同时是 W
                                 // 时域投影（kProjEvery）的调度相位，若保留预览
                                 // 计数会使自适应轨迹依赖调用分块方式；一并回滚
                                 // 后任意 n 分块与单次调用逐位一致。

    GpostPrev_ = GpostT_;        // 冻结预块后置增益（预览输出一致性）
    pf_idle_prev_ = pf_idle_;    // 门控同样冻结（预览统计不污染输出路径）
    preview_mode_ = true;
    std::fill(tmic_.begin(), tmic_.end(), 0.0f);
    std::fill(tref_.begin(), tref_.end(), 0.0f);
    std::copy(mic, mic + r, tmic_.data());
    std::copy(ref, ref + r, tref_.data());
    process_block_(tmic_.data(), tref_.data(), tout_.data());
    preview_mode_ = false;
    std::copy(tout_.begin(), tout_.begin() + r, out);

    // 回滚（全部状态，含 stats_：预览对自适应轨迹与诊断均"透明"）
    stats_ = stats;
    x_time_ = std::move(x_time);
    X_ = std::move(X);
    own_ref_ = std::move(ownref);
    W_ = std::move(W);
    Wfg_ = std::move(Wfg);
    G_ = std::move(G);
    Pbar_ = std::move(Pbar);
    R_ = std::move(R);
    Stot_sm_ = std::move(Stot);
    inited_ = std::move(inited);
    CexR_ = std::move(CexR);
    CexI_ = std::move(CexI);
    Cee_ = std::move(Cee);
    Cxx_ = std::move(Cxx);
    GpostT_ = std::move(GpostT);
    wshare_ema_ = std::move(wshare);
    gcap_use_ = std::move(gcap_use);
    xpos_ = xpos;
    act_blocks_ = act;
    copy_hold_ = hold;
    Ebg_ = ebg;
    Efg_ = efg;
    cfrac_sm_ = cfrac;
    pf_on_ = pf_on;
    det_hold_ = det_hold;
    track_mode_ = track_mode;
    post_cnt_ = post_cnt;
    pf_idle_ = pf_idle;
    pf_hold_ = pf_hold;
}

// ------------------------------------------------------------------ //
// 流式接口：任意 n，内部按块缓冲；不足块用零填充预览输出（状态不污染）
// ------------------------------------------------------------------ //
void PBFDKF::process(const float* mic, const float* ref, float* out, int n) {
    if (n <= 0) return;
    pend_mic_.insert(pend_mic_.end(), mic, mic + n);
    pend_ref_.insert(pend_ref_.end(), ref, ref + n);

    // -- 整块处理：块首若覆盖此前已"预览"输出的样本（previewed_ > 0），该部分
    //    块输出丢弃（流位置早已发出），只追加块尾新样本的输出。
    while (pend_mic_.size() - pend_off_ >= (size_t)B_) {
        process_block_(pend_mic_.data() + pend_off_, pend_ref_.data() + pend_off_,
                       tout_.data());
        const int drop = previewed_ < B_ ? previewed_ : B_;
        outq_.insert(outq_.end(), tout_.begin() + drop, tout_.end());
        previewed_ -= drop;
        pend_off_ += B_;
    }
    if (pend_off_ > (1u << 16)) {
        pend_mic_.erase(pend_mic_.begin(), pend_mic_.begin() + pend_off_);
        pend_ref_.erase(pend_ref_.begin(), pend_ref_.begin() + pend_off_);
        pend_off_ = 0;
    }

    // -- 输出不足 n 时，对未入块的新样本零填充"预览"补齐（状态回滚不污染）。
    //    预览块从块边界 pend_off_ 起算（与将来真实整块同一起点/滤波状态），
    //    取其中块内偏移 [previewed_, previewed_+r) 的输出——恰为尚未发出的
    //    流位置；已预览部分（[0, previewed_)）不重复输出。
    while (outq_.size() - out_head_ < (size_t)n) {
        const int need = n - (int)(outq_.size() - out_head_);
        const int rem = (int)(pend_mic_.size() - pend_off_);
        int r = rem - previewed_;               // 尚未输出的新样本数
        if (r > need) r = need;
        if (r <= 0) break;
        const int r2 = previewed_ + r;          // ≤ rem < B_
        run_tail_(pend_mic_.data() + pend_off_, pend_ref_.data() + pend_off_, r2,
                  tout_.data());
        outq_.insert(outq_.end(), tout_.begin() + previewed_, tout_.begin() + r2);
        previewed_ += r;
    }

    std::copy(outq_.begin() + out_head_, outq_.begin() + out_head_ + n, out);
    out_head_ += n;
    if (out_head_ > (1u << 16)) {
        outq_.erase(outq_.begin(), outq_.begin() + out_head_);
        out_head_ = 0;
    }
}

std::vector<float> PBFDKF::process_file(const std::vector<float>& mic,
                                        const std::vector<float>& ref) {
    if (mic.size() != ref.size()) return {};
    std::vector<float> out(mic.size());
    if (!mic.empty())
        process(mic.data(), ref.data(), out.data(), (int)mic.size());
    return out;
}

// ------------------------------------------------------------------ //
// 共享参考谱模式：挂外部 RefEngine；process_shared 仅接受 B 整数倍
//（无预览路径），每块读共享环最新状态（调用方须先 push 对应参考块）。
// ------------------------------------------------------------------ //
void PBFDKF::attach_shared(RefEngine* re) {
    if (herm_ && re && re->nb() == nb_ && re->xpos() >= 0) shared_ = re;
}

void PBFDKF::process_shared(const float* mic, const float* ref, float* out, int n) {
    for (int off = 0; off + B_ <= n; off += B_)
        process_block_(mic + off, ref + off, out + off);
}
