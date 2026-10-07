// afecpp/bf.cpp — DOA（GCC-PHAT 导向网格搜索）+ Beamformer（DSB / MVDR）实现。
// 纯数值处理，不播放任何音频。参考：../afe/bf.py（行为对齐）。

#include "bf.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "fft_bf.hpp"

namespace afe {

// ---------------------------------------------------------------------------
// 窗 sinc 分数延迟：out(t) = x(t - d_sec)（与 bf.py frac_shift 约定一致）
// ---------------------------------------------------------------------------
std::vector<float> frac_delay(const std::vector<float>& x, double d_sec, double sr,
                              int half) {
    const int64_t n = (int64_t)x.size();
    std::vector<float> out((size_t)n);
    const double d = -d_sec * sr;  // 采样偏移 = -d_sec*sr（out(t)=x(t-d_sec)）
    // 核只依赖 frac(d)，预先算好 2*half 个系数（t = j - i0），避免逐样点三角函数
    std::vector<double> kern((size_t)(2 * half));
    const double fr = d - std::floor(d);
    for (int t = -half + 1; t <= half; ++t) {
        double u = fr - t;
        double s = (std::fabs(u) < 1e-12) ? 1.0 : std::sin(M_PI * u) / (M_PI * u);
        double w = 0.5 * (1.0 + std::cos(M_PI * u / half));  // Hann 窗
        kern[(size_t)(t + half - 1)] = s * w;
    }
    const int64_t base = (int64_t)std::floor(d);
    for (int64_t i = 0; i < n; ++i) {
        double acc = 0.0;
        for (int t = -half + 1; t <= half; ++t) {
            int64_t j = i + base + t;
            if (j < 0 || j >= n) continue;
            acc += (double)x[(size_t)j] * kern[(size_t)(t + half - 1)];
        }
        out[(size_t)i] = (float)acc;
    }
    return out;
}

// ---------------------------------------------------------------------------
// DOA
// ---------------------------------------------------------------------------
DOA::DOA(int nch, int sr, double radius, double c, double grid_step_deg,
         double refine_step_deg, double max_dur_s, const char* array_type)
    : nch_(nch),
      sr_((double)sr),
      geom_(radius),
      c_(c),
      grid_step_(grid_step_deg),
      refine_step_(refine_step_deg) {
    if (nch_ < 2) nch_ = 2;
    ula_ = (array_type && std::strcmp(array_type, "ula") == 0);
    // 最大孔径 D：UCA=直径 2r；ULA=(nch-1)*d。GCC 限带 c/(2D) —— 全麦对
    // 无空间混叠（最疏麦对半波长采样）的上限。
    if (ula_) {
        if (!(geom_ > 0.0)) geom_ = 0.04;  // ULA 默认间距 40 mm
        aperture_ = (nch_ - 1) * geom_;
        pos_.resize(nch_);
        for (int m = 0; m < nch_; ++m) pos_[m] = (m - (nch_ - 1) / 2.0) * geom_;
    } else {
        if (!(geom_ > 0.0)) geom_ = 0.035;  // UCA 默认半径 3.5 cm
        aperture_ = 2.0 * geom_;
        phi_.resize(nch_);
        for (int m = 0; m < nch_; ++m) phi_[m] = 2.0 * M_PI * m / nch_;
    }
    fmax_ = c_ / (2.0 * aperture_);
    // GCC 分块 FFT 长：约 0.34 s 的 2 的幂（48k→16384 与旧版逐位一致；
    // 16k→8192 ≈ 0.512 s）。块长只影响分块统计的粒度，相关峰宽度由
    // fmax（几何决定）与 sr 共同决定，max_lag/NB 均按 sr 参数化。
    gcc_block_ = FFT_BF::next_pow2((int)(0.34 * sr_ + 0.5));
    if (gcc_block_ < 1024) gcc_block_ = 1024;
    if (gcc_block_ > 16384) gcc_block_ = 16384;
    for (int i = 0; i < nch_; ++i)
        for (int j = i + 1; j < nch_; ++j) pairs_.emplace_back(i, j);
    max_samples_ = (int)(max_dur_s * sr_ + 0.5);
}

void DOA::delays(double theta_rad, std::vector<double>& tau) const {
    tau.resize(nch_);
    if (ula_) {
        const double s = std::sin(theta_rad);
        for (int m = 0; m < nch_; ++m) tau[m] = pos_[m] * s / c_;
    } else {
        for (int m = 0; m < nch_; ++m)
            tau[m] = (geom_ / c_) * std::cos(theta_rad - phi_[m]);
    }
}

// ULA DOA：PHAT 白化（限带 <= fmax，去直流）后，在每个网格角 theta 以旋转
// 相量精确计算全部分数 lag 处的互相关并求和：
//     score(theta) = sum_p (2/B) * Re{ sum_{1<=b<NB} w_pb * e^{+j2pi lag_p(theta) b/B} }
// 为何不用 UCA 的"整数 lag cc + 线性插值"：ULA 的 GCC 限带更低
// （fmax = c/(2*孔径)，4 麦 d=40mm 时约 1.43 kHz），相关峰宽约 33 样点、
// 峰顶平坦，线性插值使 score 分段线性、极大值钉到最近整数 lag 上
// （如相邻对 2.8 样点被钉到 3），实测引入 2-6° 系统性 |sin(theta)| 高估。
// 精确求值代价 ~网格数 x 麦对 x NB 个复乘（<1 ms），死通道对 w 恒 0 自动剔除。
// 网格：-90..+90°（线阵前后模糊 sin 对称，只搜单侧，结果即 ±theta 等价类
// 代表），1° 步进 + 峰值 ±1.5°、0.05° 细化，与 UCA 同思路。
double DOA::ula_search_(const std::vector<std::vector<std::complex<float>>>& acc, int B) {
    const int nb2 = B / 2 + 1;
    const int NB = std::min(nb2, (int)(fmax_ * B / sr_) + 2);  // f <= fmax 的 bin 数
    const size_t P = pairs_.size();
    const int NBm = std::max(1, NB);
    // 白化谱存 float 平铺（热循环用；白化本身仍在 double 精度无损）
    std::vector<float> wbr((size_t)P * NBm, 0.0f), wbi((size_t)P * NBm, 0.0f);
    for (size_t p = 0; p < P; ++p) {
        float mx = 0.0f;
        for (int b = 0; b < nb2; ++b) mx = std::max(mx, std::abs(acc[p][b]));
        if (!(mx > 0.0f) || !std::isfinite(mx)) continue;  // 死通道对：wb 恒 0
        float thr = 1e-12f * mx;
        for (int b = 1; b < NB; ++b) {
            float mg = std::abs(acc[p][b]);
            if (mg > thr) {
                wbr[p * NBm + b] = acc[p][b].real() / mg;
                wbi[p * NBm + b] = acc[p][b].imag() / mg;
            }
        }
    }
    std::vector<double> tau(nch_);
    auto score = [&](double theta_deg) -> double {
        delays(theta_deg * M_PI / 180.0, tau);
        double s = 0.0;
        for (size_t p = 0; p < P; ++p) {
            int i = pairs_[p].first, j = pairs_[p].second;
            double lag = (tau[i] - tau[j]) * sr_;
            const double ang = 2.0 * M_PI * lag / B;
            // 相位递推 float（A7 无 double NEON）；幅值漂移每 64 bin 重归一
            // 控制在 ~1e-6，远小于 0.05° 细化网格的峰位分辨
            const float rr = (float)std::cos(ang), ri = (float)std::sin(ang);
            const float* __restrict wr = wbr.data() + p * NBm;
            const float* __restrict wi = wbi.data() + p * NBm;
            float phr = 1.0f, phi = 0.0f, sumr = 0.0f;
            for (int b = 1; b < NB; ++b) {
                sumr += wr[b] * phr - wi[b] * phi;
                const float nr = phr * rr - phi * ri;
                phi = phr * ri + phi * rr;
                phr = nr;
                if ((b & 63) == 0) {
                    const float g = 1.0f / std::sqrt(phr * phr + phi * phi);
                    phr *= g;
                    phi *= g;
                }
            }
            s += (2.0 / B) * (double)sumr;
        }
        return s;
    };
    double best_deg = -90.0, best = -1e300;
    for (double d = -90.0; d <= 90.0 + 1e-9; d += grid_step_) {
        double s = score(d);
        if (s > best) { best = s; best_deg = d; }
    }
    for (double d = best_deg - 1.5; d <= best_deg + 1.5 + 1e-9; d += refine_step_) {
        double s = score(d);
        if (s > best) { best = s; best_deg = d; }
    }
    return std::max(-90.0, std::min(90.0, best_deg));
}

double DOA::estimate(const std::vector<std::vector<float>>& x) {
    last_ = 0.0;
    if ((int)x.size() < nch_) return 0.0;
    int64_t N = (int64_t)x[0].size();
    for (int m = 1; m < nch_; ++m) N = std::min(N, (int64_t)x[m].size());
    if (N > max_samples_) N = max_samples_;  // 限制分析长度（默认 4 s，RTF 考量）
    if (N < 64) return 0.0;

    // ---- 分块 GCC-PHAT：每通道 FFT 复用于所有麦对，互谱分块累加后取平均 ----
    // 块长 gcc_block_ = next_pow2(0.34s·sr)（48k: 16384 ≈ 0.34 s，与旧版一致；
    // 16k: 8192 ≈ 0.512 s）：正/逆变换总计算量与块数平衡，相关峰不受块界影响
    // （实际 TDOA |lag| <= 孔径/c*sr，远 << 块长）。
    const int B = (N < gcc_block_) ? FFT_BF::next_pow2((int)N) : gcc_block_;
    const int nb2 = B / 2 + 1;
    const int nblocks = (int)((N + B - 1) / B);
    const FFT_BF fft(B);

    std::vector<std::vector<std::complex<float>>> acc(pairs_.size(),
                                                      std::vector<std::complex<float>>(nb2));
    std::vector<std::vector<std::complex<float>>> spec(nch_);
    for (int m = 0; m < nch_; ++m) spec[m].assign(B, {0.f, 0.f});
    std::vector<float> blk(B);

    for (int bi = 0; bi < nblocks; ++bi) {
        int64_t s0 = (int64_t)bi * B;
        int len = (int)std::min<int64_t>(B, N - s0);
        for (int m = 0; m < nch_; ++m) {
            double mean = 0.0;
            for (int i = 0; i < len; ++i) {
                blk[i] = x[m][(size_t)(s0 + i)];
                mean += blk[i];
            }
            mean /= std::max(1, len);
            for (int i = 0; i < B; ++i)
                blk[i] = (i < len) ? (float)(blk[i] - mean) : 0.0f;
            for (int i = 0; i < B; ++i) spec[m][i] = {blk[i], 0.f};
            fft.forward(spec[m]);
        }
        for (size_t p = 0; p < pairs_.size(); ++p) {
            int i = pairs_[p].first, j = pairs_[p].second;
            for (int b = 0; b < nb2; ++b) acc[p][b] += spec[i][b] * std::conj(spec[j][b]);
        }
    }
    return search_from_acc_(acc, B, 1.0 / nblocks);
}

// 块平均互谱 → 缩放 → ULA/UCA 搜索（estimate 与 inc_estimate 共用）。
// 纯代码搬移自 estimate() 尾段（数值行为不变）。
double DOA::search_from_acc_(std::vector<std::vector<std::complex<float>>>& acc, int B,
                             double inv_blocks) {
    for (auto& a : acc)
        for (auto& z : a) z *= (float)inv_blocks;

    // ---- ULA：精确分数 lag 导向网格搜索（见 ula_search_ 注释） ----
    if (ula_) {
        last_ = ula_search_(acc, B);
        return last_;
    }


    const int nb2 = B / 2 + 1;
    // ---- PHAT 白化（限带 <= fmax，去直流），直接在带内求小 lag 互相关 ----
    // cc[k] = (2/B) * Re{ sum_{1<=b<NB} W_b e^{j2pi k b / B} }，W_b = G_b/|G_b|。
    // 只需 |k| <= max_lag 的小 lag 区（导向插值用），无需整段逆 FFT。
    const int max_lag_samples =
        std::max(16, (int)std::ceil(aperture_ / c_ * sr_) + 2);
    const int NB = std::min(nb2, (int)(fmax_ * B / sr_) + 2);  // f <= fmax 的 bin 数
    std::vector<std::vector<float>> cc(pairs_.size());  // cc[p][k + max_lag] : lag k
    std::vector<float> peak(pairs_.size(), 0.0f);
    for (size_t p = 0; p < pairs_.size(); ++p) {
        float mx = 0.0f;
        for (int b = 0; b < nb2; ++b) mx = std::max(mx, std::abs(acc[p][b]));
        cc[p].assign(2 * max_lag_samples + 1, 0.0f);
        if (!(mx > 0.0f) || !std::isfinite(mx)) continue;  // 死通道对：cc 恒 0
        float thr = 1e-12f * mx;
        float pk = 0.0f;
        for (int k = -max_lag_samples; k <= max_lag_samples; ++k) {
            const double ang = 2.0 * M_PI * k / B;
            const std::complex<double> step(std::cos(ang), std::sin(ang));
            std::complex<double> rot(1.0, 0.0), accv(0.0, 0.0);
            for (int b = 1; b < NB; ++b) {  // b=0 直流排除
                float mg = std::abs(acc[p][b]);
                if (mg > thr) {
                    std::complex<double> wb(acc[p][b].real() / mg, acc[p][b].imag() / mg);
                    accv += wb * rot;
                }
                rot *= step;
            }
            float v = (float)(2.0 * accv.real() / B);
            cc[p][(size_t)(k + max_lag_samples)] = v;
            pk = std::max(pk, std::fabs(v));
        }
        peak[p] = pk;
    }

    // ---- 导向得分网格搜索：score(theta) = sum_{i<j} peak_ij * cc_ij(lag_ij(theta)) ----
    auto cc_at = [&](size_t p, double lag) -> float {
        double u = lag + max_lag_samples;
        if (u < 0.0) u = 0.0;
        if (u > 2.0 * max_lag_samples) u = 2.0 * max_lag_samples;
        int lo = (int)u;
        double fr = u - lo;
        if (lo >= 2 * max_lag_samples) return cc[p][(size_t)2 * max_lag_samples];
        return (float)((1.0 - fr) * cc[p][(size_t)lo] + fr * cc[p][(size_t)lo + 1]);
    };
    std::vector<double> tau(nch_);
    auto score = [&](double theta_deg) -> double {
        double th = theta_deg * M_PI / 180.0;
        delays(th, tau);
        double s = 0.0;
        for (size_t p = 0; p < pairs_.size(); ++p) {
            if (!(peak[p] > 1e-12f)) continue;
            int i = pairs_[p].first, j = pairs_[p].second;
            s += (double)peak[p] * cc_at(p, (tau[i] - tau[j]) * sr_);
        }
        return s;
    };

    // UCA 全周搜索 [0,360)°，峰值附近细化（±1.5°，0.05° 步进）
    double best_deg = 0.0, best = -1e300;
    for (double d = 0.0; d < 360.0; d += grid_step_) {
        double s = score(d);
        if (s > best) { best = s; best_deg = d; }
    }
    for (double d = best_deg - 1.5; d <= best_deg + 1.5 + 1e-9; d += refine_step_) {
        double s = score(d);
        if (s > best) { best = s; best_deg = d; }
    }
    last_ = std::fmod(best_deg, 360.0);
    if (last_ < 0) last_ += 360.0;
    return last_;
}

// ---------------------------------------------------------------------------
// 增量 GCC（流式快速路径）：满块才 FFT，通道 Hermitian 半谱入槽环；估计时
// 从槽全量重建互谱（无增量漂移）。与 estimate() 在"分析窗恰为整数个完整
// 块"时逐位一致——同 B、同逐块去均值、同槽序（最旧→最新）累加、同缩放。
// ---------------------------------------------------------------------------
void DOA::inc_reset() {
    inc_ = Inc{};
    inc_fft_.reset();
}

int DOA::inc_push(const float* const* x, int n) {
    if (n <= 0 || x == nullptr) return 0;
    if (inc_.B == 0) {   // 惰性初始化：槽环容量 = ceil(分析窗/块长)
        inc_.B = gcc_block_;
        inc_.nb2 = gcc_block_ / 2 + 1;
        inc_.nslots = (max_samples_ + gcc_block_ - 1) / gcc_block_;
        if (inc_.nslots < 1) inc_.nslots = 1;
        inc_.stage.assign((size_t)nch_ * gcc_block_, 0.0f);
        inc_.slot.assign((size_t)inc_.nslots * nch_ * inc_.nb2,
                         std::complex<float>{0.0f, 0.0f});
        inc_.zbuf.assign(gcc_block_, std::complex<float>{0.0f, 0.0f});
        inc_fft_ = std::make_unique<FFT_BF>(gcc_block_);
    }
    int done = 0;
    int off = 0;
    while (off < n) {
        const int take = std::min(inc_.B - inc_.staged, n - off);
        for (int m = 0; m < nch_; ++m)
            std::copy(x[m] + off, x[m] + off + take,
                      inc_.stage.begin() + (size_t)m * inc_.B + inc_.staged);
        inc_.staged += take;
        off += take;
        if (inc_.staged < inc_.B) break;
        // 整块完成：逐通道去均值（与 estimate() 整块路径同式）→ FFT → 半谱入槽
        const int slot = inc_.wpos;
        for (int m = 0; m < nch_; ++m) {
            const float* s = inc_.stage.data() + (size_t)m * inc_.B;
            double mean = 0.0;
            for (int i = 0; i < inc_.B; ++i) mean += s[i];
            mean /= inc_.B;
            for (int i = 0; i < inc_.B; ++i)
                inc_.zbuf[i] = std::complex<float>((float)(s[i] - mean), 0.0f);
            inc_fft_->forward(inc_.zbuf);
            std::copy(inc_.zbuf.begin(), inc_.zbuf.begin() + inc_.nb2,
                      inc_.slot.begin() + ((size_t)slot * nch_ + (size_t)m) * inc_.nb2);
        }
        inc_.wpos = (inc_.wpos + 1) % inc_.nslots;
        if (inc_.filled < inc_.nslots) inc_.filled++;
        inc_.staged = 0;
        ++done;
    }
    inc_.total += n;
    return done;
}

double DOA::inc_estimate() {
    if (inc_.filled == 0 || inc_.B == 0) return 0.0;
    const int B = inc_.B, nb2 = inc_.nb2;
    if (inc_.acc.empty())
        inc_.acc.assign(pairs_.size(), std::vector<std::complex<float>>(nb2));
    for (auto& a : inc_.acc)
        std::fill(a.begin(), a.end(), std::complex<float>{0.0f, 0.0f});
    // 从最旧槽开始累加（与批式分块时间序一致 → 逐位一致）
    const int first = (inc_.filled == inc_.nslots) ? inc_.wpos : 0;
    for (int s = 0; s < inc_.filled; ++s) {
        const int slot = (first + s) % inc_.nslots;
        const std::complex<float>* base =
            inc_.slot.data() + (size_t)slot * nch_ * nb2;
        for (size_t p = 0; p < pairs_.size(); ++p) {
            const std::complex<float>* xi = base + (size_t)pairs_[p].first * nb2;
            const std::complex<float>* xj = base + (size_t)pairs_[p].second * nb2;
            std::complex<float>* a = inc_.acc[p].data();
            for (int b = 0; b < nb2; ++b) a[b] += xi[b] * std::conj(xj[b]);
        }
    }
    return search_from_acc_(inc_.acc, B, 1.0 / inc_.filled);
}


// ---------------------------------------------------------------------------
// 复数工具：Hermitian Jacobi 最小特征值 + Gauss-Jordan 线性求解（6x6，手写）
// ---------------------------------------------------------------------------
namespace {

using cd = std::complex<double>;

// 复 Hermitian 矩阵的最小特征值（仅特征值，循环 Jacobi 两对角化）。
double hermitian_min_eig(std::vector<std::vector<cd>> A, int n, double scale) {
    const double eps_off = 1e-24 * std::max(scale * scale, 1e-300);
    for (int sweep = 0; sweep < 24; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q) off += std::norm(A[p][q]);
        if (off < eps_off) break;
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                cd c = A[p][q];
                double ac = std::abs(c);
                if (ac < 1e-300) continue;
                double a = A[p][p].real(), b = A[q][q].real();
                double tau = (b - a) / (2.0 * ac);
                double t = (tau >= 0.0 ? 1.0 : -1.0) /
                           (std::fabs(tau) + std::sqrt(1.0 + tau * tau));
                double cs = 1.0 / std::sqrt(1.0 + t * t), sn = t * cs;
                // 相位旋转使 A[p][q] 变实（酉对角相似，特征值不变）
                cd eta = std::polar(1.0, -std::arg(c));  // e^{-i*arg(c)}
                cd eta_c = std::conj(eta);
                for (int k = 0; k < n; ++k) {
                    if (k == p || k == q) continue;
                    A[k][q] *= eta;
                    A[q][k] *= eta_c;
                }
                // 实 Jacobi 旋转消 A[p][q]
                A[p][p] = a - t * ac;
                A[q][q] = b + t * ac;
                A[p][q] = {0.0, 0.0};
                A[q][p] = {0.0, 0.0};
                for (int k = 0; k < n; ++k) {
                    if (k == p || k == q) continue;
                    cd apk = A[k][p], aqk = A[k][q];
                    A[k][p] = cs * apk - sn * aqk;
                    A[k][q] = sn * apk + cs * aqk;
                    A[p][k] = std::conj(A[k][p]);
                    A[q][k] = std::conj(A[k][q]);
                }
            }
        }
    }
    double mn = A[0][0].real();
    for (int k = 1; k < n; ++k) mn = std::min(mn, A[k][k].real());
    return mn;
}

// Gauss-Jordan（部分主元）解 A x = b；奇异返回 false。
bool gj_solve(std::vector<std::vector<cd>> A, std::vector<cd> b,
              std::vector<cd>& x, double scale) {
    const int n = (int)A.size();
    const double tiny = 1e-12 * std::max(scale, 1e-300);
    for (int col = 0; col < n; ++col) {
        int piv = col;
        double best = std::abs(A[col][col]);
        for (int r = col + 1; r < n; ++r) {
            double v = std::abs(A[r][col]);
            if (v > best) { best = v; piv = r; }
        }
        if (!(best > tiny)) return false;
        if (piv != col) {
            std::swap(A[piv], A[col]);
            std::swap(b[piv], b[col]);
        }
        cd d = A[col][col];
        for (int j = col; j < n; ++j) A[col][j] /= d;
        b[col] /= d;
        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            cd f = A[r][col];
            if (std::abs(f) == 0.0) continue;
            for (int j = col; j < n; ++j) A[r][j] -= f * A[col][j];
            b[r] -= f * b[col];
        }
    }
    x = b;
    return true;
}

inline bool finite_cd(const cd& z) {
    return std::isfinite(z.real()) && std::isfinite(z.imag());
}

}  // namespace

// ---------------------------------------------------------------------------
// Beamformer
// ---------------------------------------------------------------------------
Beamformer::Beamformer(int nch, int sr, const char* method, const char* array_type,
                       double geom, double c, int frame_size)
    : nch_(nch),
      sr_(sr),
      frame_(frame_size),
      hop_(frame_size / 2),
      method_(method ? method : "dsb"),
      doa_(nch, sr, geom, c, 1.0, 0.05, 4.0, array_type) {
    if (method_ != "dsb" && method_ != "mvdr") method_ = "dsb";
    // STFT 帧长参数化：frame_size <= 0 时按 ~21.3 ms（2 的幂）由采样率推导
    // （48k→1024、16k→512）；非法显式值回退旧默认 1024（48k 行为不变）。
    if (frame_ <= 0) {
        frame_ = 256;
        while (frame_ < (int)(0.02133 * sr_ + 0.5)) frame_ <<= 1;
        hop_ = frame_ / 2;
    } else if (frame_ < 16 || frame_ % 2 != 0) {
        frame_ = 1024;
        hop_ = 512;
    }
    // MVDR 统计窗/warm-up 时长参数化：与旧 48k 默认（200 帧 = 2.13 s、
    // 12 帧 = 128 ms @hop 512）等价——48k/1024 下算得 200/12，逐位一致；
    // 16k/512 下自动为 133/8。set_stat_frames/set_warmup_frames 仍可覆盖。
    {
        const double hop_s = (double)hop_ / (double)sr_;
        const int sf = (int)(2.133 / hop_s + 0.5);
        const int wf = (int)(0.128 / hop_s + 0.5);
        stat_frames_ = sf < 8 ? 8 : sf;
        warmup_ = wf < 0 ? 0 : wf;
    }
    // ULA 4 麦紧凑阵 MPDR 防自消除：绝对加载下限提高到 5%（UCA 维持 0.1%，
    // 数值与旧版一致）。见 hpp set_abs_load_frac 注释。
    abs_load_frac_ = doa_.is_ula() ? 0.05 : 0.001;
    theta_ = 0.0;
}

std::pair<std::vector<float>, float> Beamformer::process_file(
    const std::vector<std::vector<float>>& ch) {
    theta_ = doa_.estimate(ch);
    std::vector<float> y;
    if (method_ == "mvdr")
        y = mvdr_full_(ch, theta_);
    else
        y = dsb_full_(ch, theta_);
    return {std::move(y), (float)theta_};
}

std::vector<float> Beamformer::process_at(const std::vector<std::vector<float>>& ch,
                                          double theta_deg) {
    theta_ = theta_deg;
    if (method_ == "mvdr") return mvdr_full_(ch, theta_deg);
    return dsb_full_(ch, theta_deg);
}

// DSB：out(t) = mean_m x_m(t + tau_m)，窗 sinc 分数延迟对齐（全长，无分帧伪影）。
// frac_delay(x, d)=x(t-d)，取 d=-tau_m 即得 x_m(t+tau_m)。
std::vector<float> Beamformer::dsb_full_(const std::vector<std::vector<float>>& x,
                                         double theta_deg) {
    if ((int)x.size() < nch_) return {};
    size_t N = x[0].size();
    for (int m = 1; m < nch_; ++m) N = std::min(N, x[m].size());
    std::vector<double> tau;
    doa_.delays(theta_deg * M_PI / 180.0, tau);
    std::vector<float> out(N, 0.0f);
    for (int m = 0; m < nch_; ++m) {
        std::vector<float> sh = frac_delay(x[m], -tau[m], (double)sr_, 16);
        for (size_t i = 0; i < N; ++i) out[i] += sh[i];
    }
    float inv = 1.0f / (float)nch_;
    for (auto& v : out) v *= inv;
    return out;
}

// MVDR：两遍批处理。第一遍收集全部帧谱，用最近 stat_frames_ 帧做每频点 SMI
// 协方差（Hermitian 强制对称 + 对角加载 lambda_min + abs_load_frac_*mean_diag；
// abs_load_frac_：UCA=0.001，ULA=0.05 防导向误差自消除），第二遍
// 加权 WOLA 重建；前 warmup_ 帧用 DSB 权重（warm-up）。
std::vector<float> Beamformer::mvdr_full_(const std::vector<std::vector<float>>& x,
                                          double theta_deg) {
    if ((int)x.size() < nch_) return {};
    const size_t N = x[0].size();
    for (int m = 1; m < nch_; ++m) if (x[m].size() < N) return {};  // 长度须一致
    if (N < (size_t)frame_) return dsb_full_(x, theta_deg);

    const int nf = frame_, nh = hop_, nb = nf / 2 + 1;
    const FFT_BF fft(nf);
    std::vector<float> win(nf);
    for (int i = 0; i < nf; ++i)
        win[i] = (float)std::sqrt(0.5 - 0.5 * std::cos(2.0 * M_PI * i / nf));

    // 前后各 pad 一帧零
    const size_t total = N + 2 * (size_t)nf;
    std::vector<std::vector<float>> xp(nch_, std::vector<float>(total, 0.0f));
    for (int m = 0; m < nch_; ++m)
        std::copy(x[m].begin(), x[m].begin() + (std::ptrdiff_t)N, xp[m].begin() + nf);

    int nfr = 0;
    for (size_t s = 0; s + (size_t)nf <= total; s += (size_t)nh) ++nfr;
    if (nfr == 0) return dsb_full_(x, theta_deg);

    // ---- 第一遍：帧谱 ----
    std::vector<std::vector<std::vector<std::complex<float>>>> specs(  // [k][m][b]
        nfr, std::vector<std::vector<std::complex<float>>>(
                 nch_, std::vector<std::complex<float>>(nb)));
    {
        std::vector<std::complex<float>> buf(nf);
        int k = 0;
        for (size_t s = 0; s + (size_t)nf <= total; s += (size_t)nh, ++k) {
            for (int m = 0; m < nch_; ++m) {
                for (int i = 0; i < nf; ++i)
                    buf[i] = {xp[m][s + i] * win[i], 0.0f};
                fft.forward(buf);
                std::copy(buf.begin(), buf.begin() + nb, specs[k][m].begin());
            }
        }
    }

    // ---- 权重：DSB 权重 + MVDR 权重（最近 stat_frames_ 帧协方差） ----
    std::vector<double> tau;
    doa_.delays(theta_deg * M_PI / 180.0, tau);
    std::vector<std::vector<std::complex<float>>> Wdsb(
        nb, std::vector<std::complex<float>>(nch_));
    std::vector<std::vector<std::complex<float>>> W(
        nb, std::vector<std::complex<float>>(nch_));
    for (int b = 0; b < nb; ++b) {
        double f = (double)b * sr_ / nf;
        for (int m = 0; m < nch_; ++m) {
            double ph = 2.0 * M_PI * f * tau[m];
            Wdsb[b][m] = std::complex<float>(
                (float)(std::cos(ph) / nch_), (float)(std::sin(ph) / nch_));
        }
    }
    W = Wdsb;

    const int k0 = std::max(0, nfr - stat_frames_);  // 最近 stat_frames_ 帧
    const int K = nfr - k0;
    for (int b = 0; b < nb; ++b) {
        // SMI 协方差（double 累加）
        std::vector<std::vector<cd>> R(nch_, std::vector<cd>(nch_, {0.0, 0.0}));
        for (int k = k0; k < nfr; ++k) {
            std::vector<cd> X(nch_);
            for (int m = 0; m < nch_; ++m) X[m] = cd(specs[k][m][b]);
            for (int m = 0; m < nch_; ++m)
                for (int n2 = m; n2 < nch_; ++n2) {
                    cd v = X[m] * std::conj(X[n2]);
                    if (m == n2)
                        R[m][m] += v;
                    else
                        R[m][n2] += v;
                }
        }
        double invK = 1.0 / K;
        double scale = 0.0;
        for (int m = 0; m < nch_; ++m)
            for (int n2 = m; n2 < nch_; ++n2) {
                R[m][n2] *= invK;
                if (m != n2) R[n2][m] = std::conj(R[m][n2]);
                scale = std::max(scale, std::abs(R[m][n2]));
            }
        if (!(scale > 0.0) || !std::isfinite(scale)) continue;  // 静音频点：保持 DSB

        double dmean = 0.0;
        for (int m = 0; m < nch_; ++m) dmean += R[m][m].real();
        dmean = dmean / nch_ + 1e-30;
        double lmin = std::max(0.0, hermitian_min_eig(R, nch_, scale));
        double gamma = diag_load_ * lmin + abs_load_frac_ * dmean;

        std::vector<std::vector<cd>> A(nch_, std::vector<cd>(nch_));
        for (int m = 0; m < nch_; ++m)
            for (int n2 = 0; n2 < nch_; ++n2) {
                A[m][n2] = R[m][n2];
                if (m == n2) A[m][m] += cd(gamma, 0.0);
            }
        // 导向矢量 a[m] = exp(-j2pi f tau_m)
        double f = (double)b * sr_ / nf;
        std::vector<cd> a(nch_);
        for (int m = 0; m < nch_; ++m) {
            double ph = -2.0 * M_PI * f * tau[m];
            a[m] = cd(std::cos(ph), std::sin(ph));
        }
        std::vector<cd> u;
        if (!gj_solve(A, a, u, scale)) continue;  // 奇异：回退 DSB
        cd denom{0.0, 0.0};
        bool ok = true;
        for (int m = 0; m < nch_ && ok; ++m) {
            denom += std::conj(a[m]) * u[m];
            ok = ok && finite_cd(u[m]);
        }
        if (!ok || !finite_cd(denom) || std::abs(denom) < 1e-12) continue;
        for (int m = 0; m < nch_; ++m) {
            cd wm = u[m] / denom;
            W[b][m] = std::complex<float>((float)wm.real(), (float)(-wm.imag()));
        }
    }

    // ---- 第二遍：加权 + WOLA 重建；前 warmup_ 帧用 DSB 权重 ----
    std::vector<float> out(total, 0.0f);
    {
        std::vector<std::complex<float>> Y(nf);
        int k = 0;
        for (size_t s = 0; s + (size_t)nf <= total; s += (size_t)nh, ++k) {
            const auto& Wk = (k < warmup_) ? Wdsb : W;
            for (int i = 0; i < nf; ++i) Y[i] = {0.f, 0.f};
            for (int m = 0; m < nch_; ++m) {
                const auto& Xk = specs[k][m];
                for (int b = 0; b < nb; ++b) {
                    std::complex<float> v = Wk[b][m] * Xk[b];
                    Y[b] += v;
                    if (b > 0 && b < nf - b) Y[nf - b] += std::conj(v);
                }
            }
            // Nyquist bin 只出现一次
            fft.inverse(Y);
            for (int i = 0; i < nf; ++i) out[s + i] += Y[i].real() * win[i];
        }
    }
    return std::vector<float>(out.begin() + nf, out.begin() + (std::ptrdiff_t)(nf + N));
}

}  // namespace afe
