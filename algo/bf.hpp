// voice/algo/bf.hpp — DOA + 波束成形（DSB / MVDR），支持两种阵型：
// （自 afecpp/bf.hpp 拷贝改造：GCC 分块长 / MVDR STFT 帧长与统计窗均按
//   采样率参数化，48k 默认参数下与原版数值一致；几何与算法不变）
//   UCA（6 麦均匀圆阵，小米 LX06）与 ULA（4 麦均匀线阵，间距 40 mm）。
//
// 几何（远场平面波、仅方位角，声速 c=343）：
//   UCA: 第 m 麦方位角 phi_m = 2*pi*m/nch，半径 r（默认 0.035 m），
//        tau_m(theta) = r/c * cos(theta - phi_m)，theta ∈ [0,360)。
//   ULA: 第 i 麦位置 x_i = (i - (nch-1)/2)*d（中心对称，d=相邻间距，默认 0.04 m），
//        broadside 角 theta ∈ [-90,+90]（0°=正对阵列法向），
//        tau_i(theta) = x_i * sin(theta) / c。
//        线阵固有前后模糊（sin(theta) = sin(180°-theta)），DOA 只搜单侧
//        [-90,+90]，结果即 ±theta 等价类代表。
//
// DOA        : 全 C(nch,2) 麦对 GCC-PHAT 互相关 + 导向网格搜索（1° 步进 + 细化），
//              UCA 搜 0-360°，ULA 搜 -90..+90°。
//              GCC 限带 <= c/(2*D)（D=最大孔径：UCA 2r ≈ 2.45 kHz @ r=0.035；
//              ULA (nch-1)d ≈ 1.43 kHz @ d=0.04, nch=4）防空间混叠。
// Beamformer : DSB（按 DOA 分数延迟对齐后求平均，窗sinc 插值）
//              MVDR（STFT 1024/50% overlap，最近 200 帧 SMI 协方差 +
//              对角加载 lambda_min + 0.1%*mean_diag，nch x nch 复数 Gauss-Jordan
//              手写求解，逐 bin 权重，warm-up 帧退化为 DSB）。
//
// 接口与 afe/bf.py（Python 参考）对齐；只做数值处理，不播放任何音频。

#pragma once
#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "fft_bf.hpp"

namespace afe {

// 窗 sinc 分数延迟：out(t) = x(t + d_sec)（半宽度 half 个样点的 Hann 窗 sinc 核，
// 越界补零）。用于 DSB 对齐与测试场景合成。
std::vector<float> frac_delay(const std::vector<float>& x, double d_sec, double sr,
                              int half = 16);

class DOA {
public:
    // array_type: "uca"（geom=半径 m，theta ∈ [0,360)）| "ula"（geom=相邻间距 m，
    // theta ∈ [-90,+90]，broadside 角）。追加在参数表末尾，旧位置调用不受影响。
    DOA(int nch = 6, int sr = 48000, double radius = 0.035, double c = 343.0,
        double grid_step_deg = 1.0, double refine_step_deg = 0.05,
        double max_dur_s = 4.0, const char* array_type = "uca");

    // 估计方位角（度）。UCA ∈ [0,360)，ULA ∈ [-90,+90]（±theta 等价类代表）。
    // x: ch[c][n]。静音/退化输入返回 0。
    double estimate(const std::vector<std::vector<float>>& x);

    // ---- 流式增量 GCC（快速路径）-----------------------------------------
    // estimate() 每次对整个分析窗全量重做 nch×nblocks 个 B 点 FFT；流式场景
    // 相邻两次估计窗口高度重叠（@16k 窗 4s、节拍 0.5s，重叠 87%）。增量接口
    // 把每个完整块的通道谱缓存进槽环，估计时只重求和 + 搜索——每 0.5s 新
    // 数据仅需 nch 个 FFT。求和每次从槽全量重建（无增量漂移）；满块对齐窗
    // 下与 estimate() 逐位一致（同分块、同均值去除、同累加顺序）。分析窗 =
    // 最近 filled 个完整块（稳态 @16k 为 8×8192 ≈ 4.1 s，尾块不足部分不计）。
    // 返回值：inc_push 为本次新完成的块数；inc_estimate 无整块时返回 0。
    int inc_push(const float* const* x, int n);
    double inc_estimate();
    void inc_reset();
    long long inc_total_samples() const { return inc_.total; }

    // tau_m(theta_rad)，秒，相对阵中心
    void delays(double theta_rad, std::vector<double>& tau) const;

    int nch() const { return nch_; }
    double sr() const { return sr_; }
    double radius() const { return geom_; }  // 兼容旧名：返回 geom
    double geom() const { return geom_; }    // ula=间距 m | uca=半径 m
    double c() const { return c_; }
    double fmax() const { return fmax_; }  // GCC 带宽上限 c/(2*孔径)
    int gcc_block() const { return gcc_block_; }  // GCC 分块 FFT 长（参数化）
    bool is_ula() const { return ula_; }
    const char* array_type() const { return ula_ ? "ula" : "uca"; }
    double last_doa_deg() const { return last_; }

private:
    // ULA 专用：块平均互谱 acc[p][b]（B 点 FFT）上的精确分数 lag 导向网格搜索。
    // 整数 lag 采样 + 线性插值会把宽相关峰（fmax 低）钉到整数 lag，产生
    // 2-6° 系统偏差；此处在每个网格角用旋转相位直接求分数 lag 处的互相关。
    // 热循环（网格×麦对×频带 bin 的相位递推求和）用 float——A7 无 double
    // NEON，float 相位递推 731 步幅值漂移 ~1e-5、每 64 bin 重归一，对 0.05°
    // 细化网格的 argmax 无影响（白化与几何仍在 double）。
    double ula_search_(const std::vector<std::vector<std::complex<float>>>& acc, int B);

    // 块平均互谱 → 缩放 → ULA/UCA 搜索（estimate 与 inc_estimate 共用）。
    double search_from_acc_(std::vector<std::vector<std::complex<float>>>& acc, int B,
                            double inv_blocks);

    // 增量 GCC 状态（inc_* 用；槽环缓存每块通道 Hermitian 半谱）
    struct Inc {
        int B = 0, nb2 = 0, nslots = 0;
        int staged = 0;                 // 当前暂存块已收样本数（< B）
        int wpos = 0, filled = 0;       // 下一写入槽 / 已填槽数（满后最旧槽 = wpos）
        long long total = 0;            // 累计接收样本数
        std::vector<float> stage;                        // nch × B 平铺暂存
        std::vector<std::complex<float>> slot;           // nslots × nch × nb2 平铺半谱
        std::vector<std::complex<float>> zbuf;           // 块 FFT scratch（B）
        std::vector<std::vector<std::complex<float>>> acc;  // 估计求和 scratch
    };
    Inc inc_;
    std::unique_ptr<FFT_BF> inc_fft_;

    int nch_;
    double sr_, geom_, c_;   // geom_: uca=半径 | ula=相邻间距
    bool ula_ = false;
    double aperture_;        // 最大孔径 D：uca=2r | ula=(nch-1)*d
    double fmax_;            // c/(2*aperture_)，GCC 防混叠限带
    int gcc_block_;          // GCC 分块 FFT 长：next_pow2(0.34s·sr)——采样率参数化
    std::vector<double> phi_;  // UCA 麦克风方位角
    std::vector<double> pos_;  // ULA 麦克风 x 坐标（中心对称）
    std::vector<std::pair<int, int>> pairs_;
    double grid_step_, refine_step_;
    int max_samples_;
    double last_ = 0.0;
};

class Beamformer {
public:
    // method: "dsb" | "mvdr"；array_type: "uca"（默认，geom=半径）| "ula"（geom=间距）。
    // 旧调用 Beamformer(nch, sr, method) 行为不变（uca / 0.035）。
    Beamformer(int nch = 6, int sr = 48000, const char* method = "dsb",
               const char* array_type = "uca", double geom = 0.035,
               double c = 343.0, int frame_size = 1024);

    // 整段批处理：内部先估 DOA，再按 method 波束成形。返回 (mono, doa_deg)。
    std::pair<std::vector<float>, float> process_file(
        const std::vector<std::vector<float>>& ch);

    // 用给定导向角处理（例如把干净参考信号过同一波束，消除时域错位）。
    std::vector<float> process_at(const std::vector<std::vector<float>>& ch,
                                  double theta_deg);

    float get_doa() const { return (float)theta_; }
    const std::string& method() const { return method_; }

    // MVDR 参数（默认按规格：200 帧统计、12 帧 warm-up 用 DSB、对角加载 1.0）
    void set_stat_frames(int n) { stat_frames_ = n < 8 ? 8 : n; }
    void set_warmup_frames(int n) { warmup_ = n < 0 ? 0 : n; }
    void set_diag_load(double d) { diag_load_ = d; }
    // 绝对加载下限系数 gamma += abs_load_frac_ * mean_diag（MPDR 防自消除）。
    // ULA 默认 0.05：4 麦紧凑阵对导向误差更敏感（麦数少 + 高频孔径大，
    // ±0.2° 误差即可自消除 ~5 dB），5% 下限把容差扩到约 ±0.45°；
    // UCA 维持历史值 0.001（6 麦冗余度高，行为与旧版完全一致）。
    void set_abs_load_frac(double f) { abs_load_frac_ = f > 0.0 ? f : 0.001; }

private:
    std::vector<float> dsb_full_(const std::vector<std::vector<float>>& x,
                                 double theta_deg);
    std::vector<float> mvdr_full_(const std::vector<std::vector<float>>& x,
                                  double theta_deg);

    int nch_, sr_, frame_, hop_;
    std::string method_;
    DOA doa_;
    double theta_ = 0.0;
    // MVDR
    int stat_frames_ = 200;   // 最近 N 帧协方差
    int warmup_ = 12;         // 前 N 帧（及统计不足时）退化为 DSB
    double diag_load_ = 1.0;  // 加载系数：gamma = 1.0*lambda_min + abs_load_frac_*mean_diag
    double abs_load_frac_ = 0.001;  // 绝对加载下限（ULA 构造时自动取 0.05）
};

}  // namespace afe
