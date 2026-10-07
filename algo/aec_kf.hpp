// voice/algo/aec_kf.hpp — PBFDKF：分块频域卡尔曼声学回声消除
// （自 afecpp/aec_kf.hpp 同步：含流式预览错位修复与 FIXES_AEC 四项修复；
//   帧长/块数经构造参数由 voice_algo 包装按采样率选择）
//
// 算法：Enzner & Vary 2006 频域卡尔曼 AEC 的分块（partitioned-block）扩展，
// 骨架与 MDF 相同的 overlap-save 块频域结构：
//   - 块长 B（默认 1024 @48k ≈ 21.3ms；16k 由 voice_algo 包装传入 512 = 32ms），
//     FFT 长 M = 2B（自写 radix-2）
//   - 滤波器分 K 块（默认 16 @48k 覆盖 341ms；16k 传 10 块覆盖 320ms）
//   - 每个 FFT bin 上把（该 bin 的）回声路径系数建模为复随机游走状态，
//     K 个分块的系数共用一个观测（残差谱），逐 bin 卡尔曼递推：
//         预测  P = P + Q                     （过程噪声 = 路径变化率）
//         增益  S   = R + Σ_k |X_k|²·P_k       （X_k = 延迟 k 块的参考谱）
//               G_k = P_k·X_k^H / S
//         更新  W_k += G_k·E，  P_k = (1 - G_k·X_k)·P_k
//
// 协方差采用"标量 P̄ × 分块份额 g_k"的比例化（proportionate）结构：
//   P_k[m] = P̄[m]·g_k[m]，g_k 由各分块 W 能量分布 |W_k[m]|²/Σ_j|W_j[m]|² 的
//   EMA 学习（混入均匀份额保证再跟踪）。纯对称分块会把每块修正的 (K-1)/K
//   撒到不含路径的分块上（收敛慢 K 倍）；比例化后增益自动集中到路径所在的
//   分块，回声路径稀疏时（真实房间常见）收敛接近单块滤波器速度。
//
// 测量噪声 R 的在线估计是核心自适应机制（替代 NLMS+Geigel 双端开关）：
//   R ≈ 残差功率谱非对称 EMA（快升 ~2 块/慢降 ~4 块，5-bin 谱平滑）
//     = 近端语音功率 + 背景噪声功率 + 残余失配
//   双端/强噪声时 R 升高 → 卡尔曼增益自动收缩 → 滤波器天然"让路"，
//   无需任何双端检测开关。
//
// 增益调度（三重）：
//   1) Q = c_q·R/Σ|X_k|²：自尺度过程噪声（稳态等效 NLMS 步长 ∝ c_q）；
//   2) 新息一致性膨胀：平滑残差显著超过预测新息方差 S（谱突变/回声再
//      出现/路径变化）时按比例放大 P̄（一块内恢复误差量级）；
//   3) 单 bin 总增益上限 μmax（NLMS μ=0.5 稳定界），膨胀只改方向分配。
//   发散保护：残差远超 mic+ref 功率时复位（极端异常，正常不触发）。
//
// 双路径（two-path，Speex 同款思想）：背景滤波器 = 上面的卡尔曼自适应
// 滤波器；前景滤波器 W_fg 只用于输出，当背景残差能量（EMA 窗口比较）明显
// 优于前景时整体复制过去。双端通话期间背景即便被近端语音污染也不会污染
// 输出（近端分量在 E_bg/E_f 比较中抵消，比较的是残余回声+失配噪声）；
// 近端静音间隙背景以单讲速度再收敛并完成复制。卡尔曼的 R 增益调节决定
// 背景在双端下的收敛/污染速度，双路径保证输出始终用"历史最优"滤波器。
//
// 流式接口：process() 支持任意长度 n，内部按块缓冲累积；块尾不足 B 的部分用
// 零填充做一次"预览"滤波并回滚自适应状态后输出——保证每次调用返回恰 n 个
// 样本且不污染状态。process_file() 整段处理，仅末尾不足一块时走该路径。

#pragma once
#include <complex>
#include <vector>

// FFTPlan：迭代 radix-2（自 PBFDKF 私有嵌套提升为文件级——RefEngine 亦需
// 持有同款变换；纯结构移动，数值行为不变）。每级独立连续 twiddle 表；
// inverse 为编译期模板参数消除内层分支；逆变换含 1/n 缩放。
struct AecFFTPlan {
    int n = 0;
    std::vector<int> rev;                    // 位反转置换表
    std::vector<int> stage_half;             // 每级半长
    std::vector<float> twr, twi;             // 平铺每级 twiddle（正变换）
    std::vector<float> twr_inv, twi_inv;     // 逆变换（共轭）
    void init(int n);
    void run(std::complex<float>* a, bool inverse) const;  // 就地变换
};

class PBFDKF;

// RefEngine —— 参考谱服务（Hermitian 半谱，16k 引擎路径专用）。
// 多通道 AEC 共享同一远端参考时，x_time 滑窗→FFT→K 槽半谱环 + 每槽 |X|² 表
// 只算一次，各实例读同一份（替代每实例各一遍 FFT+谱环）。槽值/|X|² 表与
// PBFDKF 独立（非共享）模式逐位一致：同一 FFTPlan、同一 nx 表达式、
// stot() 按 k 升序累加与独立模式 pass1 内联累加同序。
class RefEngine {
public:
    void init(int B, int K);
    void push(const float* ref);   // 恰 B 样本（调用方保证块对齐）
    int xpos() const { return xpos_; }
    int nb() const { return nb_; }
    const std::complex<float>* X(int slot) const {
        return &Xring_[(size_t)slot * nb_];
    }
    const float* nx(int slot) const { return &nxring_[(size_t)slot * nb_]; }
    // Σ_k nx_k[m]（k=0 最新 → K-1 最旧，与独立模式累加序一致）→ out[nb]
    void stot(float* out) const;
    // 快照直接用值拷贝（成员皆 vector/int，预览回滚 = own 实例整体赋值）

private:
    int B_ = 0, M_ = 0, nb_ = 0, K_ = 0, xpos_ = 0;
    AecFFTPlan fft_;
    std::vector<float> x_time_;
    std::vector<std::complex<float>> Xring_, fftbuf_;
    std::vector<float> nxring_;
};

class PBFDKF {
public:
    // half=true 启用 Hermitian 半谱快速路径：实输入谱共轭对称（X[M-m]=conj(X[m])，
    // W 的卡尔曼更新项为 Hermitian 谱乘积、对称性天然保持，投影步骤定期再强制），
    // K×M 热循环（回声估计/卡尔曼更新/g 学习/前景混合）与状态存储只算/只存
    // bins [0, M/2]，计算量近乎减半。48k 旧参数路径（half=false）逐位不变。
    // half 模式下参考谱由 RefEngine 承载：独立实例用内置 own_ref_，多通道
    // 共享参考时 attach_shared() 挂外部服务（x FFT 全通道只算一次）。
    PBFDKF(int frame = 1024, int blocks = 16, int sr = 48000, bool half = false);

    // 挂接共享参考谱服务（须 half 模式；push 由调用方在每块处理前完成）。
    // 挂接后只允许 process_shared()（块对齐），process() 交由 C 层拒绝。
    void attach_shared(RefEngine* re);
    bool shared_attached() const { return shared_ != nullptr; }

    // 共享模式流式：n 须为 B 的整数倍（无预览路径）。ref 仍须传入（时域能量
    // 统计用；谱已由 RefEngine 承载，不做 FFT）
    void process_shared(const float* mic, const float* ref, float* out, int n);

    // 流式处理任意长度：mic/ref 各 n 样本 → 残差 out（恰 n 样本）
    void process(const float* mic, const float* ref, float* out, int n);

    // 整段处理（mic/ref 等长）
    std::vector<float> process_file(const std::vector<float>& mic,
                                    const std::vector<float>& ref);

    struct Stats {
        long blocks = 0;     // 已处理块数
        long q_boosts = 0;   // 新息一致性膨胀触发块数（诊断）
        long resets = 0;     // 发散保护复位次数
        long fg_copies = 0;  // 前景复制次数
    };
    const Stats& stats() const { return stats_; }

    int frame_size() const { return B_; }
    int n_blocks() const { return K_; }
    int fft_size() const { return M_; }

private:
    void process_block_(const float* mic, const float* ref, float* out);
    void run_tail_(const float* mic, const float* ref, int r, float* out);
    // 半谱辅助：全谱标量和（herm 下镜像对计双，与全 M bin 和语义一致）
    double sum_bins_(const float* x) const;
    // 半谱辅助：Hermitian 半谱 → 全谱（镜像共轭填充 fftbuf_ 尾部）
    void herm_expand_(const std::vector<std::complex<float>>& half);

    int B_, M_, K_;
    int sr_ = 0;   // 采样率（记录用；块长/块数已确定全部数学）
    bool herm_ = false;   // Hermitian 半谱快速路径
    int nb_ = 0;          // 每 bin 循环上界：herm ? M/2+1 : M
    AecFFTPlan fft_;
    Stats stats_;

    // ---- 自适应状态（herm 下所有 per-bin 行只存 nb_ = M/2+1 个 bin）----
    std::vector<float> x_time_;                         // 远端时域缓冲 [旧B | 新B]（legacy 路径用）
    RefEngine own_ref_;                                 // herm 路径参考谱（共享时挂外部）
    RefEngine* shared_ = nullptr;                       // 共享参考谱服务（attached 时）
    std::vector<std::vector<std::complex<float>>> X_;   // 参考谱环形缓冲 K × nb_（legacy 路径用）
    std::vector<std::vector<std::complex<float>>> W_;   // 背景子滤波器谱 K × nb_
    std::vector<std::vector<std::complex<float>>> Wfg_; // 前景子滤波器谱 K × nb_（输出用）
    double Ebg_ = 0.0, Efg_ = 0.0;                      // 背景/前景残差能量 EMA
    int copy_hold_ = 0;                                 // 复制冷却计数
    long fg_copies_ = 0;                                // 前景复制次数（诊断）
    std::vector<float> Pbar_;                           // 每 bin 标量协方差 P̄
    std::vector<float> gprofile_, gcapk_;              // 分块先验 w̄_k 与 g 上限
    std::vector<std::vector<float>> G_;                 // 分块份额 g_k（Σ_k g_k = 1）
    std::vector<float> R_;                              // 测量噪声 PSD（残差 EMA）
    std::vector<float> Stot_sm_;                        // 平滑 Σ_k|X_k|²（Q 的分母）
    std::vector<char> inited_;                          // 每 bin P/R 初始化标志
    int xpos_ = 0;                                      // X_ 环形缓冲最新槽位
    int act_blocks_ = 0;                                // 激励预热计数（P 初始化门）

    // ---- 流式队列 ----
    std::vector<float> pend_mic_, pend_ref_, outq_;
    size_t pend_off_ = 0, out_head_ = 0;
    // pend 缓冲 [pend_off_, ...) 中已以"预览"形式发出的样本数（< B_）。
    // 后续整块覆盖这些位置时，块输出的对应头部直接丢弃，保证每个输入样本
    // 恰好输出一次（修复多次 process() 调用时预览样本被整块输出重复追加
    // 导致的流错位——lag≈129，见 FIXES_AEC 缺陷 4）。
    int previewed_ = 0;

    // ---- 后置滤波 / 欠消检测（FIXES_AEC）----
    std::vector<float> CexR_, CexI_;          // E·X0^* 互谱 EMA 实/虚部（无偏）
    std::vector<float> Cee_, Cxx_;            // 残差/参考 PSD EMA
    std::vector<float> Cbin_;                 // 每 bin 相干占比 c[m]
    std::vector<float> Gpost_, GpostT_, Gsm_; // 后置增益（原始/时间平滑/谱平滑 scratch）
    std::vector<float> GpostPrev_;            // 预览输出用冻结增益（流式一致性）
    bool preview_mode_ = false;               // run_tail_ 预览中（输出用冻结增益）
    bool pf_idle_prev_ = true;                // 预览门控冻结值
    std::vector<std::complex<float>> Efg_spec_;  // 前景残差谱（后置滤波用）
    double cfrac_sm_ = 0.0;                   // 全局相干分数（平滑）
    bool pf_on_ = false;                      // 后置滤波门（回声主导）
    int pf_hold_ = 0;                         // 后置滤波开去抖计数
    int sil_hold_ = 0;                        // 参考连续静音块计数（门控 hold）
    bool pf_idle_ = true;                       // 门关且增益已回 1（跳过后置 FFT）
    int det_hold_ = 0;                        // 欠消检测持续计数
    bool track_mode_ = false;                 // 跟踪模式（Q 自适应）
    int post_cnt_ = 0;                        // 后置滤波预热计数

    // ---- 证据自适应晚期块上限 ----
    std::vector<double> wshare_ema_;           // 每块 W 能量份额 EMA（证据自适应上限）
    std::vector<float> wsum_f_;                // 每块 W 能量（float 热循环累积）
    std::vector<float> gcap_use_;              // 当前生效的 g 上限

    // ---- scratch（避免每块堆分配）----
    std::vector<std::complex<float>> fftbuf_, Y_, E_, Yfg_;
    std::vector<float> e_blk_, efg_blk_, stot_, pef_, stf_, qv_, Sm_, invSm_, pfl_, pcp_;
    std::vector<float> gex_, g2ex_, gnew_, invg_;
    std::vector<float> tmic_, tref_, tout_;
};
