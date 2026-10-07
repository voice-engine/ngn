// main_engine.cpp — voice-engine daemon for NanoPi Air (4-mic AC108 array)
//
// 架构（详见 ENGINE_REPORT.md）：
//   线程1 采集:  [--capture-fs 16k（默认）] hw:1,0 32k S32 2ch（线上流）
//                → tag 解码 + float 转换一步完成 → 16k 4ch float → capture 环
//                （ADC 直接 16k，无软件抽取；配方：~/ac108/mic4-record16.sh cfg）
//                [--capture-fs 48k（旧路径）] hw:1,0 96k 线上流 → tag 解码 4ch@48k
//                → ÷3 多相抽取(48 抽头 FIR) → 16k 4ch float（配方：mic4-record.sh）
//                [--simulate-capture 时改为从 4ch16k WAV 循环、按实时节奏读入]
//   线程2 管线:  逐 512 样本块：va_aec_ref_push(共享参考谱，一次 FFT) →
//                4×va_aec_process_shared → va_doa_bf（2026-10-06 单核化：
//                AEC worker 已撤，算法 rtf ~0.5 后双核并行收益不再）
//                → 16k mono S16 → [--ipc fifo(旧)] /tmp/voice_mic.fifo（O_NONBLOCK 丢弃计数）
//                              → [--ipc shm(默认)] /dev/shm/voice_mic 数据环（shm_voice.h：
//                写者覆盖、多读者各自游标+心跳、epoch 重启、per-reader wake fifo）
//   线程3 回采:  [--ipc fifo] /tmp/voice_spk.fifo（O_RDWR 自持 + poll，50ms 兜底；
//                短命写者数据不丢，断流由 ref 环断供表达）
//                → 48k mono → ①÷3 重采样 → 16k ref 环（供线程2 AEC）
//                             ②speaker 输出（默认 --null-playback 丢弃；
//                               --device 打开/写失败自动回退 null 并告警一次）
//              [--ipc shm(默认)] /dev/shm/voice_spk 段收割混音线程：poll 写者唤醒
//                fifo（20ms 兜底）→ 收割活跃写者槽各自新增数据（落后超环长则跳最新）
//                → float 求和混音 + 限幅 → 同上 ①②；--dump-mix 可把混音落盘 wav。
//
// FIFO 半开语义（--ipc fifo）：voice_mic.fifo 无读者 → 写得 EPIPE/EAGAIN → 忽略
// SIGPIPE、非阻塞、丢弃并计数，读者出现自动恢复；voice_spk.fifo 引擎侧常持
// O_RDWR|O_NONBLOCK 自持 fd（永不半开）+ poll 收割（50ms 兜底）——短命写者
// close 后数据仍滞留管道可读，消除旧"EOF 轮询 50ms 盲窗丢数据"竞态；写者断流
// 由 ref 环断供 → 管线 DRAINED 表达（ref 同步状态机 HOLD 超时丢弃滞留样本防
// DRAINED↔HOLD 乒乓活锁，详见 pipeline_thread 注释）。
// SHM 语义（--ipc shm）：无 EOF/半开概念——读者/写者注册表 + 心跳超时回收 +
// epoch（引擎重启++，客户端自动重注册），详见 shm_voice.h 与 SHM_IPC_REPORT.md。
//
// Build: 板上 gcc；/home/i/voice/algo/libvoice_algo.so 缺席时 Makefile 加
//        -DVA_MOCK 链接内置桩（mock AEC: out=mic；mock BF: out=ch0）。

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <string>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>   // nanosleep（aec worker 空闲微睡）
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#include <alsa/asoundlib.h>

#include "shm_voice.h"

#ifdef VA_MOCK
#include "va_mock.h"
#define VA_ALGO_DESC "mock-stubs"
#else
#include <voice_algo.h>
#define VA_ALGO_DESC "libvoice_algo"
#endif

// ----------------------------------------------------------------------------
// 常量与配置
// ----------------------------------------------------------------------------
static constexpr int   kRateWire48 = 96000;  // 48k 模式线上 I2S 速率（2x48k 打包）
static constexpr int   kRateWire16 = 32000;  // 16k 模式线上 I2S 速率（2x16k 打包）
static constexpr int   kRateMic    = 48000;  // tag 解码后每通道速率（48k 模式）
static constexpr int   kRateEng    = 16000;  // 引擎内部工作速率
static constexpr int   kNChan      = 4;      // 麦克风通道数
static constexpr int   kBlock      = 512;    // 管线块长（32ms @16k）。实测 160(10ms) 时 RTF 0.75→2.4（AEC 每调用固定开销×3.2，且热降频螺旋）——算法帧被 PBFDKF 内部 512 帧结构钉死；延迟优化放在采集/播放两侧
static constexpr float kMicSpacing = 0.04f;  // ULA 麦间距 40mm
static constexpr size_t kSpkChunk  = 4800;   // 回采读块（100ms @48k）

static std::atomic<bool> g_stop{false};
// 采集线程写、main 退出时读——原为裸 bool（跨线程无同步），改 atomic 消除数据竞争 UB
static std::atomic<bool> g_capture_failed{false};

enum class IpcMode { SHM, FIFO };

struct Config {
    bool        simulate       = false;
    const char* sim_wav        = nullptr;
    const char* capture_dev    = "hw:1,0";
    int         capture_fs     = 16000;            // ADC 采样率：16000（默认，直采）或 48000（旧路径）
    const char* playback_dev   = nullptr;          // null → 丢弃（--null-playback 默认）
    IpcMode     ipc            = IpcMode::SHM;     // 默认 SHM 环（--ipc fifo 走旧 fifo 路径）
    const char* mic_fifo_path  = "/tmp/voice_mic.fifo"; // 引擎写：处理后 16k S16 mono
    const char* spk_fifo_path  = "/tmp/voice_spk.fifo"; // 引擎读：回采 48k S16 mono
    const char* shm_mic_path   = "/dev/shm/voice_mic";  // shm 模式段路径（shm_voice.h 协议）
    const char* shm_spk_path   = "/dev/shm/voice_spk";
    const char* dump_mix       = nullptr;          // --dump-mix：回采/混音结果（48k mono S16）落盘 wav
    int         stats_interval = 5;                // 秒；<=0 关闭
    // 位置k←通道ch 映射（DOA/BF 导向用）。自然序 {0,1,2,3}；本板实测 {0,3,2,1}
    // （analyze_geometry.py 三方法互验 2026-10-06）。--mic-order 0,3,2,1 覆盖。
    int         mic_order[kNChan] = {0, 1, 2, 3};
};

// ----------------------------------------------------------------------------
// 统计计数器（各线程写，stats 打印读；relaxed 足够）
// ----------------------------------------------------------------------------
struct Stats {
    std::atomic<uint64_t> captured_samples{0}; // 16k 通道-样本数（帧数×4）
    std::atomic<uint64_t> aec_blocks{0};
    std::atomic<uint64_t> fifo_written{0};     // 16k 样本
    std::atomic<uint64_t> fifo_dropped{0};     // 16k 样本
    std::atomic<uint64_t> fifo_read{0};        // 48k 样本（voice_spk.fifo）
    std::atomic<uint64_t> ref_samples{0};      // 16k ref 样本（入 ref 环）
    std::atomic<uint64_t> tag_counts[4]{};
    std::atomic<uint64_t> wire_words{0};
    std::atomic<uint64_t> cap_reads{0};         // readi 调用数（诊断）
    std::atomic<int64_t>  cap_min_n{1 << 30};   // 最小返回帧数（诊断）
    std::atomic<uint64_t> overrun{0};          // ALSA capture xrun
    std::atomic<uint64_t> ring_overflow{0};    // capture 环溢出丢帧（帧）
    std::atomic<uint64_t> ref_overflow{0};     // ref 环溢出丢样本
    std::atomic<uint64_t> ref_padded{0};       // ref 环断供时补静音的样本数（联调诊断）
    std::atomic<uint64_t> ref_discarded{0};    // HOLD 超时丢弃的滞留 ref 样本（防活锁修复）
    std::atomic<uint64_t> pb_errors{0};        // 播放写错误
    std::atomic<uint64_t> shm_mic_written{0};  // shm 模式：mic 环写入 16k 样本
    std::atomic<uint64_t> mixed_frames{0};     // shm 模式：spk 混音输出 48k 帧
    std::atomic<int>      mic_readers{0};      // shm 模式：当前活跃读者数（注册表视角）
    std::atomic<int>      spk_writers{0};      // shm 模式：当前活跃写者数（注册表视角）
    std::atomic<uint64_t> shm_dropped{0};      // shm 模式：spk 收割追赶丢弃帧（全部槽合计）
    std::atomic<float>    spk_path_rtf{-1.0f}; // 回采路径（fifo 读 或 shm 收割混音）忙时/音频时长
    std::atomic<float>    mic_in_ms{-1.0f};    // capture 入环 → 管线消费的环滞留 EMA（ALSA 驱动缓冲另计）
    std::atomic<float>    doa_latest{-1.0f};
};
static Stats g_stats;


static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// capture → 管线滞留观测：capture 每 push 记 (全局帧号, 时刻)；管线 pop 后按
// 累计消费帧号查表差值。64 条环形（capture 5ms period → 覆盖 320ms > 环深）。
// 诊断用途，跨线程无锁：读侧按写序号取完整条目并容忍个别撕裂（校验 frame 单调）。
struct CapStamp { uint64_t frame; double t; };
static CapStamp g_cap_stamps[64];
static std::atomic<uint64_t> g_cap_stamp_wr{0};
static uint64_t g_pipe_consumed = 0;   // 管线累计消费帧数（仅管线线程写）
static void cap_stamp(uint64_t total_frames) {
    uint64_t i = g_cap_stamp_wr.fetch_add(1, std::memory_order_relaxed);
    g_cap_stamps[i % 64] = {total_frames, now_s()};
}
static double cap_lag_ms(uint64_t consumed) {   // 消费尾帧的入环滞留（ms；查不到返回 -1）
    uint64_t wr = g_cap_stamp_wr.load(std::memory_order_relaxed);
    for (int k = 0; k < 64; k++) {
        uint64_t i = wr - 1 - (uint64_t)k;
        const CapStamp& cs = g_cap_stamps[i % 64];
        if (cs.frame <= consumed && consumed - cs.frame < 16000)   // ≤1s 内的最近打点
            return (now_s() - cs.t) * 1000.0;
    }
    return -1.0;
}



static uint64_t now_ms() {
    using namespace std::chrono;
    return (uint64_t)duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

// 进程存活探测（回收注册表槽用）：ESRCH=不存在；EPERM=存在但非本用户
static bool pid_alive(pid_t pid) {
    if (pid <= 0) return true; // 未知 pid：保守视为存活，交给心跳超时
    return kill(pid, 0) == 0 || errno != ESRCH;
}

// 线程绑核（联调：rtf 在自适应/系统抖动期一度 >1；4×A7 独占核后恢复余量）。
// 失败（核数不足等）仅告警一次，不影响运行。
static void pin_cpu(int n, const char* who) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(n, &s);
    if (pthread_setaffinity_np(pthread_self(), sizeof(s), &s) != 0)
        fprintf(stderr, "[engine] %s 绑核 cpu%d 失败: %s（忽略）\n", who, n, strerror(errno));
}

// ----------------------------------------------------------------------------
// SPSC float 环（单生产者/单消费者，容量 2 的幂；溢出丢新数据）
// ----------------------------------------------------------------------------
class FloatRing {
public:
    explicit FloatRing(size_t cap_floats) {
        size_t p = 1;
        while (p < cap_floats) p <<= 1;
        cap_ = p;
        buf_.resize(p);
    }
    size_t writable() const {
        return cap_ - (tail_.load(std::memory_order_relaxed) - head_.load(std::memory_order_acquire));
    }
    size_t readable() const {
        return tail_.load(std::memory_order_acquire) - head_.load(std::memory_order_relaxed);
    }
    // 调用方保证 n <= writable()；返回写入数
    size_t push(const float* src, size_t n) {
        size_t t = tail_.load(std::memory_order_relaxed);
        for (size_t i = 0; i < n; i++) buf_[(t + i) & (cap_ - 1)] = src[i];
        tail_.store(t + n, std::memory_order_release);
        return n;
    }
    size_t pop(float* dst, size_t n) {
        size_t r = readable();
        if (n > r) n = r;
        size_t h = head_.load(std::memory_order_relaxed);
        for (size_t i = 0; i < n; i++) dst[i] = buf_[(h + i) & (cap_ - 1)];
        head_.store(h + n, std::memory_order_release);
        return n;
    }
private:
    std::vector<float> buf_;
    size_t cap_ = 0;
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
};

// 4ch 交错 16k 帧环：帧（4 float）为最小单位，丢弃按整帧计
class FrameRing {
public:
    explicit FrameRing(size_t cap_frames) : ring_(cap_frames * 4) {}
    size_t push_frames(const float* src4i, size_t nframes) {
        size_t wf = ring_.writable() / 4;
        size_t n = nframes < wf ? nframes : wf;
        if (n) ring_.push(src4i, n * 4);
        g_stats.ring_overflow += nframes - n;
        return n;
    }
    size_t readable_frames() const { return ring_.readable() / 4; }
    size_t pop_frames(float* dst4i, size_t nframes) { return ring_.pop(dst4i, nframes * 4) / 4; }
private:
    FloatRing ring_;
};

// ----------------------------------------------------------------------------
// ÷3 多相抽取器：48 抽头 FIR（Blackman 窗 sinc，fc=7.4kHz）抗混叠
// y[m] = Σ_b Σ_a h[b][a]·xb[m−a]，其中 h[b][a] = h[3a+b]，xb[n] = x[3n−b]
// ----------------------------------------------------------------------------
class PolyDecim3 {
public:
    static constexpr int M = 3, TAPS = 48, Q = TAPS / M; // 每相 16 抽头
    void init() {
        const double fc = 7400.0 / (double)kRateMic;
        double h[TAPS], sum = 0;
        for (int k = 0; k < TAPS; k++) {
            double m = k - (TAPS - 1) / 2.0;
            double s = (fabs(m) < 1e-12) ? 2.0 * fc : sin(2.0 * M_PI * fc * m) / (M_PI * m);
            double w = 0.42 - 0.5 * cos(2.0 * M_PI * k / (TAPS - 1)) + 0.08 * cos(4.0 * M_PI * k / (TAPS - 1));
            h[k] = s * w; sum += h[k];
        }
        for (int k = 0; k < TAPS; k++) h[k] /= sum; // DC 增益 1
        for (int b = 0; b < M; b++)
            for (int a = 0; a < Q; a++) h_[b][a] = (float)h[a * M + b];
        memset(hist_, 0, sizeof hist_);
        memset(hi_, 0, sizeof hi_);
        in_count_ = 0;
    }
    // 输入一个样本；产生输出时写 *out 并返回 true
    inline bool push(float x, float* out) {
        uint64_t j = in_count_++;
        int r = (int)(j % M);
        int b = (M - r) % M;               // xb[n]=x[3n−b] ⇒ b=(3−j%3)%3
        hist_[b][hi_[b]] = x;
        hi_[b] = (hi_[b] + 1) % Q;
        if (r != 0) return false;          // j≡0 (mod 3) 时产出
        float acc = 0.f;
        for (int ph = 0; ph < M; ph++) {
            const float* hp = hist_[ph];
            int idx = hi_[ph];             // 下一个写位置；最新样本在 idx-1
            for (int a = 0; a < Q; a++) {
                int pos = (idx - 1 - a + Q) % Q; // 最新样本在 idx-1，往前数 a 个
                acc += h_[ph][a] * hp[pos];
            }
        }
        *out = acc;
        return true;
    }
    int process(const float* in, int n, float* out) { // out 容量 ≥ n/3+1
        int m = 0;
        for (int i = 0; i < n; i++)
            if (push(in[i], &out[m])) m++;
        return m;
    }
private:
    float h_[M][Q];
    float hist_[M][Q];
    int hi_[M];
    uint64_t in_count_;
};

// ----------------------------------------------------------------------------
// WAV 读取（simulate 模式；PCM S16 1ch/4ch；1ch 复制为 4ch）
// ----------------------------------------------------------------------------
struct SimWav {
    int nch = 0, rate = 0;
    std::vector<int16_t> samples; // 4ch interleaved
    bool load(const char* path) {
        FILE* f = fopen(path, "rb");
        if (!f) { fprintf(stderr, "[sim] cannot open %s: %s\n", path, strerror(errno)); return false; }
        char hdr[12];
        if (fread(hdr, 1, 12, f) != 12 || memcmp(hdr, "RIFF", 4) || memcmp(hdr + 8, "WAVE", 4)) {
            fprintf(stderr, "[sim] %s: not RIFF/WAVE\n", path); fclose(f); return false;
        }
        bool got_fmt = false; uint32_t data_off = 0, data_len = 0;
        uint16_t fmt = 0, bits = 0, nch = 0; uint32_t rate = 0;
        for (;;) {
            char id[4]; uint32_t sz;
            if (fread(id, 1, 4, f) != 4 || fread(&sz, 1, 4, f) != 4) break;
            if (!memcmp(id, "fmt ", 4)) {
                uint8_t fb[16];
                size_t take = sz < 16 ? sz : 16;
                if (fread(fb, 1, take, f) != take) break;
                if (sz > take) fseek(f, (long)(sz - take + (sz & 1)), SEEK_CUR);
                fmt = *(uint16_t*)(fb + 0); nch = *(uint16_t*)(fb + 2);
                rate = *(uint32_t*)(fb + 4); bits = *(uint16_t*)(fb + 14);
                got_fmt = true;
            } else if (!memcmp(id, "data", 4)) {
                data_off = (uint32_t)ftell(f); data_len = sz;
                fseek(f, (long)(sz + (sz & 1)), SEEK_CUR);
            } else {
                fseek(f, (long)(sz + (sz & 1)), SEEK_CUR);
            }
        }
        if (!got_fmt || !data_off || fmt != 1 || bits != 16 || (nch != 1 && nch != 4)) {
            fprintf(stderr, "[sim] %s: need PCM S16 1ch/4ch wav (fmt=%u bits=%u ch=%u rate=%u)\n",
                    path, fmt, bits, nch, rate);
            fclose(f); return false;
        }
        fseek(f, (long)data_off, SEEK_SET);
        std::vector<int16_t> raw(data_len / 2);
        size_t rd = fread(raw.data(), 2, raw.size(), f);
        raw.resize(rd);
        fclose(f);
        if (nch == 1) {
            samples.resize(raw.size() * 4);
            for (size_t i = 0; i < raw.size(); i++)
                for (int c = 0; c < 4; c++) samples[i * 4 + c] = raw[i];
        } else {
            samples.swap(raw);
        }
        nch = 4; this->rate = (int)rate;
        fprintf(stderr, "[sim] %s: %dch %dHz %zu frames (%.1fs)\n",
                path, nch, this->rate, samples.size() / 4, (double)(samples.size() / 4) / rate);
        return !samples.empty();
    }
};

// ----------------------------------------------------------------------------
// voice_mic.fifo 写端：无读者时丢弃计数，读者出现自动恢复
// ----------------------------------------------------------------------------
class FifoWriter {
public:
    void try_open(const char* path) {
        if (fd_ >= 0) return;
        int fd = ::open(path, O_WRONLY | O_NONBLOCK | O_CLOEXEC); // 无读者 → ENXIO
        if (fd >= 0) fd_ = fd;
    }
    // 返回丢弃样本数（写成功部分计入调用方 fifo_written）
    size_t write_s16(const int16_t* s, size_t n) {
        if (fd_ < 0) return n;
        const char* p = (const char*)s;
        size_t bytes = n * 2, done = 0;
        while (done < bytes) {
            ssize_t w = ::write(fd_, p + done, bytes - done);
            if (w > 0) { done += (size_t)w; continue; }
            if (errno == EAGAIN || errno == EWOULDBLOCK) break; // 管道满：丢剩余
            ::close(fd_); fd_ = -1;                              // EPIPE 等：等读者重现
            break;
        }
        return n - done / 2;
    }
    bool is_open() const { return fd_ >= 0; }
    void close() { if (fd_ >= 0) { ::close(fd_); fd_ = -1; } }
private:
    int fd_ = -1;
};

// ----------------------------------------------------------------------------
// SHM mic 总线（shm_voice.h）：引擎=单写者。建段（可复用旧段，epoch++）、发布
// 数据环、per-reader wake fifo 广播、读者注册表心跳回收。
// ----------------------------------------------------------------------------
class ShmMic {
public:
    bool init(const char* path) {
        path_ = path;
        uint32_t old_epoch = 0;
        int fd = ::open(path, O_RDWR, 0666);
        if (fd >= 0) {
            vshm_mic_t tmp;
            struct stat st_;
            if (::fstat(fd, &st_) == 0 && st_.st_size == (off_t)sizeof(vshm_mic_t) &&
                ::read(fd, &tmp, sizeof tmp) == (ssize_t)sizeof tmp &&
                tmp.magic == VSHM_MIC_MAGIC && tmp.version == VSHM_PROTO_VERSION) {
                old_epoch = tmp.epoch;                      // 复用同一段：老客户端映射立即看到 epoch 变化
            } else {
                ::close(fd);                                // 段损坏/协议不符：重建
                unlink(path);                               // 损坏段直接删除重建
                fd = -1;
            }
        }
        if (fd < 0) {
            fd = ::open(path, O_RDWR | O_CREAT, 0666);
            if (fd < 0) { fprintf(stderr, "[shm-mic] create %s 失败: %s\n", path, strerror(errno)); return false; }
            if (ftruncate(fd, (off_t)sizeof(vshm_mic_t)) != 0) {
                fprintf(stderr, "[shm-mic] ftruncate 失败: %s\n", strerror(errno)); ::close(fd); return false;
            }
        }
        (void)chmod(path, 0666);                            // sudo 运行时 umask 会砍组/其他位
        void* m = mmap(nullptr, sizeof(vshm_mic_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ::close(fd);
        if (m == MAP_FAILED) { fprintf(stderr, "[shm-mic] mmap 失败: %s\n", strerror(errno)); return false; }
        rb_ = (vshm_mic_t*)m;

        // header 单写者初始化：先清注册表/游标，最后 release 发布 epoch
        for (uint32_t i = 0; i < VSHM_MIC_READERS; i++) {
            __atomic_store_n(&rb_->readers[i].read_seq, 0, __ATOMIC_RELAXED);
            __atomic_store_n(&rb_->readers[i].hb_ms, 0, __ATOMIC_RELAXED);
            __atomic_store_n(&rb_->readers[i].dropped, 0, __ATOMIC_RELAXED);
            __atomic_store_n(&rb_->readers[i].pid, 0, __ATOMIC_RELAXED);
            __atomic_store_n(&rb_->readers[i].used, 0, __ATOMIC_RELAXED);
        }
        __atomic_store_n(&rb_->write_seq, 0, __ATOMIC_RELAXED);
        __atomic_store_n(&rb_->writer_hb_ms, now_ms(), __ATOMIC_RELAXED);
        __atomic_store_n(&rb_->max_lag, 0, __ATOMIC_RELAXED);
        memset(rb_->data, 0, sizeof rb_->data);
        rb_->magic = VSHM_MIC_MAGIC;
        rb_->version = VSHM_PROTO_VERSION;
        rb_->sample_rate = VSHM_MIC_RATE;
        rb_->channels = 1;
        rb_->format_s16le = 1;
        rb_->frame_bytes = 2;
        rb_->log2_frames = VSHM_MIC_LOG2;
        rb_->n_slots = VSHM_MIC_READERS;
        epoch_ = old_epoch + 1;
        __atomic_store_n(&rb_->epoch, epoch_, __ATOMIC_RELEASE);

        // per-reader 唤醒 fifo：引擎持 O_RDWR|O_NONBLOCK（永不半开），发布后写 token
        char wpath[128];
        for (uint32_t i = 0; i < VSHM_MIC_READERS; i++) {
            snprintf(wpath, sizeof wpath, "%s.wake.r%u", path, i);
            mkfifo(wpath, 0666);                            // EEXIST 忽略
            (void)chmod(wpath, 0666);
            wake_fd_[i] = ::open(wpath, O_RDWR | O_NONBLOCK | O_CLOEXEC);
            if (wake_fd_[i] < 0)
                fprintf(stderr, "[shm-mic] wake fifo %s 打开失败: %s（该读者槽将退化为轮询）\n",
                        wpath, strerror(errno));
        }
        fprintf(stderr, "[shm-mic] %s 就绪: ring=%u帧 epoch=%u readers=%u\n",
                path, (unsigned)VSHM_MIC_FRAMES, (unsigned)epoch_, (unsigned)VSHM_MIC_READERS);
        return true;
    }

    // 管线线程每块调用：拷入环（覆盖旧数据）→ release 发布 write_seq → 心跳 → 活跃槽 wake token
    void publish(const int16_t* s, size_t n) {
        uint64_t w = __atomic_load_n(&rb_->write_seq, __ATOMIC_RELAXED);
        size_t done = 0;
        while (done < n) {                                  // 可能跨环尾，分两段拷
            size_t pos = (size_t)(w & (VSHM_MIC_FRAMES - 1));
            size_t seg = std::min(n - done, (size_t)(VSHM_MIC_FRAMES - pos));
            memcpy(&rb_->data[pos], s + done, seg * sizeof(int16_t));
            w += seg; done += seg;
        }
        __atomic_store_n(&rb_->write_seq, w, __ATOMIC_RELEASE);
        __atomic_store_n(&rb_->writer_hb_ms, now_ms(), __ATOMIC_RELAXED);
        const char tok = 1;
        // 向所有 wake fifo 广播 token（不只 used 槽）：引擎重启清空注册表后，
        // 幸存读者仍睡在旧槽 fifo 上——只有 token 能唤醒它去发现 epoch 变化重注册。
        // 未使用 fifo 里堆积的 token 由 64KB 管容上限自然封顶（EAGAIN 忽略）。
        for (uint32_t i = 0; i < VSHM_MIC_READERS; i++) {
            if (wake_fd_[i] < 0) continue;
            ssize_t r = ::write(wake_fd_[i], &tok, 1);      // 满则 EAGAIN 丢弃（token 只是提示）
            (void)r;
        }
    }

    // 读者注册表回收：pid 已死（hb 可不算陈旧）或心跳超时；同时刷新 max_lag 诊断。
    // 管线线程每 ~1s 调一次（每块 32ms，代价 4 槽扫描）。
    void reap() {
        uint64_t now = now_ms();
        uint64_t w = __atomic_load_n(&rb_->write_seq, __ATOMIC_RELAXED);
        int active = 0;
        uint64_t maxlag = __atomic_load_n(&rb_->max_lag, __ATOMIC_RELAXED);
        for (uint32_t i = 0; i < VSHM_MIC_READERS; i++) {
            uint32_t used = __atomic_load_n(&rb_->readers[i].used, __ATOMIC_RELAXED);
            if (!used) continue;
            uint64_t hb = __atomic_load_n(&rb_->readers[i].hb_ms, __ATOMIC_RELAXED);
            pid_t pid = (pid_t)__atomic_load_n(&rb_->readers[i].pid, __ATOMIC_RELAXED);
            bool dead_pid = !pid_alive(pid);
            bool stale = now > hb && now - hb > VSHM_REAP_HB_MS;
            if ((now > hb && now - hb > 2000 && dead_pid) || stale) {
                if (__atomic_load_n(&rb_->epoch, __ATOMIC_RELAXED) != epoch_) return; // 并发重启保护
                __atomic_store_n(&rb_->readers[i].used, 0, __ATOMIC_RELEASE);
                fprintf(stderr, "[shm-mic] 回收读者槽 %u（pid=%d，心跳 %llums 前）\n",
                        i, (int)pid, now > hb ? (unsigned long long)(now - hb) : 0ull);
                continue;
            }
            active++;
            uint64_t lag = w - __atomic_load_n(&rb_->readers[i].read_seq, __ATOMIC_RELAXED);
            if (lag > maxlag) maxlag = lag;
        }
        __atomic_store_n(&rb_->max_lag, maxlag, __ATOMIC_RELAXED);
        readers_ = active;
    }
    int readers() const { return readers_; }
    void deinit() {
        if (rb_) { munmap(rb_, sizeof(vshm_mic_t)); rb_ = nullptr; }
        for (auto& fd : wake_fd_) { if (fd >= 0) ::close(fd); fd = -1; }
    }
private:
    const char* path_ = nullptr;
    vshm_mic_t* rb_ = nullptr;
    uint32_t epoch_ = 0;
    int wake_fd_[VSHM_MIC_READERS];
    int readers_ = 0;
};

// ----------------------------------------------------------------------------
// SHM spk 总线（shm_voice.h）：应用=多写者，引擎=单读者/混音者。收割线程每周期
// poll 写者唤醒 fifo（20ms 兜底）→ 各活跃槽新增数据入本地 staging（float）→
// 求和混音 + 限幅输出；槽积压超环长则跳到最新（count dropped）。
// ----------------------------------------------------------------------------
class ShmSpkBus {
public:
    bool init(const char* path) {
        uint32_t old_epoch = 0;
        int fd = ::open(path, O_RDWR, 0666);
        if (fd >= 0) {
            vshm_spk_t tmp;
            struct stat st_;
            if (::fstat(fd, &st_) == 0 && st_.st_size == (off_t)sizeof(vshm_spk_t) &&
                ::read(fd, &tmp, sizeof tmp) == (ssize_t)sizeof tmp &&
                tmp.magic == VSHM_SPK_MAGIC && tmp.version == VSHM_PROTO_VERSION) {
                old_epoch = tmp.epoch;
            } else {
                ::close(fd);
                unlink(path);
                fd = -1;
            }
        }
        if (fd < 0) {
            fd = ::open(path, O_RDWR | O_CREAT, 0666);
            if (fd < 0) { fprintf(stderr, "[shm-spk] create %s 失败: %s\n", path, strerror(errno)); return false; }
            if (ftruncate(fd, (off_t)sizeof(vshm_spk_t)) != 0) {
                fprintf(stderr, "[shm-spk] ftruncate 失败: %s\n", strerror(errno)); ::close(fd); return false;
            }
        }
        (void)chmod(path, 0666);
        void* m = mmap(nullptr, sizeof(vshm_spk_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ::close(fd);
        if (m == MAP_FAILED) { fprintf(stderr, "[shm-spk] mmap 失败: %s\n", strerror(errno)); return false; }
        rb_ = (vshm_spk_t*)m;

        for (uint32_t i = 0; i < VSHM_SPK_WRITERS; i++) {
            __atomic_store_n(&rb_->writers[i].write_seq, 0, __ATOMIC_RELAXED);
            __atomic_store_n(&rb_->writers[i].consumed_seq, 0, __ATOMIC_RELAXED);
            __atomic_store_n(&rb_->writers[i].hb_ms, 0, __ATOMIC_RELAXED);
            __atomic_store_n(&rb_->writers[i].dropped, 0, __ATOMIC_RELAXED);
            __atomic_store_n(&rb_->writers[i].pid, 0, __ATOMIC_RELAXED);
            __atomic_store_n(&rb_->writers[i].used, 0, __ATOMIC_RELAXED);
            stage_[i].clear(); head_[i] = 0;
            engine_seq_[i] = 0;
            last_data_ms_[i] = 0;
        }
        last_mix_t_ = -1; credit_ = 0;
        __atomic_store_n(&rb_->mixer_hb_ms, now_ms(), __ATOMIC_RELAXED);
        __atomic_store_n(&rb_->mixed_frames, 0, __ATOMIC_RELAXED);
        rb_->magic = VSHM_SPK_MAGIC;
        rb_->version = VSHM_PROTO_VERSION;
        rb_->sample_rate = VSHM_SPK_RATE;
        rb_->channels = 1;
        rb_->format_s16le = 1;
        rb_->frame_bytes = 2;
        rb_->log2_frames = VSHM_SPK_LOG2;
        rb_->n_slots = VSHM_SPK_WRITERS;
        epoch_ = old_epoch + 1;
        __atomic_store_n(&rb_->epoch, epoch_, __ATOMIC_RELEASE);

        // 写者→引擎 唤醒 fifo：引擎持读端（O_RDWR 持双端防半开噪音），写者写 token
        char wpath[128];
        snprintf(wpath, sizeof wpath, "%s.wake", path);
        mkfifo(wpath, 0666);
        (void)chmod(wpath, 0666);
        wake_fd_ = ::open(wpath, O_RDWR | O_NONBLOCK | O_CLOEXEC);
        if (wake_fd_ < 0)
            fprintf(stderr, "[shm-spk] wake fifo %s 打开失败: %s（退化为 20ms 定时轮询）\n",
                    wpath, strerror(errno));
        fprintf(stderr, "[shm-spk] %s 就绪: %u 写者槽×%u帧 epoch=%u\n",
                path, (unsigned)VSHM_SPK_WRITERS, (unsigned)VSHM_SPK_FRAMES, (unsigned)epoch_);
        return true;
    }

    // 收割+混音一个批次（spk 线程调用）。busy_s 返回本周期非等待耗时（RTF 统计）。
    size_t harvest(float* out, size_t maxn, double* busy_s) {
        struct pollfd pfd = { wake_fd_, POLLIN, 0 };
        if (wake_fd_ >= 0) poll(&pfd, 1, 8);                // 写者 token 提前唤醒；8ms 兜底（IO 8ms 帧设计）
        double t0 = now_s();
        if (wake_fd_ >= 0) {                                // 排干 token（非阻塞）
            char t[256];
            while (::read(wake_fd_, t, sizeof t) == (ssize_t)sizeof t) {}
        }
        uint64_t now = now_ms();
        __atomic_store_n(&rb_->mixer_hb_ms, now, __ATOMIC_RELAXED);

        // ① 各活跃槽：收割新增数据进 staging float；回收死写者；追赶超限积压
        size_t maxlen = 0;
        int active = 0;
        for (uint32_t i = 0; i < VSHM_SPK_WRITERS; i++) {
            if (__atomic_load_n(&rb_->writers[i].used, __ATOMIC_RELAXED) == 0) continue;
            uint64_t hb = __atomic_load_n(&rb_->writers[i].hb_ms, __ATOMIC_RELAXED);
            pid_t pid = (pid_t)__atomic_load_n(&rb_->writers[i].pid, __ATOMIC_RELAXED);
            if ((now > hb && now - hb > 2000 && !pid_alive(pid)) ||
                (now > hb && now - hb > VSHM_REAP_HB_MS)) {
                if (__atomic_load_n(&rb_->epoch, __ATOMIC_RELAXED) != epoch_) break;
                __atomic_store_n(&rb_->writers[i].used, 0, __ATOMIC_RELEASE);
                // 槽释放但 staging 可能还有尾巴：留给孤儿排空（见②），不在此 clear
                fprintf(stderr, "[shm-spk] 回收写者槽 %u（pid=%d，staging 余 %zu 帧）\n",
                        i, (int)pid, stage_[i].size() - head_[i]);
                continue;
            }
            active++;
            uint64_t w = __atomic_load_n(&rb_->writers[i].write_seq, __ATOMIC_ACQUIRE);
            uint64_t c = engine_seq_[i];
            if (w < c) { engine_seq_[i] = w; c = w; }       // 写者重注册归零：重新锚定
            if (w > c) {
                if (w - c > VSHM_SPK_FRAMES) {                  // 积压超环长：跳到最新（丢旧）
                    uint64_t drop = w - c - VSHM_SPK_FRAMES;
                    __atomic_store_n(&rb_->writers[i].dropped,
                        __atomic_load_n(&rb_->writers[i].dropped, __ATOMIC_RELAXED) + drop,
                        __ATOMIC_RELAXED);
                    g_stats.shm_dropped += drop;
                    c = w - VSHM_SPK_FRAMES;
                }
                // 拷 c..w 进 staging（可能跨环尾，两段）
                size_t n = (size_t)(w - c);
                size_t old_sz = stage_[i].size();
                stage_[i].resize(old_sz + n);
                size_t done = 0;
                while (done < n) {
                    size_t pos = (size_t)(c & (VSHM_SPK_FRAMES - 1));
                    size_t seg = std::min(n - done, (size_t)(VSHM_SPK_FRAMES - pos));
                    const int16_t* src = &rb_->data[i][pos];
                    float* dst = &stage_[i][old_sz + done];
                    for (size_t k = 0; k < seg; k++) dst[k] = src[k] / 32768.f;
                    c += seg; done += seg;
                }
                engine_seq_[i] = w;
                last_data_ms_[i] = now;
                __atomic_store_n(&rb_->writers[i].consumed_seq, w, __ATOMIC_RELEASE);
            }
        }
        writers_ = active;

        // ② 混音——墙钟节拍 + 最小前沿（关键：不能"到多少泄多少"，否则各写者的
        //    ~85ms chunk 互相错过、永远不重叠、输出变成单路交替；须等活跃写者
        //    的 chunk 都到齐（前沿=min 各路 staged），再按 elapsed×rate 的额度求和输出）。
        //    - ACTIVE（used 且心跳新鲜且 250ms 内有到货或手上有数据）：约束前沿；
        //    - 手上无数据的 ACTIVE 写者（下个 chunk 在途）→ 本次 hold（输出 0 等对齐）；
        //      超过 250ms 无到货（应用暂停）→ 视为静音、不再约束前沿；
        //    - 孤儿 staging（写者已注销但数据未放完）→ 只参与求和与上限，不约束。
        double nowf = now_s();
        if (last_mix_t_ < 0) last_mix_t_ = nowf;
        credit_ = std::min(credit_ + (nowf - last_mix_t_) * (double)VSHM_SPK_RATE,
                           (double)maxn);
        last_mix_t_ = nowf;
        bool hold = false;
        size_t frontier = SIZE_MAX;                       // ACTIVE 写者的最小 staged
        for (uint32_t i = 0; i < VSHM_SPK_WRITERS; i++) {
            size_t avail = stage_[i].size() - head_[i];
            maxlen = std::max(maxlen, avail);
            if (__atomic_load_n(&rb_->writers[i].used, __ATOMIC_RELAXED) == 0) continue;
            uint64_t hb = __atomic_load_n(&rb_->writers[i].hb_ms, __ATOMIC_RELAXED);
            if (now > hb && now - hb > VSHM_HB_GRACE_MS) continue;
            if (avail > 0) frontier = std::min(frontier, avail);
            else if (last_data_ms_[i] == 0 || now > last_data_ms_[i] + 250) continue;
            else hold = true;                             // 活跃且刚有数据流：等它的下一段
        }
        size_t n;
        if (frontier == SIZE_MAX)
            n = hold ? 0 : std::min(maxlen, maxn);        // 无 ACTIVE：仅孤儿排空
        else if (hold)
            n = 0;                                        // 有 ACTIVE 缺货：等对齐
        else
            n = std::min(std::min(frontier, maxlen), maxn);
        n = std::min(n, (size_t)credit_);
        for (size_t k = 0; k < n; k++) {
            float acc = 0.f;
            for (uint32_t i = 0; i < VSHM_SPK_WRITERS; i++) {
                size_t len = stage_[i].size();
                if (head_[i] + k < len) acc += stage_[i][head_[i] + k];
            }
            out[k] = acc < -1.f ? -1.f : (acc > 1.f ? 1.f : acc);
        }
        for (uint32_t i = 0; i < VSHM_SPK_WRITERS; i++) {   // 消费头部 n 帧；周期性压实
            head_[i] = std::min(head_[i] + n, stage_[i].size());
            if (head_[i] > VSHM_SPK_FRAMES) {
                stage_[i].erase(stage_[i].begin(), stage_[i].begin() + (long)head_[i]);
                head_[i] = 0;
            }
        }
        if (n > 0) credit_ -= (double)n;
        if (n) {
            uint64_t m = __atomic_load_n(&rb_->mixed_frames, __ATOMIC_RELAXED) + n;
            __atomic_store_n(&rb_->mixed_frames, m, __ATOMIC_RELAXED);
            g_stats.mixed_frames += n;
        }
        if (busy_s) *busy_s = now_s() - t0;
        return n;
    }

    int writers() const { return writers_; }
    void deinit() {
        if (rb_) { munmap(rb_, sizeof(vshm_spk_t)); rb_ = nullptr; }
        if (wake_fd_ >= 0) { ::close(wake_fd_); wake_fd_ = -1; }
    }
private:
    vshm_spk_t* rb_ = nullptr;
    uint32_t epoch_ = 0;
    int wake_fd_ = -1;
    int writers_ = 0;
    std::vector<float> stage_[VSHM_SPK_WRITERS];            // 每槽已收割未混音数据
    size_t head_[VSHM_SPK_WRITERS] = {};                    // staging 头部消费游标
    uint64_t engine_seq_[VSHM_SPK_WRITERS] = {};            // 引擎对各槽的收割游标
    uint64_t last_data_ms_[VSHM_SPK_WRITERS] = {};          // 各槽最近到货时刻（前沿判定）
    double last_mix_t_ = -1;                                // 混音墙钟节拍
    double credit_ = 0;                                     // 可输出帧额度（elapsed×rate）
};

// ----------------------------------------------------------------------------
// 简易 WAV 写出（48k mono S16；--dump-mix 用；关闭时回填 RIFF 长度）
// ----------------------------------------------------------------------------
class WavWriter {
public:
    bool open(const char* path, uint32_t rate) {
        f_ = fopen(path, "wb");
        if (!f_) { fprintf(stderr, "[dump-mix] 打开 %s 失败: %s\n", path, strerror(errno)); return false; }
        rate_ = rate;
        static const char hdr[44] = {0};                    // 占位，close 时回填
        fwrite(hdr, 1, 44, f_);
        return true;
    }
    void write_s16(const int16_t* s, size_t n) {
        if (!f_) return;
        fwrite(s, sizeof(int16_t), n, f_);
        frames_ += n;
        if ((frames_ & 0xFFFF) == 0) fflush(f_);            // 定期刷盘，防 kill -9 丢太多
    }
    void close() {
        if (!f_) return;
        fflush(f_);
        uint32_t data_bytes = (uint32_t)(frames_ * 2);
        uint32_t riff = 36 + data_bytes;
        fseek(f_, 0, SEEK_SET);
        fputs("RIFF", f_); fwrite(&riff, 4, 1, f_); fputs("WAVE", f_);
        fputs("fmt ", f_);
        uint32_t fsz = 16;  uint16_t fmt = 1, ch = 1, bits = 16;
        fwrite(&fsz, 4, 1, f_); fwrite(&fmt, 2, 1, f_); fwrite(&ch, 2, 1, f_);
        fwrite(&rate_, 4, 1, f_);
        uint32_t bpsec = rate_ * 2; fwrite(&bpsec, 4, 1, f_);
        uint16_t align = 2; fwrite(&align, 2, 1, f_); fwrite(&bits, 2, 1, f_);
        fputs("data", f_); fwrite(&data_bytes, 4, 1, f_);
        fclose(f_); f_ = nullptr;
        fprintf(stderr, "[dump-mix] 落盘 %llu 帧（%.2fs @%uHz）\n",
                (unsigned long long)frames_, (double)frames_ / rate_, (unsigned)rate_);
    }
private:
    FILE* f_ = nullptr;
    uint32_t rate_ = 48000;
    uint64_t frames_ = 0;
};

// ----------------------------------------------------------------------------
// dump 管理（2026-10-07）：落盘应用目录而非 /tmp（tmpfs 243MB 会被写满，
// 满盘时 fopen 无声失败曾长期被误判为"引擎启动失败"），带时间戳命名 +
// 总量上限轮转（超限按 mtime 删最旧）。
//   VA_DUMP_MIC / VA_DUMP_REF 取值：含 '/' 的路径 → 原行为（写指定文件）；
//   其他任意值（"1"/"on"…）→ 自动路径 {VA_DUMP_DIR:-./dumps}/{mic|ref}_时间戳.f32
//   VA_DUMP_MAX_MB：目录轮转上限（默认 500MB，/home 分区 13G 可用）
//   轮转时机：dump 打开后 + 每 512 块（~16s）；当前打开的文件受保护不删
// ----------------------------------------------------------------------------
static std::string dump_auto_path(const char* prefix) {
    const char* dir = getenv("VA_DUMP_DIR");
    if (!dir || !*dir) dir = "./dumps";
    mkdir(dir, 0777);   // EEXIST 忽略
    char ts[32];
    time_t t = time(nullptr);
    struct tm tmv;
    localtime_r(&t, &tmv);
    strftime(ts, sizeof ts, "%Y%m%d_%H%M%S", &tmv);
    return std::string(dir) + "/" + prefix + "_" + ts + ".f32";
}

static void dump_rotate(const char* skip1, const char* skip2) {
    const char* dir = getenv("VA_DUMP_DIR");
    if (!dir || !*dir) dir = "./dumps";
    long maxmb = 500;
    const char* m = getenv("VA_DUMP_MAX_MB");
    if (m && atol(m) > 0) maxmb = atol(m);
    DIR* d = opendir(dir);
    if (!d) return;
    struct DumpEnt { std::string path; long long sz; time_t mt; };
    std::vector<DumpEnt> ents;
    long long total = 0;
    while (dirent* e = readdir(d)) {
        const std::string n = e->d_name;
        if (n.size() < 5) continue;
        const std::string ext = n.substr(n.size() - 4);
        if (ext != ".f32" && ext != ".wav") continue;
        const std::string p = std::string(dir) + "/" + n;
        if (p == skip1 || p == skip2) continue;
        struct stat st;
        if (stat(p.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) continue;
        ents.push_back({p, (long long)st.st_size, st.st_mtime});
        total += st.st_size;
    }
    closedir(d);
    const long long maxb = (long long)maxmb * 1024 * 1024;
    if (total <= maxb) return;
    std::sort(ents.begin(), ents.end(),
              [](const DumpEnt& a, const DumpEnt& b) { return a.mt < b.mt; });
    for (const auto& e : ents) {
        if (total <= maxb) break;
        if (unlink(e.path.c_str()) == 0) {
            total -= e.sz;
            fprintf(stderr, "[dump] 轮转删除 %s（%.1f MB，目录超 %ldMB 上限）\n",
                    e.path.c_str(), e.sz / 1048576.0, maxmb);
        }
    }
}

// ----------------------------------------------------------------------------
// 算法封装：4×mono AEC（共享参考谱）+ 1×DOA/BF
//   2026-10-06 单核化：撤 AEC worker（旧双核并行在算法 rtf ~0.5 后收益不再，
//   且少占一核/少一线程功耗；管线线程串行 4×AEC，win_compute 口径不变）。
//   共享参考谱：4 实例同 ref，x FFT/谱环/|X|² 表只算一次（RefEngine 服务），
//   与每实例独立 FFT 输出逐位一致（test_algo 固化）。
// ----------------------------------------------------------------------------
struct Algo {
    // 句柄类型取自真库头；va_mock.h 已同步声明相同 typedef（mock 可继续编译）
    va_aec_ref* refsvc = nullptr;
    va_aec* aec[kNChan] = {};
    va_doa_bf* bf = nullptr;
    bool init() {
        refsvc = va_aec_ref_create(kRateEng);
        if (!refsvc) return false;
        for (int c = 0; c < kNChan; c++) {
            aec[c] = va_aec_create_shared(kRateEng, refsvc);
            if (!aec[c]) return false;
        }
        bf = va_doa_bf_create(kNChan, kRateEng, "ula", kMicSpacing);
        return bf != nullptr;
    }
    void deinit() {
        for (int c = 0; c < kNChan; c++) if (aec[c]) va_aec_destroy(aec[c]);
        if (refsvc) va_aec_ref_destroy(refsvc);
        if (bf) va_doa_bf_destroy(bf);
    }
};

// 显式 hw_params 配置（本板 sunxi 驱动拒绝 snd_pcm_set_params 的默认组合，readi EIO）
static int pcm_hw_setup(snd_pcm_t* pcm, snd_pcm_format_t fmt, int nch, int rate,
                        snd_pcm_uframes_t period, int nperiods) {
    snd_pcm_hw_params_t* hp;
    snd_pcm_hw_params_alloca(&hp);
    int rc = snd_pcm_hw_params_any(pcm, hp);
    if (rc < 0) return rc;
    if ((rc = snd_pcm_hw_params_set_access(pcm, hp, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0) return rc;
    if ((rc = snd_pcm_hw_params_set_format(pcm, hp, fmt)) < 0) return rc;
    if ((rc = snd_pcm_hw_params_set_channels(pcm, hp, nch)) < 0) return rc;
    unsigned r = (unsigned)rate; int dir = 0;
    if ((rc = snd_pcm_hw_params_set_rate_near(pcm, hp, &r, &dir)) < 0) return rc;
    if ((int)r != rate) return -EINVAL;
    snd_pcm_uframes_t per = period, bs = period * nperiods;
    if ((rc = snd_pcm_hw_params_set_period_size_near(pcm, hp, &per, &dir)) < 0) return rc;
    if ((rc = snd_pcm_hw_params_set_buffer_size_near(pcm, hp, &bs)) < 0) return rc;
    return snd_pcm_hw_params(pcm, hp);
}

static Algo      g_algo;
static FrameRing g_cap_ring(kRateEng * 4 * 4); // 采集 → 管线：~4s
static FloatRing g_ref_ring(kRateEng * 5);     // 回采 → 管线：~5s
static Config    g_cfg;
static FifoWriter g_mic_fifo;
static PolyDecim3 g_ref_decim;
static ShmMic    g_mic_shm;                    // shm 模式：mic 环写者（单写多读，shm_voice.h）
static ShmSpkBus g_spk_shm;                    // shm 模式：spk 收割混音者（多写单读）

// ----------------------------------------------------------------------------
// （已撤）AEC 并行 worker：曾处理通道 {2,3} 于核 3。算法三项优化
//（半谱/增量 GCC/错峰投影）后全链 rtf ~0.5，双核并行收益不再；单核串行
// 4×AEC 少占一核、少一线程空转功耗，win_compute 口径不变（管线线程窗口
// 本就包含汇合等待）。共享参考谱见 Algo 注释。
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// 线程1a：真实采集 hw:1,0 → tag 解码 → 16k 4ch 入环
//   --capture-fs 16k（默认）: 32k 线上流；剥 tag 与 float 转换合并一步
//                             （w&3 分组 + (w>>16) 取 16bit ÷32768），无软件抽取
//   --capture-fs 48k       : 96k 线上流 → tag 解码 4ch@48k → ÷3 多相抽取 → 16k
// ----------------------------------------------------------------------------
struct CaptureCtx {
    float   q[kNChan][8];
    uint64_t qw[kNChan] = {0, 0, 0, 0}; // 每通道累计入队数
    uint64_t qr[kNChan] = {0, 0, 0, 0}; // 每通道累计取走数
    PolyDecim3 dec[kNChan];
    bool direct16 = true;               // 16k 直采：跳过 decimator
    uint64_t skip_words = 32000;        // 跳过流起始 ~0.5s 瞬态（=0.5×2×wire_rate 词）
    uint64_t frames = 0;
    bool warned_skew = false;
    std::vector<float> stage;           // 16k 4ch 帧 staging（攒 kBlock 帧入环）
    void init() {
        for (int c = 0; c < kNChan; c++) dec[c].init();
        stage.reserve(kBlock * 4);
    }
    void feed_word(int32_t w) {
        int tag = w & 3;
        g_stats.tag_counts[tag]++;
        g_stats.wire_words++;
        if (skip_words) { skip_words--; return; }
        // 剥 tag（低 2 位）+ MSB 对齐取 16bit + float 归一，一步完成
        q[tag][qw[tag] & 7] = (int16_t)((uint32_t)w >> 16) / 32768.f;
        qw[tag]++;
        // 四通道各至少一词 → 产出一个 ADC 帧（tag 自带轮转相位，无需对齐逻辑）
        bool all = true;
        for (int c = 0; c < kNChan; c++) if (qw[c] == qr[c]) { all = false; break; }
        if (!all) return;
        // 装配一个 ADC 帧：4 通道各消费一词。48k 模式下 4 个 decimator 输入计数
        // 恒一致，必然同拍产出（每 3 帧同时各出一个 16k 样本）——消费与产出解耦；
        // 16k 直采模式下剥 tag 后即为最终样本，逐帧直发。
        float outs[kNChan];
        bool emitted = false;
        for (int c = 0; c < kNChan; c++) {
            float s = q[c][qr[c] & 7];
            qr[c]++;
            if (direct16) { outs[c] = s; emitted = true; continue; }
            float o;
            bool e = dec[c].push(s, &o);
            if (e) { outs[c] = o; emitted = true; }
        }
        if (!emitted) return;
        stage.insert(stage.end(), outs, outs + kNChan);
        if (stage.size() >= (size_t)kBlock * 4) {
            g_cap_ring.push_frames(stage.data(), kBlock);
            cap_stamp(frames);
            g_stats.captured_samples += (uint64_t)kBlock * 4;
            stage.clear();
        }
    }
};

static void* capture_hw_thread(void* /*arg*/) {
    pin_cpu(0, "capture");
    const bool fs16 = (g_cfg.capture_fs == 16000);
    const int   wire_rate = fs16 ? kRateWire16 : kRateWire48; // 线上速率 = 2×ADC fs
    CaptureCtx cx;
    cx.direct16 = fs16;
    cx.skip_words = (uint64_t)wire_rate;        // 0.5s × 2ch 词/秒 = wire_rate 词
    cx.init();

    snd_pcm_t* pcm = nullptr;
    int rc = snd_pcm_open(&pcm, g_cfg.capture_dev, SND_PCM_STREAM_CAPTURE, 0);
    if (rc < 0) {
        fprintf(stderr, "[capture] open %s 失败: %s\n", g_cfg.capture_dev, snd_strerror(rc));
        fprintf(stderr, "[capture] 权限不足则用 sudo 运行；AC108 未配置则先跑: ~/ac108/mic4-record%s.sh %s\n",
                fs16 ? "16" : "", fs16 ? "cfg" : "1 /dev/null");
        g_capture_failed = true;
        g_stop.store(true);
        return nullptr;
    }
    rc = pcm_hw_setup(pcm, SND_PCM_FORMAT_S32_LE, 2, wire_rate, wire_rate / 200, 8); // 5ms×8=40ms 线上
    if (rc < 0) {
        fprintf(stderr, "[capture] hw_params 失败: %s（%dk 模式线上流必须 %dk S32 2ch；"
                        "先跑 ~/ac108/mic4-record%s.sh %s 配方）\n",
                snd_strerror(rc), fs16 ? 16 : 48, wire_rate / 1000,
                fs16 ? "16" : "", fs16 ? "cfg" : "1 /dev/null");
        snd_pcm_close(pcm);
        g_capture_failed = true;
        g_stop.store(true);
        return nullptr;
    }
    fprintf(stderr, "[capture] %s: %dk S32 2ch wire stream（ADC %dk 直采，%s）\n",
            g_cfg.capture_dev, wire_rate / 1000, fs16 ? 16 : 48,
            fs16 ? "剥 tag 即 16k float，无软件抽取" : "÷3 多相抽取 → 16k");

    const snd_pcm_uframes_t PERIOD = wire_rate / 200; // 5ms 线上 = 2×PERIOD 词
    std::vector<int32_t> buf(PERIOD * 2);
    while (!g_stop.load(std::memory_order_relaxed)) {
        snd_pcm_sframes_t n = snd_pcm_readi(pcm, buf.data(), PERIOD);
        if (n == -EINTR) continue;
        if (n < 0) {
            if (n == -EPIPE) { g_stats.overrun++; snd_pcm_prepare(pcm); continue; }
            if (n == -ESTRPIPE) {
                while ((rc = snd_pcm_resume(pcm)) == -EAGAIN && !g_stop.load(std::memory_order_relaxed))
                    usleep(10000);
                if (rc < 0) snd_pcm_prepare(pcm);
                continue;
            }
            fprintf(stderr, "[capture] readi error: %s\n", snd_strerror((int)n));
            g_capture_failed = true;
            break;
        }
        size_t nwords = (size_t)n * 2;
        g_stats.cap_reads++;
        int64_t mn = g_stats.cap_min_n.load(std::memory_order_relaxed);
        while ((int64_t)n < mn && !g_stats.cap_min_n.compare_exchange_weak(mn, (int64_t)n)) {}
        for (size_t i = 0; i < nwords; i++) cx.feed_word(buf[i]);

        // tag 分布健康检查（1s 线上流后）
        if (!cx.warned_skew && g_stats.wire_words.load() > (uint64_t)wire_rate * 2) {
            uint64_t tot = 0, tc[4];
            for (int t = 0; t < 4; t++) { tc[t] = g_stats.tag_counts[t].load(); tot += tc[t]; }
            uint64_t mx = std::max(std::max(tc[0], tc[1]), std::max(tc[2], tc[3]));
            if (tot > 0 && mx * 100 > tot * 40) {
                fprintf(stderr,
                    "[capture] tag 分布异常 [%.1f %.1f %.1f %.1f]%%：线上流损坏或 AC108 未配置。\n"
                    "[capture] 请先运行: ~/ac108/mic4-record%s.sh %s\n",
                    100.0 * tc[0] / tot, 100.0 * tc[1] / tot, 100.0 * tc[2] / tot, 100.0 * tc[3] / tot,
                    fs16 ? "16" : "", fs16 ? "cfg" : "1 /dev/null");
                cx.warned_skew = true;
            }
        }
    }
    snd_pcm_close(pcm);
    fprintf(stderr, "[capture] stopped\n");
    return nullptr;
}

// ----------------------------------------------------------------------------
// 线程1b：simulate 捕获（WAV 循环，实时节奏 16000 帧/s）
// ----------------------------------------------------------------------------
static void* capture_sim_thread(void* /*arg*/) {
    pin_cpu(0, "sim-capture");
    SimWav wav;
    if (!wav.load(g_cfg.sim_wav) || wav.rate != kRateEng) {
        if (wav.rate != kRateEng && wav.rate)
            fprintf(stderr, "[sim] wav rate %d != %d\n", wav.rate, kRateEng);
        g_capture_failed = true;
        g_stop.store(true);
        return nullptr;
    }
    const size_t total = wav.samples.size() / 4;
    size_t pos = 0;
    double t0 = now_s();
    std::vector<float> blk(kBlock * 4);
    while (!g_stop.load(std::memory_order_relaxed)) {
        double allowed = (now_s() - t0) * kRateEng; // 实时节奏上限
        if ((double)(pos + kBlock) <= allowed + 1.5 * kBlock) {
            for (int i = 0; i < kBlock; i++) {
                size_t p = (pos + (size_t)i) % total;
                for (int c = 0; c < 4; c++) blk[(size_t)i * 4 + c] = wav.samples[p * 4 + c] / 32768.f;
            }
            g_cap_ring.push_frames(blk.data(), kBlock);
            cap_stamp((uint64_t)pos);
            g_stats.captured_samples += (uint64_t)kBlock * 4;
            pos += kBlock;
        } else {
            usleep(2000);
        }
    }
    fprintf(stderr, "[sim] stopped at %zu frames (%.1fs)\n", pos, (double)pos / kRateEng);
    return nullptr;
}

// ----------------------------------------------------------------------------
// 线程2：管线 AEC×4 → DOA/BF → fifo 写 + stats 打印
// ----------------------------------------------------------------------------
static void* pipeline_thread(void* /*arg*/) {
    std::vector<float> blk(kBlock * 4);
    static thread_local float micm[kNChan][kBlock], outm[kNChan][kBlock];
    static thread_local float refm[kBlock], mono[kBlock];
    static thread_local int16_t s16[kBlock];
    // 联调诊断 dump：VA_DUMP_REF / VA_DUMP_MIC（取值含 '/'=指定路径；否则
    // 自动 dumps/ 目录+时间戳；见 dump_auto_path 注释）。ref=实际喂给 AEC 的
    // 参考块（含静音垫）；mic=AEC 输入前原始 4ch 平面（物理通道序，块内平面/
    // 块间交错）——两者同块对齐，供离线复算真实声学 per-ch ERLE/DOA
    auto dump_target = [](const char* v, const char* prefix) -> std::string {
        if (!v || !*v) return {};
        if (std::strchr(v, '/')) return v;
        return dump_auto_path(prefix);
    };
    const std::string ref_dump_path = dump_target(getenv("VA_DUMP_REF"), "ref");
    const std::string mic_dump_path = dump_target(getenv("VA_DUMP_MIC"), "mic");
    FILE* ref_dump = ref_dump_path.empty() ? nullptr : fopen(ref_dump_path.c_str(), "wb");
    FILE* mic_dump = mic_dump_path.empty() ? nullptr : fopen(mic_dump_path.c_str(), "wb");
    const std::string aec_dump_path = dump_target(getenv("VA_DUMP_AEC"), "aec");
    const std::string bf_dump_path = dump_target(getenv("VA_DUMP_BF"), "bf");
    FILE* aec_dump = aec_dump_path.empty() ? nullptr : fopen(aec_dump_path.c_str(), "wb");
    FILE* bf_dump = bf_dump_path.empty() ? nullptr : fopen(bf_dump_path.c_str(), "wb");
    if (ref_dump || mic_dump) {
        dump_rotate(ref_dump_path.c_str(), mic_dump_path.c_str());
        fprintf(stderr, "[dump] ref=%s mic=%s\n",
                ref_dump_path.empty() ? "-" : ref_dump_path.c_str(),
                mic_dump_path.empty() ? "-" : mic_dump_path.c_str());
    }
    long dump_house_cnt = 0;
    double stat_t0 = now_s(), win_compute = 0, win_audio = 0, start = now_s();
    double last_reap = 0;                                   // shm 读者注册表回收节拍

    // ref↔mic 游标对齐状态机（联调核心修复，2026-10-06 重构去睡眠化）：按“累计消费
    // ref 样本数 == 累计处理 mic 样本数”配对。两条流各自按内容序号索引（sim 的回声按
    // 恒等路径烘焙 / 真实声学路径为短时延因果系统），游标配对使 AEC 有效滞后结构性
    // 为 0，与进程启动抖动无关。离线扫描（repro_aec）表明：滞后 0 时 ERLE +20~25dB；
    // 谐波回声下滞后 ≥50ms 即发散（−0.5~−8dB）——此前“固定引导延迟”方案因此失效。
    //   DRAINED: 无有效对齐——每块补静音出流，**不睡眠、不停块**；同时观察 ref 环：
    //             攒满 kRefCushion（写者活跃，吸收 40ms 块级到达抖动）→ 升格 RUN；
    //             滞留 1..kRefCushion-1 且 >1s 无增长（写者遗留死尾巴）→ 内联丢弃
    //             （防陈旧数据在下次写者到来时被错位消费；也防旧版 HOLD↔DRAINED
    //             无限乒乓活锁——管线塌缩 1 块/s、“停摆”的根因之一）。
    //   RUN:     每块等 ref 攒够整块再弹出（到达率==消费率，等待不积累欠账；空管
    //            300ms → DRAINED）。绝不 mid-stream 部分弹出撕裂参考流（联调实测
    //            5 个 513~1025 样本空洞即令有效滞后漂移 3~8 块）。
    // 旧版差异：HOLD 状态（睡眠等待 cushion ≤1s）删除——块完成率（stats 节拍、
    // voice_mic 出流）与 ref 可用性彻底解耦，写者断流/抖动只影响 ref 内容不影响节拍。
    constexpr size_t kRefCushion = 512;  // 32ms @16k（滞后恒定项；突发间隔 10.7ms，RUN 期 300ms 空管容忍兜底 DRAINED↔RUN 不乒乓）
    constexpr double kStaleRefS = 1.0;   // DRAINED 下滞留 ref 判死时长
    enum RefSync { DRAINED, RUN };
    RefSync rs = DRAINED;
    size_t stale_seen = 0;               // DRAINED 下上次观测的环存量（判“无增长”）
    double stale_t0 = now_s();
    pin_cpu(2, "pipeline");   // 管线+算法独占核2；采集核0、回采核1（核3 空闲——AEC worker 已撤）

    while (!g_stop.load(std::memory_order_relaxed)) {
        if (rs == DRAINED) {
            size_t avail = g_ref_ring.readable();
            if (avail >= kRefCushion) {
                rs = RUN;                                        // 写者活跃：cushion 攒满升格
            } else if (avail > 0) {
                double now = now_s();
                if (avail != stale_seen) { stale_seen = avail; stale_t0 = now; }
                else if (now - stale_t0 > kStaleRefS) {
                    size_t stale = 0;                            // 写者遗留死尾巴：内联丢弃
                    while (g_ref_ring.readable() > 0) stale += g_ref_ring.pop(refm, kBlock);
                    g_stats.ref_discarded += stale;
                    stale_seen = 0;
                    fprintf(stderr, "[pipeline] DRAINED：丢弃滞留 ref %zu 样本（%.1fs 无增长，写者遗留）\n",
                            stale, now - stale_t0);
                }
            } else {
                stale_seen = 0;
            }
        }
        if (g_cap_ring.readable_frames() < (size_t)kBlock) { usleep(1000); continue; }
        g_cap_ring.pop_frames(blk.data(), kBlock);
        {   // mic-in 环滞留 EMA（ALSA 驱动缓冲另计）
            g_pipe_consumed += kBlock;
            double lag = cap_lag_ms(g_pipe_consumed);
            if (lag >= 0) {
                float prev = g_stats.mic_in_ms.load(std::memory_order_relaxed);
                float v = prev < 0 ? (float)lag : prev * 0.9f + (float)lag * 0.1f;
                g_stats.mic_in_ms.store(v, std::memory_order_relaxed);
            }
        }

        double t0 = now_s();  // 含解交错（ref 同步后另有重置点）
        for (int i = 0; i < kBlock; i++)
            for (int c = 0; c < kNChan; c++) micm[c][i] = blk[(size_t)i * 4 + c];
        size_t got = 0;
        if (rs == RUN) {
            for (int w = 0; g_ref_ring.readable() < (size_t)kBlock &&
                            !g_stop.load(std::memory_order_relaxed); w++) {
                if (g_ref_ring.readable() == 0 && w >= 150) { rs = DRAINED; stale_seen = 0; break; } // 300ms 空管
                if (w >= 250) break;                                                // 500ms 仍不足整块（异常兜底）
                usleep(2000);
            }
            if (rs == RUN) got = g_ref_ring.pop(refm, kBlock);
        }
        if (got < (size_t)kBlock) g_stats.ref_padded += (size_t)kBlock - got;
        for (size_t i = got; i < (size_t)kBlock; i++) refm[i] = 0.f;
        if (ref_dump) fwrite(refm, sizeof(float), kBlock, ref_dump);
        // rtf 计时从 ref 同步之后开始：ref 等待（追赶/EOF 判定）是空转而非计算
        t0 = now_s();
        if (mic_dump)    // 离线复算插桩：AEC 前原始 4ch（物理通道序，平面 float）
            for (int c = 0; c < kNChan; c++)
                fwrite(micm[c], sizeof(float), kBlock, mic_dump);
        if ((mic_dump || ref_dump) && ++dump_house_cnt >= 512) {  // ~16s 目录轮转
            dump_house_cnt = 0;
            dump_rotate(ref_dump_path.c_str(), mic_dump_path.c_str());
        }
        // 单核串行：先推共享参考谱（一次 FFT/谱环），再 4 通道残差
        va_aec_ref_push(g_algo.refsvc, refm, kBlock);
        for (int c = 0; c < kNChan; c++)
            va_aec_process_shared(g_algo.aec[c], micm[c], refm, outm[c], kBlock);
        // 真库 ABI：in 为 4ch 平面指针数组（in[nch][n]），直接用 outm 去交错结果
        const float* in4[kNChan] = { outm[0], outm[1], outm[2], outm[3] };
        const float* bf_in[kNChan];
        for (int c = 0; c < kNChan; c++) bf_in[c] = in4[g_cfg.mic_order[c]];
        if (aec_dump)   // AEC 后（重排前，物理通道序）
            for (int c = 0; c < kNChan; c++)
                fwrite(outm[c], sizeof(float), kBlock, aec_dump);
        va_doa_bf_process(g_algo.bf, bf_in, mono, kBlock);
        if (bf_dump) fwrite(mono, sizeof(float), kBlock, bf_dump);
        g_stats.doa_latest.store(va_doa_bf_get_doa(g_algo.bf), std::memory_order_relaxed);
        for (int i = 0; i < kBlock; i++) {
            float v = mono[i];
            v = v < -1.f ? -1.f : (v > 1.f ? 1.f : v);
            s16[i] = (int16_t)lrintf(v * 32767.f);
        }
        double t1 = now_s();
        win_compute += t1 - t0;
        win_audio += (double)kBlock / kRateEng;
        g_stats.aec_blocks++;

        // mic 输出：fifo（O_NONBLOCK；无读者 → 丢弃计数）或 shm 环（覆盖+多读者）
        if (g_cfg.ipc == IpcMode::FIFO) {
            g_mic_fifo.try_open(g_cfg.mic_fifo_path);
            size_t dropped = g_mic_fifo.write_s16(s16, kBlock);
            if (dropped < (size_t)kBlock) g_stats.fifo_written += kBlock - dropped;
            g_stats.fifo_dropped += dropped;
        } else {
            g_mic_shm.publish(s16, kBlock);
            g_stats.shm_mic_written += kBlock;
            g_stats.mic_readers.store(g_mic_shm.readers(), std::memory_order_relaxed);
            if (now_s() - last_reap > 1.0) {          // 注册表心跳回收（1s 周期）
                g_mic_shm.reap();
                g_stats.mic_readers.store(g_mic_shm.readers(), std::memory_order_relaxed);
                last_reap = now_s();
            }
        }

        double now = now_s();
        if (g_cfg.stats_interval > 0 && now - stat_t0 >= g_cfg.stats_interval) {
            double rtf = win_audio > 0 ? win_compute / win_audio : 0.0;
            if (getenv("VSHM_DEBUG_SPK")) { // 各线程 CPU 时间（联调诊断：区分自旋/休眠）
                DIR* d = opendir("/proc/self/task");
                if (d) {
                    struct dirent* de;
                    char buf[512];
                    while ((de = readdir(d)) != nullptr) {
                        if (de->d_name[0] == '.') continue;
                        snprintf(buf, sizeof buf, "/proc/self/task/%s/stat", de->d_name);
                        FILE* f = fopen(buf, "r");
                        if (!f) continue;
                        char comm[32]; long uu = 0, ss = 0;
                        if (fscanf(f, "%*d (%31[^)]) %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %ld %ld",
                                   comm, &uu, &ss) >= 3)
                            fprintf(stderr, "[dbg-thr] tid=%s comm=%s utime=%ld stime=%ld\n",
                                    de->d_name, comm, uu, ss);
                        fclose(f);
                    }
                    closedir(d);
                }
            }
            uint64_t tc[4], tot = 0;
            for (int t = 0; t < 4; t++) { tc[t] = g_stats.tag_counts[t].load(); tot += tc[t]; }
            fprintf(stderr,
                "[stats] t=%.1fs captured_samples=%llu aec_blocks=%llu fifo_written=%llu "
                "fifo_dropped=%llu fifo_read=%llu ref_samples=%llu doa_latest=%.1f rtf=%.3f "
                "tag=[%.1f %.1f %.1f %.1f]%% overrun=%llu ring_ovf=%llu ref_ovf=%llu "
                "ref_padded=%llu ref_discarded=%llu wire_words=%llu reads=%llu min_n=%lld "
                "ipc=%s mic_readers=%d spk_writers=%d mixed=%llu shm_dropped=%llu spk_rtf=%.3f mic_in=%.1fms pb_err=%llu\n",
                now - start,
                (unsigned long long)g_stats.captured_samples.load(),
                (unsigned long long)g_stats.aec_blocks.load(),
                (unsigned long long)g_stats.fifo_written.load(),
                (unsigned long long)g_stats.fifo_dropped.load(),
                (unsigned long long)g_stats.fifo_read.load(),
                (unsigned long long)g_stats.ref_samples.load(),
                (double)g_stats.doa_latest.load(), rtf,
                tot ? 100.0 * tc[0] / tot : 0.0, tot ? 100.0 * tc[1] / tot : 0.0,
                tot ? 100.0 * tc[2] / tot : 0.0, tot ? 100.0 * tc[3] / tot : 0.0,
                (unsigned long long)g_stats.overrun.load(),
                (unsigned long long)g_stats.ring_overflow.load(),
                (unsigned long long)g_stats.ref_overflow.load(),
                (unsigned long long)g_stats.ref_padded.load(),
                (unsigned long long)g_stats.ref_discarded.load(),
                (unsigned long long)g_stats.wire_words.load(),
                (unsigned long long)g_stats.cap_reads.load(),
                (long long)g_stats.cap_min_n.load(),
                g_cfg.ipc == IpcMode::FIFO ? "fifo" : "shm",
                g_stats.mic_readers.load(),
                g_stats.spk_writers.load(),
                (unsigned long long)g_stats.mixed_frames.load(),
                (unsigned long long)g_stats.shm_dropped.load(),
                (double)g_stats.spk_path_rtf.load(),
                (double)g_stats.mic_in_ms.load(),
                (unsigned long long)g_stats.pb_errors.load());
            stat_t0 = now; win_compute = win_audio = 0;
        }
    }
    if (ref_dump) { fclose(ref_dump); fprintf(stderr, "[dump] ref 已落盘 %s\n", ref_dump_path.c_str()); }
    if (mic_dump) { fclose(mic_dump); fprintf(stderr, "[dump] mic 已落盘 %s\n", mic_dump_path.c_str()); }
    if (aec_dump) { fclose(aec_dump); fprintf(stderr, "[dump] aec 已落盘 %s\n", aec_dump_path.c_str()); }
    if (bf_dump) { fclose(bf_dump); fprintf(stderr, "[dump] bf 已落盘 %s\n", bf_dump_path.c_str()); }
    return nullptr;
}

// ----------------------------------------------------------------------------
// 线程3：回采 voice_spk.fifo 读取（EOF 重开 50ms 退避）→ ref 环 + 可选播放
// ----------------------------------------------------------------------------
static void* spk_thread(void* /*arg*/) {
    pin_cpu(1, "spk");
    g_ref_decim.init();
    snd_pcm_t* pb = nullptr;
    if (g_cfg.playback_dev) {
        int rc = snd_pcm_open(&pb, g_cfg.playback_dev, SND_PCM_STREAM_PLAYBACK, 0);
        if (rc >= 0) {
            rc = pcm_hw_setup(pb, SND_PCM_FORMAT_S16_LE, 2, kRateMic, 384, 8);  // 8ms period×8=64ms（IO 8ms 帧设计）
            if (rc >= 0) {
                // start_threshold=1024（=永续水位高线，见 spk_thread shm 分支）：
                // 永续播放模式下每轮补到 1024 即启动；EPIPE 自愈后同样蓄到高线
                // 再放行。avail_min=1 period 配套。
                snd_pcm_sw_params_t* swp;
                snd_pcm_sw_params_alloca(&swp);
                if (snd_pcm_sw_params_current(pb, swp) >= 0 &&
                    snd_pcm_sw_params_set_start_threshold(pb, swp, 1024) >= 0 &&
                    snd_pcm_sw_params_set_avail_min(pb, swp, 384) >= 0 &&
                    snd_pcm_sw_params(pb, swp) >= 0)
                    fprintf(stderr, "[playback] sw: start_threshold=1024 avail_min=384\n");
            }
            if (rc < 0) {
                fprintf(stderr, "[playback] set_params %s 失败: %s → 回退 null（丢弃），仅此告警一次\n",
                        g_cfg.playback_dev, snd_strerror(rc));
                snd_pcm_close(pb); pb = nullptr;
            } else {
                fprintf(stderr, "[playback] %s: 48k S16 2ch（speaker 断开写失败会自动回退 null）\n",
                        g_cfg.playback_dev);
            }
        } else {
            fprintf(stderr, "[playback] open %s 失败: %s → 回退 null（丢弃），仅此告警一次\n",
                    g_cfg.playback_dev, snd_strerror(rc));
        }
    }

    // shm 模式：收割/混音结果落盘（--dump-mix，调试/测试用；fifo 模式落原始回采流）
    WavWriter mix_dump;
    if (g_cfg.dump_mix) mix_dump.open(g_cfg.dump_mix, kRateMic);

    static int16_t raw[kSpkChunk];
    static float   mono[kSpkChunk];
    static int16_t stereo[kSpkChunk * 2];
    static float   ref16[kSpkChunk / 3 + 8];
    double stat_t0 = now_s(), busy_acc = 0, audio_acc = 0, last_rtf = -1.0;

    if (g_cfg.ipc == IpcMode::SHM) {
        // ---- shm 收割混音：poll 写者唤醒 fifo → 收割各槽新增 → 求和混音+限幅 ----
        // PCM 写侧水位节拍（修复"一卡一卡"）：写者(aplay×vshm)按槽整块(4096帧=85.3ms)
        // 喂入 → harvest 输出同粒度突发；若整块直写 PCM，填充锯齿 4096→0 锁步零裕量，
        // 任何唤醒抖动即 -EPIPE（实测 24s 播放 129 次欠载≈5.4 次/秒）。
        // 对策：入块囤 stash，每循环按 1024 帧小份补到高水位 6144(128ms) 为止——
        // 稳态锯齿 [2048,6144]，最坏点 42.7ms 抖动裕量；ref 引前 ≤128+85=213ms
        // < AEC 320ms 窗。对任意写者块型稳健。
        static int16_t stash[kSpkChunk * 8];
        size_t stash_n = 0;
        const snd_pcm_uframes_t kSpkPeriodWr = 128;  // 2.7ms 小份写
        // 永续播放水位（2026-10-07 架构）：每轮循环（8ms 兜底节拍）无条件把 PCM
        // 补到 hi=1024(21ms)——stash 空则垫静音。PCM 自启动起永续 RUNNING：
        // 无 start/underrun 状态机（间歇无声的结构性解药）；写者断续只是 PCM
        // 内容断续，流永不断。hi=21ms > 唤醒抖动 p95 10.8ms + 循环间隔 8ms，
        // 数学上不触底（触底才可能 EPIPE）。
        const snd_pcm_sframes_t kSpkWmHi = 1024;
        static float zpad[kSpkChunk];
        const bool spk_trace = getenv("VSHM_SPK_TRACE") != nullptr;  // 逐写时延追踪
        double tr_t0 = now_s(), tr_last_write = 0;
        while (!g_stop.load(std::memory_order_relaxed)) {
            double busy = 0;
            size_t ns = g_spk_shm.harvest(mono, kSpkChunk, &busy);
            g_stats.spk_writers.store(g_spk_shm.writers(), std::memory_order_relaxed);
            busy_acc += busy;
            if (ns > 0) {
                audio_acc += (double)ns / kRateMic;
                for (size_t i = 0; i < ns; i++)
                    raw[i] = (int16_t)lrintf(mono[i] * 32767.f);
                if (g_cfg.dump_mix) mix_dump.write_s16(raw, ns);
                int m = g_ref_decim.process(mono, (int)ns, ref16);    // ① ÷3 → 16k ref
                if (m > 0) {
                    size_t got = g_ref_ring.push(ref16, (size_t)m);
                    if (got < (size_t)m) g_stats.ref_overflow += (size_t)m - got;
                    g_stats.ref_samples += got;
                }
                if (ns > sizeof(stash) / sizeof(stash[0]) - stash_n)   // 防御：超容直写
                    ns = sizeof(stash) / sizeof(stash[0]) - stash_n;
                memcpy(stash + stash_n, raw, ns * sizeof(int16_t));
                stash_n += ns;
            }
            // ② speaker 永续输出：补到 hi——stash 有数据写数据（其 ref 已在收割时
            //    入环），不足垫静音（静音同步过 decim 入环）→ ref 环与 PCM 流
            //    严格同序恒速，管线侧 DRAINED/ref_padded 恒不触发，AEC ref
            //    游标结构性零漂移
            if (pb) {
                snd_pcm_sframes_t del = 0;
                if (snd_pcm_delay(pb, &del) < 0) del = 0;
                while (del < kSpkWmHi) {
                    size_t n = std::min(stash_n, (size_t)kSpkPeriodWr);
                    const bool silence = (n == 0);
                    if (silence)
                        n = (size_t)std::min<snd_pcm_sframes_t>(
                                (snd_pcm_sframes_t)kSpkPeriodWr, kSpkWmHi - del);
                    for (size_t i = 0; i < n; i++)
                        stereo[i * 2] = stereo[i * 2 + 1] =
                            silence ? (int16_t)0 : stash[i];
                    if (spk_trace && !silence)
                        fprintf(stderr, "[spk-trace] t=%.3f gap=%.1fms n=%zu delay=%ld\n",
                                now_s() - tr_t0, (now_s() - tr_last_write) * 1e3, n, (long)del);
                    tr_last_write = now_s();
                    snd_pcm_sframes_t w = snd_pcm_writei(pb, stereo, n);
                    if (w < 0) {
                        g_stats.pb_errors++;
                        if (w == -EPIPE) { snd_pcm_prepare(pb); del = 0; }
                        else { snd_pcm_close(pb); pb = nullptr; g_cfg.playback_dev = nullptr; }
                        break;
                    }
                    if (silence) {          // 静音垫同步 ref 流（恒速关键）
                        int m = g_ref_decim.process(zpad, (int)w, ref16);
                        if (m > 0) {
                            size_t got = g_ref_ring.push(ref16, (size_t)m);
                            if (got < (size_t)m) g_stats.ref_overflow += (size_t)m - got;
                            g_stats.ref_samples += got;
                        }
                    } else {
                        memmove(stash, stash + n, (stash_n - (size_t)w) * sizeof(int16_t));
                        stash_n -= (size_t)w;
                    }
                    del += w;
                }
            }
            double now = now_s();
            if (g_cfg.stats_interval > 0 && now - stat_t0 >= g_cfg.stats_interval) {
                g_stats.spk_path_rtf.store(audio_acc > 0 ? busy_acc / audio_acc : 0.f,
                                           std::memory_order_relaxed);
                stat_t0 = now; busy_acc = audio_acc = 0;
            }
        }
    } else {
        // ---- fifo 旧路径（竞态修复版）：持久 O_RDWR 自持 fd + poll + 非阻塞读 ----
        // 原实现"O_RDONLY 打开→切阻塞读→EOF→close→50ms 退避→重开"存在 50ms 盲窗：
        // 短命写者 open→write→close 全程落在盲窗内时，写者关闭瞬间引擎不持有任何
        // 管道 fd，内核按 POSIX 语义丢弃全部缓冲数据（实测 aplay 0.2s 文件整个生命
        // 仅 ~7ms，24KB 数据整段消失、fifo_read 恒 0；加 strace/日志拖慢循环偶尔能
        // 接住——海森性来源）。改为引擎常持 O_RDWR|O_NONBLOCK（与 wake fifo 同款
        // "永不半开"）：写者关闭后数据仍滞留管道可读，盲窗归零；无数据 = poll 超时/
        // read EAGAIN（"writer 全关"的 EOF 语义改由 ref 环断供→管线 DRAINED 承担，
        // 与 shm 路径一致）。
        const bool dbg = getenv("VSHM_DEBUG_SPK") != nullptr;    // fifo 读循环诊断开关
        // fifo 写者块型不受控（feed_fifo 40ms 突发 / aplay 周期）——与 shm 分支
        // 同款 stash+双水位滞回（2026-10-06 真实声学测试发现：整块直写 PCM 时
        // 填充随写者块型锯齿锁步，60s 播放 pb_err=673 ≈11 次/s；治理后与 shm
        // 路径一致：fill 锁 [12,32]ms 带，写者块型/抖动解耦）
        static int16_t fstash[kSpkChunk * 8];
        size_t fstash_n = 0;
        const snd_pcm_uframes_t kFifoWr = 128;          // 2.7ms 小份写
        const snd_pcm_sframes_t kFifoFloor = 576;       // 12ms 触底水位
        // 高水位须 > 最大写者块间隔（feed_fifo 曾 40ms 块：hi=1536(32ms) < 40ms
        // → 供给间隙必触底，实测 pb_err ~12 次/s；48ms 覆盖 40ms 块型，更细块
        // 型由滞回自然收窄锯齿）
        const snd_pcm_sframes_t kFifoHi = 2304;         // 48ms 放水高水位
        int fd = -1;
        while (!g_stop.load(std::memory_order_relaxed)) {
            if (fd < 0) {
                fd = ::open(g_cfg.spk_fifo_path, O_RDWR | O_NONBLOCK | O_CLOEXEC);
                if (fd < 0) { usleep(50000); continue; }         // fifo 未建/被删：50ms 重试
                if (dbg) fprintf(stderr, "[dbg-spk] open fd=%d（O_RDWR 自持，无盲窗）\n", fd);
            }
            struct pollfd pfd = { fd, POLLIN, 0 };
            poll(&pfd, 1, 50);                                   // 有数据即醒；50ms 兜底（原重开节拍）
            if (g_stop.load(std::memory_order_relaxed)) break;
            double t0 = now_s();
            ssize_t r = ::read(fd, raw, sizeof(raw));
            if (dbg) fprintf(stderr, "[dbg-spk] fd=%d read=%zd errno=%d\n", fd, r, r < 0 ? errno : 0);
            if (r > 0) {
                size_t ns = (size_t)r / 2;
                g_stats.fifo_read += ns;
                for (size_t i = 0; i < ns; i++) mono[i] = raw[i] / 32768.f;
                if (g_cfg.dump_mix) mix_dump.write_s16(raw, ns);
                int m = g_ref_decim.process(mono, (int)ns, ref16);   // ① ÷3 → 16k ref
                if (m > 0) {
                    size_t got = g_ref_ring.push(ref16, (size_t)m);
                    if (got < (size_t)m) g_stats.ref_overflow += (size_t)m - got;
                    g_stats.ref_samples += got;
                }
                if (ns > sizeof(fstash) / sizeof(fstash[0]) - fstash_n)   // 防御：超容截断
                    ns = sizeof(fstash) / sizeof(fstash[0]) - fstash_n;
                memcpy(fstash + fstash_n, raw, ns * sizeof(int16_t));
                fstash_n += ns;
                busy_acc += now_s() - t0;
                audio_acc += (double)ns / kRateMic;
            } else if (r < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
                ::close(fd); fd = -1;                             // 罕见真错误：重建 fd
                usleep(50000);
                continue;
            }
            // ② speaker 输出：触底才放水补到高水位（与 shm 分支同式；写者断流后
            //    del 自然降到 floor 以下，stash 尾部随之放净 = 尾冲刷）
            if (pb && fstash_n > 0) {
                snd_pcm_sframes_t del = 0;
                if (snd_pcm_delay(pb, &del) < 0) del = 0;
                if (del <= kFifoFloor) {
                    while (fstash_n > 0 && del < kFifoHi) {
                        size_t n = std::min(fstash_n, (size_t)kFifoWr);
                        for (size_t i = 0; i < n; i++)
                            stereo[i * 2] = stereo[i * 2 + 1] = fstash[i];
                        snd_pcm_sframes_t w = snd_pcm_writei(pb, stereo, (snd_pcm_uframes_t)n);
                        if (w < 0) {
                            g_stats.pb_errors++;
                            if (w == -EPIPE) { snd_pcm_prepare(pb); del = 0; }
                            else {
                                fprintf(stderr, "[playback] 持续写失败(%s) → 回退 null（丢弃），仅此告警一次\n",
                                        snd_strerror((int)w));
                                snd_pcm_close(pb); pb = nullptr;
                                g_cfg.playback_dev = nullptr;
                            }
                            break;
                        }
                        memmove(fstash, fstash + n, (fstash_n - (size_t)w) * sizeof(int16_t));
                        fstash_n -= (size_t)w;
                        del += w;
                    }
                }
            }
            // 统计窗口改为每 poll 周期关一次（不再依赖"本周期有数据"）：写者断流后
            // spk_rtf 冻结在末次实测值而非永远 -1（旧代码只在 r>0 分支关窗 → 4e fifo
            // 腿 grep 空、被记作"stats 停打"的直接原因）。
            double now = now_s();
            if (g_cfg.stats_interval > 0 && now - stat_t0 >= g_cfg.stats_interval) {
                if (audio_acc > 0) last_rtf = busy_acc / audio_acc;
                g_stats.spk_path_rtf.store((float)last_rtf, std::memory_order_relaxed);
                stat_t0 = now; busy_acc = audio_acc = 0;
            }
        }
        if (fd >= 0) ::close(fd);
    }
    mix_dump.close();
    if (pb) snd_pcm_close(pb);
    fprintf(stderr, "[spk] stopped\n");
    return nullptr;
}

// ----------------------------------------------------------------------------
// 自检：多相抽取器频率响应（无音频硬件也可验证 48k→16k DSP）
// ----------------------------------------------------------------------------
static int run_selftest() {
    PolyDecim3 d;
    d.init();
    static float in[48000], out[48000 / 3 + 8];
    int fails = 0;
    struct { double f; double lo, hi; const char* name; } cases[] = {
        {1000.0, 0.84, 1.19, "1kHz 通带增益 ~0dB"},
        {7000.0, 0.50, 1.19, "7kHz 通带边缘（48 抽头过渡带 -6dB 容限）"},
        {11000.0, 0.0, 0.20, "11kHz 阻带衰减 >14dB"},
        {15000.0, 0.0, 0.10, "15kHz 阻带衰减 >20dB"},
    };
    for (auto& tc : cases) {
        int N = 48000; // 1s
        for (int i = 0; i < N; i++)
            in[i] = 0.5f * sinf(2.f * (float)M_PI * (float)tc.f * (float)i / 48000.f);
        int m = d.process(in, N, out);
        double acc = 0; int cnt = 0;
        for (int i = 16; i < m; i++) { acc += (double)out[i] * out[i]; cnt++; } // 丢暂态
        double rms = sqrt(acc / std::max(cnt, 1));
        double gain = rms / (0.5 / 1.414213562);
        bool ok = gain >= tc.lo && gain <= tc.hi;
        printf("[selftest] %-24s f=%5.0fHz->16k gain=%.3f  %s\n", tc.name, tc.f, gain, ok ? "OK" : "FAIL");
        if (!ok) fails++;
    }
    printf(fails ? "[selftest] FAIL (%d 项)\n" : "[selftest] all OK\n", fails);
    return fails;
}

// ----------------------------------------------------------------------------
// CLI / main
// ----------------------------------------------------------------------------
static void usage(const char* p) {
    fprintf(stderr,
        "voice-engine (%s) — NanoPi Air 4-mic AC108 守护进程\n"
        "用法: %s [选项]\n"
        "  --simulate-capture <4ch16k.wav>  跳过硬件采集，从 WAV 循环、实时节奏读入\n"
        "  --capture-fs <16k|48k>           ADC 采样率（默认 16k：32k 线上流剥 tag 直出，无软件抽取；\n"
        "                                   48k 为旧路径：96k 线上流 → ÷3 抽取。配方分别用\n"
        "                                   ~/ac108/mic4-record16.sh cfg / mic4-record.sh）\n"
        "  --capture-device <pcm>           采集设备（默认 hw:1,0）\n"
        "  --null-playback                  回采仅喂 AEC 不播放（默认）\n"
        "  --device <pcm>                   speaker 播放设备（如 hw:0,0；打开/写失败自动回退 null）\n"
        "  --ipc <shm|fifo>                 应用侧 IPC：shm=共享内存环（默认，多读者/多写者混音，\n"
        "                                   /dev/shm/voice_mic|voice_spk，协议 shm_voice.h，ALSA 设备\n"
        "                                   voice_mic_shm/voice_spk_shm）；fifo=旧管道路径（voice_mic/\n"
        "                                   voice_spk，单读者、并发分流）\n"
        "  --dump-mix <out.wav>             回采路径结果落盘（shm=混音后，fifo=原始流）48k mono S16\n"
        "  --mic-fifo <path>                处理后输出 fifo（默认 /tmp/voice_mic.fifo，仅 --ipc fifo）\n"
        "  --spk-fifo <path>                回采输入 fifo（默认 /tmp/voice_spk.fifo，仅 --ipc fifo）\n"
        "  --stats-interval <sec>           统计打印间隔秒（默认 5，0=关）\n"
        "  --selftest                       多相抽取器自检后退出\n"
        "  --help                           本帮助\n",
        VA_ALGO_DESC, p);
}

int main(int argc, char** argv) {
    signal(SIGPIPE, SIG_IGN); // fifo 半开写 EPIPE 由 write 返回值处理

    for (int i = 1; i < argc; i++) {
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", what); usage(argv[0]); exit(2); }
            return argv[++i];
        };
        if (!strcmp(argv[i], "--simulate-capture")) { g_cfg.simulate = true; g_cfg.sim_wav = next("--simulate-capture"); }
        else if (!strcmp(argv[i], "--capture-fs")) {
            const char* v = next("--capture-fs");
            if (!strcmp(v, "16k") || !strcmp(v, "16000")) g_cfg.capture_fs = 16000;
            else if (!strcmp(v, "48k") || !strcmp(v, "48000")) g_cfg.capture_fs = 48000;
            else { fprintf(stderr, "--capture-fs 只接受 16k 或 48k（得到 %s）\n", v); return 2; }
        }
        else if (!strcmp(argv[i], "--capture-device")) g_cfg.capture_dev = next("--capture-device");
        else if (!strcmp(argv[i], "--device")) g_cfg.playback_dev = next("--device");
        else if (!strcmp(argv[i], "--null-playback")) g_cfg.playback_dev = nullptr;
        else if (!strcmp(argv[i], "--ipc")) {
            const char* v = next("--ipc");
            if (!strcmp(v, "shm")) g_cfg.ipc = IpcMode::SHM;
            else if (!strcmp(v, "fifo")) g_cfg.ipc = IpcMode::FIFO;
            else { fprintf(stderr, "--ipc 只接受 shm 或 fifo（得到 %s）\n", v); return 2; }
        }
        else if (!strcmp(argv[i], "--dump-mix")) g_cfg.dump_mix = next("--dump-mix");
        else if (!strcmp(argv[i], "--mic-order")) {
            const char* v = next("--mic-order");
            int a = -1, b = -1, c = -1, d = -1;
            if (sscanf(v, "%d,%d,%d,%d", &a, &b, &c, &d) == 4 &&
                a >= 0 && a < kNChan && b >= 0 && b < kNChan &&
                c >= 0 && c < kNChan && d >= 0 && d < kNChan &&
                ((1<<a) | (1<<b) | (1<<c) | (1<<d)) == 0x0F) {   // 0..3 各恰一次
                g_cfg.mic_order[0] = a; g_cfg.mic_order[1] = b;
                g_cfg.mic_order[2] = c; g_cfg.mic_order[3] = d;
            } else {
                fprintf(stderr, "--mic-order 需要 0..3 的排列，如 0,3,2,1\n");
                return 2;
            }
        }
        else if (!strcmp(argv[i], "--mic-fifo")) g_cfg.mic_fifo_path = next("--mic-fifo");
        else if (!strcmp(argv[i], "--spk-fifo")) g_cfg.spk_fifo_path = next("--spk-fifo");
        else if (!strcmp(argv[i], "--stats-interval")) g_cfg.stats_interval = atoi(next("--stats-interval"));
        else if (!strcmp(argv[i], "--selftest")) return run_selftest();
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); usage(argv[0]); return 2; }
    }

    fprintf(stderr, "[engine] voice_engine 启动: algo=%s ipc=%s 采集=%s%s %s %s 播放=%s%s\n",
            VA_ALGO_DESC, g_cfg.ipc == IpcMode::SHM ? "shm" : "fifo",
            g_cfg.simulate ? g_cfg.sim_wav : g_cfg.capture_dev,
            g_cfg.simulate ? "" : (g_cfg.capture_fs == 16000 ? " (fs=16k 直采)" : " (fs=48k+÷3)"),
            g_cfg.ipc == IpcMode::SHM ? g_cfg.shm_mic_path : g_cfg.mic_fifo_path,
            g_cfg.ipc == IpcMode::SHM ? g_cfg.shm_spk_path : g_cfg.spk_fifo_path,
            g_cfg.playback_dev ? g_cfg.playback_dev : "null(丢弃)",
            g_cfg.dump_mix ? "（dump-mix 落盘）" : "");

    if (!g_algo.init()) { fprintf(stderr, "[engine] algo init failed\n"); return 1; }

    if (g_cfg.ipc == IpcMode::SHM) {
        if (!g_mic_shm.init(g_cfg.shm_mic_path) || !g_spk_shm.init(g_cfg.shm_spk_path)) return 1;
    } else {
        mkfifo(g_cfg.mic_fifo_path, 0666); // EEXIST 忽略
        mkfifo(g_cfg.spk_fifo_path, 0666);
        chmod(g_cfg.mic_fifo_path, 0666);  // sudo 运行时 umask 会砍掉组/其他位，应用侧打不开
        chmod(g_cfg.spk_fifo_path, 0666);
    }

    struct sigaction sa = {};
    sa.sa_handler = [](int) { g_stop.store(true); };
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    pthread_t t_cap, t_pipe, t_spk;
    void* (*cap_fn)(void*) = g_cfg.simulate ? capture_sim_thread : capture_hw_thread;
    pthread_create(&t_cap, nullptr, cap_fn, nullptr);
    pthread_create(&t_pipe, nullptr, pipeline_thread, nullptr);
    pthread_create(&t_spk, nullptr, spk_thread, nullptr);

    while (!g_stop.load(std::memory_order_relaxed)) pause();
    fprintf(stderr, "[engine] shutting down...\n");
    pthread_join(t_cap, nullptr);
    pthread_join(t_pipe, nullptr);
    pthread_join(t_spk, nullptr);
    g_mic_fifo.close();
    g_mic_shm.deinit();
    g_spk_shm.deinit();
    g_algo.deinit();
    fprintf(stderr, "[engine] bye\n");
    return g_capture_failed ? 3 : 0;
}
