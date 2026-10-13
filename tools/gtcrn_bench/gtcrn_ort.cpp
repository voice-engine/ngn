// GTCRN 流式 ONNX 基准：逐帧推理（与 ngn 引擎将来的接入形态一致）
//   读 16k mono PCM16 wav → sqrt-Hann STFT(512/256, center) → 每帧 Run() →
//   WOLA 合成 → 写 wav + 每帧耗时统计（可绑指定 CPU 核）
// 用法: ./gtcrn_ort model.onnx in.wav out.wav [cpu]
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sched.h>
#include <sys/time.h>
#include <vector>

static double now_s() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// ---------- 极简 wav I/O（PCM16 mono） ----------
static std::vector<float> wav_read(const char* path, int* sr) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", path); exit(1); }
    char h[44];
    if (fread(h, 1, 44, f) != 44 || memcmp(h, "RIFF", 4) || memcmp(h + 8, "WAVE", 4)) {
        fprintf(stderr, "%s: not a plain 44-byte-header wav\n", path); exit(1);
    }
    *sr = *(int32_t*)(h + 24);
    int32_t data_len = *(int32_t*)(h + 40);
    int n = data_len / 2;
    std::vector<int16_t> pcm(n);
    if (fread(pcm.data(), 2, n, f) != (size_t)n) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f);
    std::vector<float> x(n);
    for (int i = 0; i < n; i++) x[i] = pcm[i] / 32768.f;
    return x;
}

static void wav_write(const char* path, const std::vector<float>& x, int sr) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "open %s for write failed\n", path); exit(1); }
    int32_t n = (int32_t)x.size(), byte_len = n * 2;
    int32_t hdr[11] = {0x46464952, 36 + byte_len, 0x45564157, 0x20746d66,
                       16, 0x00010001, sr, sr * 2, 0x00100002, 0x61746164, byte_len};
    fwrite(hdr, 4, 11, f);
    for (float s : x) {
        float v = std::max(-1.f, std::min(1.f, s));
        int16_t q = (int16_t)lrintf(v * 32767.f);
        fwrite(&q, 2, 1, f);
    }
    fclose(f);
}

// ---------- 512 点 radix-2 复 FFT（基准够用；ngn 集成时换 algo 库半谱 FFT） ----------
struct FFT {
    int n;
    std::vector<uint32_t> rev;
    std::vector<float> wr, wi;
    FFT(int n) : n(n), rev(n), wr(n), wi(n) {
        int lg = 0;
        while ((1 << lg) < n) lg++;
        for (int i = 0; i < n; i++) rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (lg - 1));
        for (int i = 0; i < n; i++) {
            wr[i] = cosf(-2.f * (float)M_PI * i / n);
            wi[i] = sinf(-2.f * (float)M_PI * i / n);
        }
    }
    void run(float* re, float* im, bool inverse) const {
        for (int i = 0; i < n; i++) {
            uint32_t j = rev[i];
            if (i < (int)j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
        }
        for (int len = 2; len <= n; len <<= 1) {
            int step = n / len;
            for (int i = 0; i < n; i += len) {
                for (int k = 0; k < len / 2; k++) {
                    int tw = k * step;
                    float c = wr[tw], s = inverse ? -wi[tw] : wi[tw];
                    float ur = re[i + k], ui = im[i + k];
                    float vr = re[i + k + len / 2] * c - im[i + k + len / 2] * s;
                    float vi = re[i + k + len / 2] * s + im[i + k + len / 2] * c;
                    re[i + k] = ur + vr; im[i + k] = ui + vi;
                    re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
                }
            }
        }
        if (inverse) {
            for (int i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
        }
    }
};

static void stats(const char* tag, std::vector<double>& s) {
    std::sort(s.begin(), s.end());
    double mean = 0;
    for (double v : s) mean += v;
    mean /= s.size();
    printf("%-8s us: mean %6.0f  p50 %6.0f  p99 %6.0f  max %6.0f   RTF %.3f\n",
           tag, mean, s[s.size() / 2], s[(size_t)(s.size() * 0.99)], s.back(), mean / 16000.0);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.onnx in.wav out.wav [cpu]\n", argv[0]);
        return 1;
    }
    if (argc > 4) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(atoi(argv[4]), &set);
        if (sched_setaffinity(0, sizeof(set), &set) == 0) printf("pinned to cpu%s\n", argv[4]);
    }

    int sr = 0;
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<float> x = wav_read(argv[2], &sr);
    if (sr != 16000) { fprintf(stderr, "need 16k wav, got %d\n", sr); return 1; }
    int nx = (int)x.size();
    int T = nx / 256 + 1;  // center=True 帧数

    float win[512];
    for (int i = 0; i < 512; i++) win[i] = sqrtf(0.5f * (1.f - cosf(2.f * (float)M_PI * i / 512.f)));

    // 前后各 256 零 pad 的完整信号 + WOLA 累加缓冲
    std::vector<float> buf(nx + 512 + 1024, 0.f);
    memcpy(buf.data() + 256, x.data(), nx * 4);
    std::vector<float> ola(nx + 1024, 0.f);

    Ort::Env env(ORT_LOGGING_LEVEL_ERROR);
    Ort::SessionOptions opt;
    opt.SetIntraOpNumThreads(1);
    opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session session(env, argv[1], opt);
    Ort::AllocatorWithDefaultOptions alloc;
    for (size_t i = 0; i < session.GetInputCount(); i++) {
        auto info = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
        auto s = info.GetShape();
        printf("in [%s]: ", session.GetInputNameAllocated(i, alloc).get());
        for (auto d : s) printf("%lld ", (long long)d);
        printf("\n");
    }
    for (size_t i = 0; i < session.GetOutputCount(); i++) {
        auto info = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
        auto s = info.GetShape();
        printf("out[%s]: ", session.GetOutputNameAllocated(i, alloc).get());
        for (auto d : s) printf("%lld ", (long long)d);
        printf("\n");
    }

    // 输入/缓存缓冲（输出缓存拷回即下帧输入，实现状态机）
    const int F = 257;
    std::vector<float> mix(257 * 2), enh(257 * 2);
    std::vector<float> conv_c(2 * 16 * 16 * 33), tra_c(2 * 3 * 16), inter_c(2 * 33 * 16);

    std::vector<std::vector<int64_t>> dims = {{1, F, 1, 2}, {2, 1, 16, 16, 33},
                                              {2, 3, 1, 1, 16}, {2, 1, 33, 16}};
    float* bufs[4] = {mix.data(), conv_c.data(), tra_c.data(), inter_c.data()};
    size_t cnt[4] = {mix.size(), conv_c.size(), tra_c.size(), inter_c.size()};
    Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);
    std::vector<Ort::Value> in;
    for (int i = 0; i < 4; i++)
        in.push_back(Ort::Value::CreateTensor<float>(mi, bufs[i], cnt[i],
                                                     dims[i].data(), dims[i].size()));
    const char* names[] = {"mix", "conv_cache", "tra_cache", "inter_cache"};
    const char* onames[] = {"enh", "conv_cache_out", "tra_cache_out", "inter_cache_out"};

    FFT fft(512);
    float re[512], im[512];
    std::vector<double> t_model(T), t_frame(T);

    for (int t = 0; t < T; t++) {
        double t0 = now_s();
        for (int i = 0; i < 512; i++) { re[i] = buf[t * 256 + i] * win[i]; im[i] = 0.f; }
        fft.run(re, im, false);
        for (int f = 0; f < F; f++) { mix[f * 2] = re[f]; mix[f * 2 + 1] = im[f]; }

        double tm0 = now_s();
        std::vector<Ort::Value> outs;
        try {
            auto r = session.Run(Ort::RunOptions{nullptr}, names, in.data(), 4, onames, 4);
            outs = std::move(r);
        } catch (const Ort::Exception& e) {
            printf("Run failed at frame %d: %s\n", t, e.what());
            return 1;
        }
        t_model[t] = (now_s() - tm0) * 1e6;

        if (t == 0) {
            for (size_t k = 0; k < outs.size(); k++) {
                auto ti = outs[k].GetTensorTypeAndShapeInfo();
                printf("out%zu [%s]: ", k, onames[k]);
                for (auto d : ti.GetShape()) printf("%lld ", (long long)d);
                printf(" count=%zu\n", (size_t)ti.GetElementCount());
            }
        }

        memcpy(enh.data(), outs[0].GetTensorMutableData<float>(), enh.size() * 4);
        memcpy(conv_c.data(), outs[1].GetTensorMutableData<float>(), conv_c.size() * 4);
        memcpy(tra_c.data(), outs[2].GetTensorMutableData<float>(), tra_c.size() * 4);
        memcpy(inter_c.data(), outs[3].GetTensorMutableData<float>(), inter_c.size() * 4);

        for (int f = 0; f < F; f++) {
            re[f] = enh[f * 2]; im[f] = enh[f * 2 + 1];
            if (f > 0 && f < 256) { re[512 - f] = re[f]; im[512 - f] = -im[f]; }
        }
        im[0] = im[256] = 0.f;
        fft.run(re, im, true);
        int base = t * 256;
        for (int i = 0; i < 512; i++) ola[base + i] += re[i] * win[i];
        t_frame[t] = (now_s() - t0) * 1e6;
    }

    std::vector<float> y(ola.begin() + 256, ola.begin() + 256 + nx);
    wav_write(argv[3], y, 16000);

    printf("frames %d (%.2fs audio), hop 256 = 16ms budget\n", T, nx / 16000.0);
    stats("model", t_model);
    stats("frame", t_frame);  // 含 STFT+iSTFT+拷贝
    printf("wrote %s\n", argv[3]);
    return 0;
}
