// GTCRN ncnn 基准：整段离线推理（与 ORT 对比同口径）
//   读 16k mono wav → STFT → (1,257,T,2) → ncnn 前向 → iSTFT/WOLA → wav + 计时
// 用法: ./gtcrn_ncnn model.ncnn.param model.ncnn.bin in.wav out.wav [threads]
#include <net.h>

#include <sys/time.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static double now_s() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static std::vector<float> wav_read(const char* path, int* sr) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", path); exit(1); }
    char h[44];
    if (fread(h, 1, 44, f) != 44 || memcmp(h, "RIFF", 4) || memcmp(h + 8, "WAVE", 4)) {
        fprintf(stderr, "%s: not plain wav\n", path); exit(1);
    }
    *sr = *(int32_t*)(h + 24);
    int n = *(int32_t*)(h + 40) / 2;
    std::vector<int16_t> pcm(n);
    if (fread(pcm.data(), 2, n, f) != (size_t)n) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f);
    std::vector<float> x(n);
    for (int i = 0; i < n; i++) x[i] = pcm[i] / 32768.f;
    return x;
}

static void wav_write(const char* path, const float* x, int n, int sr) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "write %s failed\n", path); exit(1); }
    int32_t bl = n * 2;
    int32_t hdr[11] = {0x46464952, 36 + bl, 0x45564157, 0x20746d66, 16, 0x00010001,
                       sr, sr * 2, 0x00100002, 0x61746164, bl};
    fwrite(hdr, 4, 11, f);
    for (int i = 0; i < n; i++) {
        float v = std::max(-1.f, std::min(1.f, x[i]));
        int16_t q = (int16_t)lrintf(v * 32767.f);
        fwrite(&q, 2, 1, f);
    }
    fclose(f);
}

struct FFT {
    int n;
    std::vector<uint32_t> rev;
    std::vector<float> wr, wi;
    FFT(int n) : n(n), rev(n), wr(n), wi(n) {
        int lg = 0;
        while ((1 << lg) < n) lg++;
        rev[0] = 0;
        for (int i = 1; i < n; i++)
            rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (lg - 1));
        for (int i = 0; i < n; i++) {
            wr[i] = cosf(-2.f * (float)M_PI * i / n);
            wi[i] = sinf(-2.f * (float)M_PI * i / n);
        }
    }
    void run(float* re, float* im, bool inv) const {
        for (int i = 0; i < n; i++) {
            uint32_t j = rev[i];
            if ((uint32_t)i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
        }
        for (int len = 2; len <= n; len <<= 1) {
            int step = n / len;
            for (int i = 0; i < n; i += len)
                for (int k = 0; k < len / 2; k++) {
                    int tw = k * step;
                    float c = wr[tw], s = inv ? -wi[tw] : wi[tw];
                    float ur = re[i + k], ui = im[i + k];
                    float vr = re[i + k + len / 2] * c - im[i + k + len / 2] * s;
                    float vi = re[i + k + len / 2] * s + im[i + k + len / 2] * c;
                    re[i + k] = ur + vr; im[i + k] = ui + vi;
                    re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
                }
        }
        if (inv)
            for (int i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
    }
};

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s model.param model.bin in.wav out.wav [threads]\n", argv[0]);
        return 1;
    }
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 5) {  // 可选绑核：环境 TASKSET/TASKSET_CPU 由调用方处理
    }

    int threads = argc > 5 ? atoi(argv[5]) : 2;
    int sr = 0;
    std::vector<float> x = wav_read(argv[3], &sr);
    if (sr != 16000) { fprintf(stderr, "need 16k\n"); return 1; }
    int nx = (int)x.size();
    int T = nx / 256 + 1;            // torch.stft center 语义
    int padded = T * 256;
    x.resize(padded + 512, 0.f);     // pad + WOLA 余量

    float win[512];
    for (int i = 0; i < 512; i++) win[i] = sqrtf(0.5f * (1.f - cosf(2.f * (float)M_PI * i / 512.f)));

    // STFT 全帧 → spec[t][f][2]（本地布局）
    std::vector<float> spec((size_t)T * 257 * 2);
    FFT fft(512);
    float re[512], im[512];
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < 512; i++) { re[i] = x[t * 256 + i] * win[i]; im[i] = 0.f; }
        fft.run(re, im, false);
        float* sp = &spec[(size_t)t * 257 * 2];
        for (int f = 0; f < 257; f++) { sp[2 * f] = re[f]; sp[2 * f + 1] = im[f]; }
    }

    // 输入 Mat：torch (1,257,T,2) → ncnn Mat(w=2, h=T, d=257, c=1)，内存序 f-major
    ncnn::Mat in(2, T, 257, 1);
    {
        float* d = in;
        for (int f = 0; f < 257; f++)
            for (int t = 0; t < T; t++) {
                const float* sp = &spec[(size_t)t * 257 * 2];
                *d++ = sp[2 * f];
                *d++ = sp[2 * f + 1];
            }
    }

    ncnn::Net net;
    net.opt.num_threads = threads;
    net.opt.use_winograd_convolution = true;
    net.opt.use_sgemm_convolution = true;
    if (net.load_param(argv[1]) || net.load_model(argv[2])) {
        fprintf(stderr, "load model failed\n");
        return 1;
    }

    if (argc > 6) {  // 调试：逐 blob 探测，找第一个 extract 失败的层
        FILE* bf = fopen(argv[6], "r");
        char name[128];
        while (fgets(name, sizeof name, bf)) {
            name[strcspn(name, "\n")] = 0;
            if (!name[0]) continue;
            ncnn::Extractor ex2 = net.create_extractor();
            ex2.input("in0", in);
            ncnn::Mat om;
            int r = ex2.extract(name, om);
            printf("blob %-20s %s\n", name, r == 0 ? "OK" : "FAIL");
        }
        fclose(bf);
        return 0;
    }
    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in);
    ncnn::Mat out;
    double t0 = now_s();
    if (ex.extract("out0", out)) { fprintf(stderr, "extract failed\n"); return 1; }
    double dt = now_s() - t0;
    printf("audio %.2fs (%d frames)  ncnn %.3fs  rtf %.3f  per-frame %.0f us\n",
           nx / 16000.0, T, dt, dt / (nx / 16000.0), dt / T * 1e6);

    // 输出 (1,257,T,2) → (t,f) → iFFT → WOLA
    std::vector<float> ola(padded + 512, 0.f);
    const float* o = out;
    for (int t = 0; t < T; t++) {
        for (int f = 0; f < 257; f++) {
            re[f] = o[(size_t)f * T * 2 + 2 * t];
            im[f] = o[(size_t)f * T * 2 + 2 * t + 1];
            if (f > 0 && f < 256) { re[512 - f] = re[f]; im[512 - f] = -im[f]; }
        }
        im[0] = im[256] = 0.f;
        fft.run(re, im, true);
        for (int i = 0; i < 512; i++) ola[t * 256 + i] += re[i] * win[i];
    }
    wav_write(argv[4], ola.data() + 256, nx, sr);
    printf("wrote %s\n", argv[4]);
    return 0;
}
