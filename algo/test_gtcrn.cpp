// test_gtcrn — GTCRN C 实现单测：wav 进出 + 逐帧计时 + 可选中间量 dump
//
// 用法: ./test_gtcrn in.wav out.wav [块样本数(默认512)]
//   in.wav: 16k mono PCM16（44 字节头）
// env: VA_GTCRN_DUMP=目录 VA_GTCRN_DUMP_FRAMES=N（默认 8）→ 落 f%03d_阶段.bin
//
// 输出对齐说明：末尾补 256 零（对齐 torch.stft center），输出裁回输入长度；
// 首尾各 256 样本含 WOLA 边界效应（torch.istft 会做边缘归一化），对拍跳过。
#include "gtcrn.h"

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
        fprintf(stderr, "%s: 需要 44 字节头的 PCM16 wav\n", path);
        exit(1);
    }
    *sr = *(int32_t*)(h + 24);
    int32_t len = *(int32_t*)(h + 40);
    int n = len / 2;
    std::vector<int16_t> pcm(n);
    if (fread(pcm.data(), 2, n, f) != (size_t)n) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f);
    std::vector<float> x(n);
    for (int i = 0; i < n; i++) x[i] = pcm[i] / 32768.f;
    return x;
}

static void wav_write(const char* path, const float* x, int n, int sr) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "open %s for write failed\n", path); exit(1); }
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

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s in.wav out.wav [block_samples]\n", argv[0]);
        return 1;
    }
    setvbuf(stdout, nullptr, _IONBF, 0);
    int sr = 0;
    std::vector<float> x = wav_read(argv[1], &sr);
    if (sr != 16000) { fprintf(stderr, "need 16k wav, got %d\n", sr); return 1; }
    int block = argc > 3 ? atoi(argv[3]) : 512;
    if (256 % block == 0 || block % 256) {  // 只允许多个 256
        if (block % 256) { fprintf(stderr, "block 须为 256 的倍数\n"); return 1; }
    }

    // 末尾补 256 零对齐 torch center 语义（帧数 = nx/256 + 1）
    int nx = (int)x.size();
    int padded = (nx + 255) / 256 * 256 + 256;
    x.resize(padded, 0.f);
    int T = padded / 256;

    va_gtcrn* g = va_gtcrn_create();
    std::vector<float> y(padded);
    std::vector<double> t_us(T);

    double t0 = now_s();
    for (int off = 0, fr = 0; off < padded; off += block) {
        double f0 = now_s();
        va_gtcrn_process(g, x.data() + off, y.data() + off, block);
        int nf = block / 256;
        double dt = (now_s() - f0) * 1e6;
        for (int k = 0; k < nf && fr < T; k++, fr++) t_us[fr] = dt / nf;
    }
    double total = now_s() - t0;
    (void)total;

    // y 流相对输入滞后 1 帧（16ms WOLA 流水延迟；模型因果窗另含 1 帧，
    // 总算法延迟 32ms）。写 wav 时前移 256 对齐时间轴便于对拍。
    wav_write(argv[2], y.data() + 256, nx, sr);

    std::vector<double> s = t_us;
    std::sort(s.begin(), s.end());
    double mean = 0;
    for (double v : s) mean += v;
    mean /= s.size();
    printf("frames %d (%.2fs)  block %d\n", T, nx / 16000.0, block);
    printf("per-frame us: mean %7.0f  p50 %7.0f  p99 %7.0f  max %7.0f   RTF %.3f\n",
           mean, s[s.size() / 2], s[(size_t)(s.size() * 0.99)], s.back(), mean / 16000.0);
#ifdef VA_PROFILE
    extern void va_gtcrn_prof_print(void);
    va_gtcrn_prof_print();
#endif
    va_gtcrn_destroy(g);
    return 0;
}
