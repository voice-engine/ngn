// GRU 微基准：ncnn GRU 层 vs 手写 gru_step（GTCRN 实际形态）
// 用法: ./gru_bench_ncnn model.ncnn.param model.ncnn.bin
#include <net.h>

#include <sys/time.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static double now_s() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s param bin\n", argv[0]); return 1; }
    setvbuf(stdout, nullptr, _IONBF, 0);
    ncnn::Net net;
    net.opt.num_threads = 1;
    net.opt.use_winograd_convolution = false;
    net.opt.use_sgemm_convolution = false;
    if (net.load_param(argv[1]) || net.load_model(argv[2])) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    // 输入 (1,33,8) → Mat(w=8,h=33,c=1)
    ncnn::Mat in(8, 33, 1);
    float* d = in;
    for (int i = 0; i < 8 * 33; i++) d[i] = 0.01f * (i % 17);

    ncnn::Mat out;
    // 预热
    for (int i = 0; i < 20; i++) {
        ncnn::Extractor ex = net.create_extractor();
        ex.input("x", in);
        ex.extract("y", out);
    }
    const int N = 1000;
    double t0 = now_s();
    for (int i = 0; i < N; i++) {
        ncnn::Extractor ex = net.create_extractor();
        ex.input("x", in);
        ex.extract("y", out);
    }
    double dt = (now_s() - t0) / N;
    printf("ncnn: %.1f us/次 (33 步 GRU)\n", dt * 1e6);
    (void)out;
    return 0;
}
