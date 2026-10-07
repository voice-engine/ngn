// repro_aec.cpp — 离线复现：单通道 va_aec 吃 mic + ref，512/块流式
// delay = ref 内容相对 mic 的滞后样本数（引擎里 ≈ 引导缓冲 2400 + 管线延迟）
// 用法: repro_aec <mic.f32> <ref.f32> <out.f32> [delay]
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "voice_algo.h"
int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s mic ref out [delay>=0]\n", argv[0]); return 2; }
    long delay = argc > 4 ? atol(argv[4]) : 0;
    FILE* fm = fopen(argv[1], "rb"); FILE* fr = fopen(argv[2], "rb");
    if (!fm || !fr) { perror("open"); return 2; }
    fseek(fm, 0, SEEK_END); long nm = ftell(fm)/4; fseek(fm, 0, SEEK_SET);
    fseek(fr, 0, SEEK_END); long nr = ftell(fr)/4; fseek(fr, 0, SEEK_SET);
    std::vector<float> mic(nm), ref(nr);
    if (fread(mic.data(),4,nm,fm)!=(size_t)nm || fread(ref.data(),4,nr,fr)!=(size_t)nr) { perror("read"); return 2; }
    std::vector<float> out(nm, 0.f);
    va_aec* aec = va_aec_create(16000);
    if (!aec) { fprintf(stderr, "create failed\n"); return 1; }
    const long B = 512;
    long done = 0;
    for (long i = 0; i + B <= nm; i += B) {
        long ri = i + delay;                    // ref 对应内容在 ref 数组中滞后 delay 出现
        if (ri + B > nr) break;
        const float* mp[1] = { mic.data()+i };
        float* op[1] = { out.data()+i };
        va_aec_process(aec, mp, ref.data()+ri, op, (int)B);
        done = i + B;
    }
    va_aec_destroy(aec);
    FILE* fo = fopen(argv[3], "wb"); fwrite(out.data(),4,done,fo); fclose(fo);
    fprintf(stderr, "processed %ld samples, delay=%ld\n", done, delay);
    return 0;
}
