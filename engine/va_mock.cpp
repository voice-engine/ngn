// va_mock.cpp - mock implementation of the voice_algo interface.
// Built only when the real /home/i/voice/algo/libvoice_algo.so is absent
// (Makefile adds -DVA_MOCK). Semantics per task spec:
//   mock AEC : out[0] <- mic[0] (ref ignored)
//   mock BF  : out <- ch0 (in[0]) of planar 4ch input
//   mock DOA : constant 0
// 签名与真库 voice_algo.h 对齐（平面指针数组）。
#include "va_mock.h"
#include <string.h>

struct va_aec { int rate; int shared; };
struct va_aec_ref { int rate; };
struct va_doa_bf { int nchan; int rate; };

extern "C" {

va_aec* va_aec_create(int sr) {
    va_aec* a = new va_aec();
    a->rate = sr;
    return a;
}

int va_aec_process(va_aec* aec, const float* const* mic, const float* /*ref*/,
                   float* const* out, int n) {
    (void)aec;
    if (!mic || !out || n <= 0 || !mic[0] || !out[0]) return -1;
    memcpy(out[0], mic[0], (size_t)n * sizeof(float));
    return 0;
}

void va_aec_destroy(va_aec* aec) { delete aec; }

va_aec_ref* va_aec_ref_create(int sr) {
    if (sr != 16000) return nullptr;
    va_aec_ref* r = new va_aec_ref();
    r->rate = sr;
    return r;
}
void va_aec_ref_destroy(va_aec_ref* r) { delete r; }
int va_aec_ref_push(va_aec_ref* r, const float* ref, int n) {
    if (!r || !ref || n <= 0 || n % 512) return -1;
    return 0;   // mock：谱服务无操作
}
va_aec* va_aec_create_shared(int sr, va_aec_ref* r) {
    if (sr != 16000 || !r || r->rate != sr) return nullptr;
    va_aec* a = new va_aec();
    a->rate = sr;
    a->shared = 1;
    return a;
}
int va_aec_process_shared(va_aec* aec, const float* mic, const float* /*ref*/,
                          float* out, int n) {
    if (!aec || !aec->shared || !mic || !out || n <= 0 || n % 512) return -1;
    memcpy(out, mic, (size_t)n * sizeof(float));   // mock AEC：out = mic
    return 0;
}

va_doa_bf* va_doa_bf_create(int nch, int sr, const char* /*array_type*/, double /*geom*/) {
    va_doa_bf* b = new va_doa_bf();
    b->nchan = nch;
    b->rate = sr;
    return b;
}

int va_doa_bf_process(va_doa_bf* bf, const float* const* in, float* out, int n) {
    va_doa_bf* b = bf;
    if (!b || !in || !out || n <= 0 || !in[0]) return -1;
    const int nc = b->nchan > 0 ? b->nchan : 4;
    for (int c = 1; c < nc; c++) (void)in[c]; // 平面布局：各通道独立缓冲
    memcpy(out, in[0], (size_t)n * sizeof(float)); // ch0 copy
    return 0;
}

float va_doa_bf_get_doa(const va_doa_bf* /*bf*/) { return 0.0f; }

void va_doa_bf_destroy(va_doa_bf* bf) { delete bf; }

} // extern "C"
