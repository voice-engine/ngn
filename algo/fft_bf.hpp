// afecpp/fft_bf.hpp — 自写 radix-2 迭代 FFT（复数 float），BF/VAD 模块共用。
// 独立文件命名 fft_bf.hpp，避免与其他并行模块（如 aec_kf）的 FFT 实现冲突。

#pragma once
#include <cmath>
#include <complex>
#include <cstddef>
#include <utility>
#include <vector>

namespace afe {

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class FFT_BF {
public:
    explicit FFT_BF(int n) : n_(n), rev_(n), tw_(n > 1 ? n / 2 : 1) {
        int lg = 0;
        while ((1 << lg) < n_) ++lg;
        if ((1 << lg) != n_) lg = -1;  // caller guarantees power of two
        for (int i = 0; i < n_; ++i) {
            int r = 0;
            for (int b = 0; b < lg; ++b)
                if (i & (1 << b)) r |= 1 << (lg - 1 - b);
            rev_[i] = r;
        }
        for (int k = 0; k < (int)tw_.size(); ++k) {
            double a = -2.0 * 3.14159265358979323846 * k / n_;
            tw_[k] = std::complex<float>((float)std::cos(a), (float)std::sin(a));
        }
    }

    int size() const { return n_; }

    // 原地正向 DFT（未归一化）
    void forward(std::vector<std::complex<float>>& a) const { run(a); }

    // 原地逆向 DFT（除以 n 归一化，输出共轭对称时为实信号）
    void inverse(std::vector<std::complex<float>>& a) const {
        for (auto& z : a) z = std::conj(z);
        run(a);
        float s = 1.0f / (float)n_;
        for (auto& z : a) z = std::conj(z) * s;
    }

    static int next_pow2(int n) {
        int p = 1;
        while (p < n) p <<= 1;
        return p;
    }

private:
    void run(std::vector<std::complex<float>>& a) const {
        for (int i = 0; i < n_; ++i)
            if (i < rev_[i]) std::swap(a[i], a[rev_[i]]);
        if (n_ >= 2) {  // 末级 len=2：twiddle 恒为 1，省去复乘
            for (int i = 0; i + 1 < n_; i += 2) {
                std::complex<float> u = a[i], v = a[i + 1];
                a[i] = u + v;
                a[i + 1] = u - v;
            }
        }
        for (int len = 4; len <= n_; len <<= 1) {
            int half = len >> 1, step = n_ / len;
            for (int i = 0; i < n_; i += len) {
                const std::complex<float>* tw = tw_.data();
                std::complex<float>* p0 = a.data() + i;
                std::complex<float>* p1 = p0 + half;
                for (int j = 0; j < half; ++j) {
                    std::complex<float> w = tw[j * step];
                    std::complex<float> u = p0[j];
                    std::complex<float> v = p1[j] * w;
                    p0[j] = u + v;
                    p1[j] = u - v;
                }
            }
        }
    }

    int n_;
    std::vector<int> rev_;
    std::vector<std::complex<float>> tw_;
};

}  // namespace afe
