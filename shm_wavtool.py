#!/usr/bin/env python3
"""Dependency-free helper for the SHM IPC data-flow tests (test_shm.sh).
Pure stdlib (no numpy on the board). Subcommands:

  gen FILE RATE DUR_S FREQ AMP        - sine wav S16_LE mono (amp 0..1)
  info FILE                           - wav params as key=value
  frames FILE                         - print frame count
  extractch IN OUT CH                 - extract one channel to mono wav
  rms FILE MIN_RMS                    - rms/peak; exit 1 if rms < MIN_RMS
  lagfind A B EXPECT_S TOL_S          - find sample lag of B vs A near EXPECT_S
                                       (coarse decimated search + full-rate refine);
                                       prints "lag=<frames> corr=<best>"; exit 1 if
                                       best normalized corr < 0.95
  cmpwin A B LAG N [START]            - bitwise compare B[START+i] == A[LAG+START+i]
                                       for N frames; exit 1 on any mismatch
  corrwin A B LAG N [START]           - Pearson corr over the same window; exit 1 if < 0.999
  fftpeaks FILE T0_S T1_S F...        - radix-2 FFT (Hann) of mono window [T0,T1];
                                       decimated to ~6kHz. Verifies every expected F has
                                       a peak >= -9dB of max within +-4 Hz and no other
                                       peak above -20dB of max. Prints peak table.
"""
import array
import cmath
import math
import struct
import sys
import wave


def load_wav(path):
    w = wave.open(path, "rb")
    if w.getsampwidth() != 2:
        raise SystemExit("need 16-bit pcm wav: %s" % path)
    nch, rate = w.getnchannels(), w.getframerate()
    data = w.readframes(w.getnframes())
    w.close()
    s = array.array("h")
    s.frombytes(data)
    if sys.byteorder != "little":
        s.byteswap()
    if nch > 1:  # take channel 0
        s = s[::nch]
    return s, rate


def save_wav(path, samples, rate):
    w = wave.open(path, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(rate)
    w.writeframes(samples.tobytes())
    w.close()


def cmd_gen(args):
    path, rate, dur, freq, amp = args[0], int(args[1]), float(args[2]), float(args[3]), float(args[4])
    n = int(round(rate * dur))
    w = wave.open(path, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(rate)
    pack = struct.Struct("<h").pack
    a = int(amp * 32000.0)
    w.writeframes(b"".join(pack(int(a * math.sin(2.0 * math.pi * freq * i / rate))) for i in range(n)))
    w.close()
    print("gen %s: rate=%d frames=%d freq=%.1f amp=%.2f" % (path, rate, n, freq, amp))


def cmd_info(args):
    w = wave.open(args[0], "rb")
    print("info %s: rate=%d channels=%d sampwidth=%d frames=%d duration=%.3f" % (
        args[0], w.getframerate(), w.getnchannels(), w.getsampwidth(),
        w.getnframes(), w.getnframes() / float(w.getframerate())))
    w.close()


def cmd_frames(args):
    w = wave.open(args[0], "rb")
    print(w.getnframes())
    w.close()


def cmd_extractch(args):
    src, dst, ch = args[0], args[1], int(args[2])
    w = wave.open(src, "rb")
    nch = w.getnchannels()
    s = array.array("h")
    s.frombytes(w.readframes(w.getnframes()))
    rate = w.getframerate()
    w.close()
    if sys.byteorder != "little":
        s.byteswap()
    save_wav(dst, s[ch::nch], rate)
    print("extractch %s[%d] -> %s (rate=%d)" % (src, ch, dst, rate))


def cmd_rms(args):
    path, min_rms = args[0], float(args[1])
    s, _ = load_wav(path)
    acc = 0
    peak = 0
    for v in s:
        acc += v * v
        if abs(v) > peak:
            peak = abs(v)
    rms = math.sqrt(acc / max(len(s), 1))
    ok = rms >= min_rms and peak > 0
    print("rms %s: rms=%.1f peak=%d n=%d (min %.1f) -> %s" % (
        path, rms, peak, len(s), min_rms, "PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


def _norm_corr(a, ai, b, bi, n):
    """normalized correlation of a[ai:ai+n] vs b[bi:bi+n]"""
    if ai < 0 or bi < 0 or ai + n > len(a) or bi + n > len(b):
        return -1.0
    sab = sa = sb = 0.0
    for k in range(n):
        x = a[ai + k]
        y = b[bi + k]
        sab += x * y
        sa += x * x
        sb += y * y
    d = math.sqrt(sa * sb)
    return sab / d if d > 0 else -1.0


def cmd_lagfind(args):
    pa, pb = args[0], args[1]
    expect_s, tol_s = float(args[2]), float(args[3])
    a, rate = load_wav(pa)
    b, rate2 = load_wav(pb)
    if rate != rate2:
        print("lagfind: rate mismatch %d vs %d" % (rate, rate2))
        sys.exit(2)
    # coarse: decimate by 32, correlate a 256-sample (~0.5s) window of B over +-tol
    d = 32
    da, db = a[::d], b[::d]
    wpos = len(db) // 4
    wn = min(256, len(db) - wpos - 1)
    if wn < 64 or len(da) < wn:
        print("lagfind: files too short")
        sys.exit(2)
    cen = int(wpos + expect_s * rate / d)  # wpos 已是抽取索引：期望帧数须同样除以 d
    span = int((tol_s * rate) / d) + 2
    best_c, best_cc = cen, -2.0
    for cand in range(max(0, cen - span), min(len(da) - wn, cen + span)):
        c = _norm_corr(da, cand, db, wpos, wn)
        if c > best_cc:
            best_cc, best_c = c, cand
    # refine at full rate +-2*d around best_c*d
    coarse = best_c * d
    refb = wpos * d
    rn = min(4096, len(b) - refb - 1)
    best, best_corr = coarse, -2.0
    for cand in range(max(0, coarse - 2 * d), min(len(a) - rn, coarse + 2 * d)):
        c = _norm_corr(a, cand, b, refb, rn)
        if c > best_corr:
            best_corr, best = c, cand
    lag = best - refb
    ok = best_corr >= 0.95
    print("lagfind %s vs %s: lag=%d frames (%.3fs @%dHz) corr=%.4f -> %s" % (
        pa, pb, lag, lag / float(rate), rate, best_corr, "PASS" if ok else "FAIL"))
    print("LAG=%d" % lag)
    sys.exit(0 if ok else 1)


def cmd_cmpwin(args):
    pa, pb, lag, n = args[0], args[1], int(args[2]), int(args[3])
    start = int(args[4]) if len(args) > 4 else 0
    a, _ = load_wav(pa)
    b, _ = load_wav(pb)
    n = min(n, len(b) - start, len(a) - lag - start)
    if n <= 0 or lag + start < 0:
        print("cmpwin: empty window")
        sys.exit(2)
    mism = 0
    first = -1
    for k in range(n):
        if b[start + k] != a[lag + start + k]:
            mism += 1
            if first < 0:
                first = k
    ok = mism == 0
    print("cmpwin %s vs %s(lag=%d): %d frames compared, %d mismatches%s -> %s" % (
        pb, pa, lag, n, mism, (" (first@%d)" % first) if first >= 0 else "",
        "PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


def cmd_corrwin(args):
    pa, pb, lag, n = args[0], args[1], int(args[2]), int(args[3])
    start = int(args[4]) if len(args) > 4 else 0
    thresh = float(args[5]) if len(args) > 5 else 0.999
    a, _ = load_wav(pa)
    b, _ = load_wav(pb)
    c = _norm_corr(a, lag + start, b, start, min(n, len(b) - start))
    ok = c >= thresh
    print("corrwin %s vs %s(lag=%d): corr=%.5f (min %.3f) -> %s" % (
        pb, pa, lag, c, thresh, "PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


def _fft(re, im):
    n = len(re)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            re[i], re[j] = re[j], re[i]
            im[i], im[j] = im[j], im[i]
    size = 2
    while size <= n:
        half = size // 2
        step = cmath.exp(-2j * math.pi / size)
        for start in range(0, n, size):
            tw = 1.0 + 0.0j
            for k in range(half):
                a0 = re[start + k] + tw.real * re[start + k + half] - tw.imag * im[start + k + half]
                a1 = im[start + k] + tw.real * im[start + k + half] + tw.imag * re[start + k + half]
                b0 = re[start + k] - tw.real * re[start + k + half] + tw.imag * im[start + k + half]
                b1 = im[start + k] - tw.real * im[start + k + half] - tw.imag * re[start + k + half]
                re[start + k], im[start + k] = a0, a1
                re[start + k + half], im[start + k + half] = b0, b1
                tw *= step
        size <<= 1


def cmd_fftpeaks(args):
    path, t0, t1 = args[0], float(args[1]), float(args[2])
    expected = [float(f) for f in args[3:]]
    s, rate = load_wav(path)
    decim = max(1, rate // 6000)
    ds = s[::decim]
    drate = rate / float(decim)
    i0, i1 = int(t0 * drate), int(t1 * drate)
    n = 8192
    i1 = min(i1, len(ds))
    if i1 - i0 > n:
        i0 = i1 - n
    if i0 < 0 or i1 - i0 < n // 2 or i0 + n > len(ds) + n // 2:
        print("fftpeaks: window too short (need >=%.1fs of audio)" % (n / drate))
        sys.exit(2)
    seg = ds[i0:i0 + n]
    if len(seg) < n:  # 窗口贴文件尾部不足 N：右移补齐
        i0 = max(0, len(ds) - n)
        seg = ds[i0:i0 + n]
    if len(seg) < n // 2:
        print("fftpeaks: file too short")
        sys.exit(2)
    re = [float(v) * (0.5 - 0.5 * math.cos(2 * math.pi * i / (n - 1))) for i, v in enumerate(seg)]
    im = [0.0] * n
    _fft(re, im)
    mags = [math.hypot(re[k], im[k]) for k in range(n // 2)]
    gmax = max(mags[1:], default=0.0) or 1.0
    binhz = drate / n
    lo, hi = int(20 / binhz), int(2900 / binhz)
    peaks = []
    for k in range(lo + 1, min(hi, n // 2 - 1)):
        if mags[k] > mags[k - 1] and mags[k] >= mags[k + 1]:
            peaks.append((k * binhz, mags[k]))
    peaks.sort(key=lambda p: -p[1])
    print("fftpeaks %s [%g,%g]s N=%d @%.0fHz bin=%.2fHz:" % (path, t0, t1, n, drate, binhz))
    for f, m in peaks[:8]:
        print("  peak %8.2f Hz  %6.1f dB rel-max" % (f, 20 * math.log10(m / gmax)))
    ok = True
    msgs = []
    for ef in expected:
        cand = [(f, m) for f, m in peaks if abs(f - ef) <= 4.0]
        if not cand or cand[0][1] < gmax * 10 ** (-9 / 20.0):
            ok = False
            msgs.append("expected %.1f Hz peak missing/weak" % ef)
    for f, m in peaks:
        if m < gmax * 10 ** (-20 / 20.0):
            continue
        if all(abs(f - ef) > 4.0 for ef in expected):
            ok = False
            msgs.append("spurious peak %.1f Hz (%.1f dB)" % (f, 20 * math.log10(m / gmax)))
    if len(expected) >= 2:
        top = [next((m for f, m in peaks if abs(f - ef) <= 4.0), 0.0) for ef in expected]
        if min(top) > 0 and max(top) / min(top) > 10 ** (3 / 20.0):
            ok = False
            msgs.append("expected peaks differ >3dB: %s" % ["%.0f" % t for t in top])
    print("  -> %s%s" % ("PASS" if ok else "FAIL", "; " + "; ".join(msgs) if msgs else ""))
    sys.exit(0 if ok else 1)


def cmd_rmswin(args):
    path, t0, t1, min_rms = args[0], float(args[1]), float(args[2]), float(args[3])
    s, rate = load_wav(path)
    seg = s[int(t0 * rate):int(t1 * rate)]
    if not seg:
        print("rmswin: empty window")
        sys.exit(2)
    rms = math.sqrt(sum(v * v for v in seg) / len(seg))
    ok = rms >= min_rms
    print("rmswin %s [%g,%g]s: rms=%.1f (min %.1f) -> %s" % (
        path, t0, t1, rms, min_rms, "PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    cmd = sys.argv[1]
    fns = {"gen": cmd_gen, "info": cmd_info, "frames": cmd_frames,
           "extractch": cmd_extractch, "rms": cmd_rms, "lagfind": cmd_lagfind,
           "cmpwin": cmd_cmpwin, "corrwin": cmd_corrwin, "fftpeaks": cmd_fftpeaks,
           "rmswin": cmd_rmswin}
    if cmd not in fns:
        print("unknown command %r" % cmd)
        sys.exit(2)
    fns[cmd](sys.argv[2:])


if __name__ == "__main__":
    main()
