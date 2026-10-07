#!/usr/bin/env python3
"""Dependency-free helper for the fifo plugin data-flow tests.

Subcommands:
  gen FILE RATE DUR_S FREQ CHANNELS   - write a sine wav (S16_LE)
  info FILE                           - print wav params as key=value
  rawcheck FILE RATE CHANNELS MIN_BYTES EXPECT_FREQ TOL_PCT
                                      - verify raw S16_LE byte count and
                                        dominant frequency (zero-crossing
                                        estimate + Goertzel confirmation)
"""
import array
import math
import struct
import sys
import wave

FULL_SCALE = 32000.0


def cmd_gen(args):
    path, rate, dur, freq, channels = args[0], int(args[1]), float(args[2]), float(args[3]), int(args[4])
    nframes = int(round(rate * dur))
    wav = wave.open(path, "wb")
    wav.setnchannels(channels)
    wav.setsampwidth(2)
    wav.setframerate(rate)
    amp = 0.6
    pack = struct.Struct("<h").pack
    for n in range(nframes):
        v = int(amp * FULL_SCALE * math.sin(2.0 * math.pi * freq * n / rate))
        frame = pack(v) * channels
        wav.writeframesraw(frame)
    wav.close()
    print("gen %s: rate=%d channels=%d frames=%d freq=%.1f" % (path, rate, channels, nframes, freq))


def cmd_info(args):
    path = args[0]
    wav = wave.open(path, "rb")
    print("info %s: rate=%d channels=%d sampwidth=%d frames=%d duration=%.3f" % (
        path, wav.getframerate(), wav.getnchannels(), wav.getsampwidth(),
        wav.getnframes(), wav.getnframes() / float(wav.getframerate())))
    wav.close()


def load_raw(path):
    data = open(path, "rb").read()
    if len(data) % 2:
        data = data[:-1]
    samples = array.array("h")
    samples.frombytes(data)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def goertzel(samples, rate, freq):
    n = len(samples)
    k = 2.0 * math.pi * freq / rate
    coeff = 2.0 * math.cos(k)
    s1 = s2 = 0.0
    for x in samples:
        s0 = x + coeff * s1 - s2
        s2 = s1
        s1 = s0
    return s1 * s1 + s2 * s2 - coeff * s1 * s2


def cmd_rawcheck(args):
    path, rate, channels = args[0], int(args[1]), int(args[2])
    min_bytes, expect_freq, tol_pct = int(args[3]), float(args[4]), float(args[5])
    samples = load_raw(path)
    nbytes = len(samples) * 2
    nframes = len(samples) // channels if channels else 0
    mono = samples[::channels] if channels == 2 else samples

    ok = True
    msgs = []
    if nbytes < min_bytes:
        ok = False
        msgs.append("bytes %d < expected %d" % (nbytes, min_bytes))

    # zero-crossing frequency estimate (O(n), robust for a clean sine)
    crossings = 0
    prev = mono[0] if mono else 0
    for x in mono[1:]:
        if (prev >= 0) != (x >= 0):
            crossings += 1
        prev = x
    duration = len(mono) / float(rate)
    est = crossings / 2.0 / duration if duration > 0 else 0.0

    # Goertzel: power at the estimate must dominate its neighbours
    p_est = goertzel(mono, rate, est)
    p_lo = goertzel(mono, rate, max(1.0, est - 15.0))
    p_hi = goertzel(mono, rate, est + 15.0)
    peak = max(abs(v) for v in mono) if mono else 0
    dom_ok = p_lo == 0 or p_est > 10 * p_lo
    dom_ok = dom_ok and (p_hi == 0 or p_est > 10 * p_hi)

    if abs(est - expect_freq) > expect_freq * tol_pct / 100.0:
        ok = False
        msgs.append("dominant freq %.1f Hz != expected %.1f Hz" % (est, expect_freq))
    if not dom_ok:
        ok = False
        msgs.append("tone not dominant (p_est=%.3g p_lo=%.3g p_hi=%.3g)" % (p_est, p_lo, p_hi))
    if peak < 1000:
        ok = False
        msgs.append("peak amplitude %d too small" % peak)

    print("rawcheck %s: bytes=%d frames=%d rate=%d channels=%d peak=%d "
          "dominant_freq=%.2fHz expected=%.1fHz (%.2f%% off) -> %s%s" % (
              path, nbytes, nframes, rate, channels, peak, est, expect_freq,
              abs(est - expect_freq) / expect_freq * 100.0 if expect_freq else 0.0,
              "PASS" if ok else "FAIL", "; " + "; ".join(msgs) if msgs else ""))
    sys.exit(0 if ok else 1)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    cmd = sys.argv[1]
    if cmd == "gen":
        cmd_gen(sys.argv[2:])
    elif cmd == "info":
        cmd_info(sys.argv[2:])
    elif cmd == "rawcheck":
        cmd_rawcheck(sys.argv[2:])
    else:
        print("unknown command %r" % cmd)
        sys.exit(2)


if __name__ == "__main__":
    main()
