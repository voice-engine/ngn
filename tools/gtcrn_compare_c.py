#!/usr/bin/env python3
"""C dump（VA_GTCRN_DUMP）与 torch StreamGTCRN 逐级对拍。

用法: gtcrn_compare_c.py <dump_dir> [帧数]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from gtcrn_check_export import fresh_torch_caches, torch_stages  # noqa: E402

GT = os.path.expanduser("~/lab/nanopi-neo-air/gtcrn")


def main():
    dump_dir = sys.argv[1]
    nframes = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    sys.path.insert(0, os.path.join(GT, "stream"))
    import soundfile as sf
    import torch
    import gtcrn
    from gtcrn_stream import StreamGTCRN
    from modules.convert import convert_to_stream

    offline = gtcrn.GTCRN().eval()
    offline.load_state_dict(torch.load(
        os.path.join(GT, "stream/onnx_models/model_trained_on_dns3.tar"),
        map_location="cpu")["model"])
    tm = StreamGTCRN().eval()
    convert_to_stream(tm, offline)

    mix, _ = sf.read(os.path.join(GT, "test_wavs/mix.wav"), dtype="float32")
    mix = np.pad(mix, (0, -len(mix) % 256))
    win = torch.hann_window(512).pow(0.5)
    spec = torch.stft(torch.from_numpy(mix), 512, 256, 512, win, return_complex=True)
    X = torch.view_as_real(spec).numpy()
    # torch.stft 默认 reflect 填充，C 流式为零历史：把参考帧 0 谱换成
    # rfft([0…0, x0..255]·win)，使双方从同一初始状态演化（仅验证用）。
    f0 = np.concatenate([np.zeros(256, np.float32), mix[:256]]) * win.numpy()
    r = np.fft.rfft(f0.astype(np.float64))
    X[:, 0, 0] = r.real.astype(np.float32)
    X[:, 0, 1] = r.imag.astype(np.float32)

    tc = fresh_torch_caches()
    worst = 0.0
    # torch.stft 默认 reflect 边界填充 vs C 零历史起始：仅影响第 0 帧，
    # 流式语义下零填充才正确（无需未来样本），从第 1 帧起对拍。
    start = int(os.environ.get("SKIP", "0"))
    for t in range(nframes + start):
        if t < start:
            tst, tc = torch_stages(tm, X[:, t, :], tc)
            continue
        tst, tc = torch_stages(tm, X[:, t, :], tc)
        for name in sorted(tst):
            p = os.path.join(dump_dir, f"f{t:03d}_{name}.bin")
            if not os.path.exists(p):
                print(f"  [C 缺] {name}")
                continue
            ref = tst[name]
            got = np.fromfile(p, np.float32)
            if got.size != ref.size:
                print(f"  {name}: 尺寸 {got.size} vs 参考 {ref.size}")
                continue
            got = got.reshape(ref.shape)
            d = np.abs(got - ref).max()
            rel = d / (np.abs(ref).max() + 1e-12)
            worst = max(worst, rel)
            flag = " <-- 发散" if rel > 1e-3 else ""
            print(f"  f{t} {name:16s} max|Δ| {d:.2e} rel {rel:.2e}{flag}")
    print(f"WORST rel {worst:.2e}  ({'通过' if worst < 1e-3 else '未过'})")


if __name__ == "__main__":
    main()
