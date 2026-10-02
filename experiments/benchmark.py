"""Measure IDM-VTON speed and memory on this machine, and check seeds are reproducible.

Run from the repo root, one GPU at a time:

    CUDA_VISIBLE_DEVICES=0 python experiments/benchmark.py --steps 30 --num-samples 3

Prints seconds per sample, peak GPU memory and samples/hour, which is what you
need to size the best-of-N experiments. Results are saved to --out.
"""

import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from PIL import Image

from tryon_engine import TryOnEngine


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def load_engine():
    engine = TryOnEngine()
    engine.to_device()
    return engine


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--human', default=os.path.join(REPO_ROOT, 'example', 'human', '00035_00.jpg'))
    parser.add_argument('--garment', default=os.path.join(REPO_ROOT, 'example', 'cloth', '09163_00.jpg'))
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--num-samples', type=int, default=3)
    parser.add_argument('--out', default=os.path.join(REPO_ROOT, 'outputs', 'benchmark'))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU visible; set CUDA_VISIBLE_DEVICES to a GPU index.")
    props = torch.cuda.get_device_properties(0)
    total_gib = props.total_memory / 2**30
    print(f"GPU: {props.name}, {total_gib:.1f} GiB, compute capability {props.major}.{props.minor}")

    engine, load_s = timed(load_engine)
    print(f"Model load: {load_s:.1f}s")

    human = Image.open(args.human)
    garment = Image.open(args.garment)
    person, prep_s = timed(lambda: engine.prepare_person(human))
    prompts, prompt_s = timed(lambda: engine.encode_prompts(""))
    print(f"Person preprocessing (pose, parsing, DensePose): {prep_s:.1f}s, prompt encoding: {prompt_s:.1f}s")

    os.makedirs(args.out, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    sample_times = []
    first_image = None
    for seed in range(args.num_samples):
        image, sample_s = timed(lambda: engine.generate(person, garment, prompts, seed=seed, steps=args.steps))
        sample_times.append(sample_s)
        if first_image is None:
            first_image = image
        engine.paste_back(person, image).save(os.path.join(args.out, f'seed{seed}.png'))
        print(f"  seed {seed}: {sample_s:.1f}s")
    peak_gib = torch.cuda.max_memory_allocated() / 2**30

    # Best-of-N pools must be regenerable from their seeds, so check seed 0 again.
    repeat = engine.generate(person, garment, prompts, seed=0, steps=args.steps)
    max_diff = np.abs(np.asarray(first_image, dtype=np.int16) - np.asarray(repeat, dtype=np.int16)).max()

    # The first sample includes CUDA warm-up, so leave it out of the average when we can.
    steady = sample_times[1:] or sample_times
    per_sample = sum(steady) / len(steady)
    print()
    print(f"Per sample at {args.steps} steps: {per_sample:.1f}s ({per_sample / args.steps:.2f}s/step)")
    print(f"Throughput: ~{3600 / per_sample:.0f} samples/hour on this GPU")
    print(f"Peak PyTorch GPU memory: {peak_gib:.1f} of {total_gib:.1f} GiB")
    print(f"Seed 0 regenerated, max pixel difference: {max_diff} (0 = bit-identical)")
    print(f"Images saved to {args.out}")


if __name__ == '__main__':
    main()
