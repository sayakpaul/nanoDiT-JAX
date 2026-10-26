"""Benchmark fixed effective-batch TPU training; discard all benchmark weights."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def measure(microbatch, remat):
    import jax
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model import NanoDiT
    from train import create_state, make_train_step
    if jax.default_backend() != 'tpu':
        raise RuntimeError('Benchmark requires a TPU')
    model = NanoDiT(remat=remat)
    state = create_state(model)
    step = make_train_step(model, microbatch)
    rng = np.random.default_rng(42)
    images = jax.device_put(rng.integers(0, 256, (128, 64, 64, 3), dtype=np.uint8))
    labels = jax.device_put(rng.integers(0, 75, (128,), dtype=np.int32))
    compiled = step.lower(state, images, labels).compile()
    timings = []
    for i in range(7):
        start = time.perf_counter()
        state, metrics = compiled(state, images, labels)
        metrics = jax.device_get(metrics)
        elapsed = time.perf_counter() - start
        if not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError('Nonfinite benchmark metrics')
        if i >= 2:
            timings.append(elapsed)
    memory = compiled.memory_analysis()
    return dict(microbatch_size=microbatch, remat=remat, seconds_per_step=float(np.median(timings)),
                estimated_bytes=memory.argument_size_in_bytes + memory.output_size_in_bytes
                    + memory.temp_size_in_bytes - memory.alias_size_in_bytes,
                memory_stats=jax.devices()[0].memory_stats())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--microbatch', type=int)
    p.add_argument('--remat', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--output', type=Path, default=Path('runs/benchmark.json'))
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.microbatch:
        result = measure(args.microbatch, args.remat)
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return
    results = []
    for microbatch, remat in [(8, True), (16, True), (32, True), (8, False), (16, False)]:
        result_file = args.output.with_name(f'benchmark-{microbatch}-{remat}.json')
        command = [sys.executable, '-u', __file__, '--microbatch', str(microbatch),
                   '--remat' if remat else '--no-remat', '--output', str(result_file)]
        print(f'Benchmarking microbatch={microbatch}, remat={remat}', flush=True)
        try:
            subprocess.run(command, check=True, timeout=300)
            results.append(json.loads(result_file.read_text()))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            print(f'Configuration failed: {type(error).__name__}', flush=True)
    if not results:
        raise RuntimeError('No configuration completed successfully')
    best = min(results, key=lambda row: row['seconds_per_step'])
    args.output.write_text(json.dumps(dict(best=best, configurations=results), indent=2))
    print('Selected configuration:', json.dumps(best), flush=True)


if __name__ == '__main__':
    main()
