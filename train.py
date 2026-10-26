"""Rectified-flow training on one TPU (also runs on CPU/GPU)."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training import train_state
from PIL import Image

from data import ImageFolder, prefetch
from model import NanoDiT


class TrainState(train_state.TrainState):
    ema_params: object
    rng: object
    epoch_loss_sum: object
    epoch_loss_count: object


def flow_lerp(x, key):
    time_key, noise_key = jax.random.split(key)
    t = jax.random.uniform(time_key, (x.shape[0],), dtype=jnp.float32)
    noise = jax.random.normal(noise_key, x.shape, dtype=jnp.float32)
    tb = t[:, None, None, None]
    return (1 - tb) * x + tb * noise, noise, t


def create_state(model, seed=42, learning_rate=1e-4, warmup_steps=0, weight_decay=0.01):
    # uint32 PRNGKey arrays serialize directly in the msgpack checkpoint.
    init_key, train_key = jax.random.split(jax.random.PRNGKey(seed))
    params = jax.jit(model.init)(init_key,
        jnp.zeros((1, model.input_size, model.input_size, model.in_channels)),
        jnp.zeros((1,)), jnp.zeros((1,), dtype=jnp.int32))["params"]
    schedule = optax.linear_schedule(0.0, learning_rate, warmup_steps) if warmup_steps else learning_rate
    # Match torch.optim.AdamW defaults: all parameters decay, no clipping.
    tx = optax.adamw(schedule, b1=0.9, b2=0.999, eps=1e-8, weight_decay=weight_decay)
    # Distinct buffers are important when donating the state to the training JIT.
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx,
                             ema_params=jax.tree.map(jnp.copy, params), rng=train_key,
                             epoch_loss_sum=jnp.float32(0), epoch_loss_count=jnp.int32(0))


def make_train_step(model, microbatch_size, ema_decay=0.9999, steps_per_epoch=50):
    def loss_fn(params, images, labels, key):
        flow_key, dropout_key = jax.random.split(key)
        x = images.astype(jnp.float32) / 127.5 - 1
        zt, noise, t = flow_lerp(x, flow_key)
        velocity = model.apply({"params": params}, zt, t, labels, train=True, rngs={"dropout": dropout_key})
        return jnp.mean(jnp.square(velocity - (noise - x)), dtype=jnp.float32)

    @jax.jit(donate_argnums=(0,))
    def step(state, images, labels):
        count = images.shape[0] // microbatch_size
        images = images.reshape(count, microbatch_size, *images.shape[1:])
        labels = labels.reshape(count, microbatch_size)
        next_key, key = jax.random.split(state.rng)
        keys = jax.random.split(key, count)
        zeros = jax.tree.map(jnp.zeros_like, state.params)

        def accumulate(carry, batch):
            grad_sum, loss_sum = carry
            loss, grads = jax.value_and_grad(loss_fn)(state.params, *batch)
            return (jax.tree.map(jnp.add, grad_sum, grads), loss_sum + loss), None

        (grads, loss), _ = jax.lax.scan(accumulate, (zeros, jnp.float32(0)), (images, labels, keys))
        grads = jax.tree.map(lambda g: g / count, grads)
        grad_norm = optax.tree.norm(grads)
        state = state.apply_gradients(grads=grads)
        ema = jax.tree.map(lambda old, new: ema_decay * old + (1 - ema_decay) * new, state.ema_params, state.params)
        loss = loss / count
        epoch_sum = state.epoch_loss_sum + loss
        epoch_count = state.epoch_loss_count + 1
        epoch_end = state.step % steps_per_epoch == 0
        state = state.replace(ema_params=ema, rng=next_key,
            epoch_loss_sum=jnp.where(epoch_end, 0.0, epoch_sum),
            epoch_loss_count=jnp.where(epoch_end, 0, epoch_count))
        return state, {"loss": loss, "grad_norm": grad_norm, "epoch_loss": epoch_sum / epoch_count}

    return step


def make_sampler(model, ode_steps=150, cfg_scale=5.0):
    if ode_steps <= 0:
        raise ValueError("ode_steps must be positive")

    @jax.jit
    def sample(params, key, labels):
        z = jax.random.normal(key, (labels.shape[0], model.input_size, model.input_size, model.in_channels))
        use_cfg = cfg_scale != 1.0
        y = jnp.concatenate([labels, jnp.full_like(labels, model.num_classes)]) if use_cfg else labels

        def euler(i, z):
            inputs = jnp.concatenate([z, z]) if use_cfg else z
            t = jnp.full((inputs.shape[0],), 1.0 - i / ode_steps, dtype=jnp.float32)
            v = model.apply({"params": params}, inputs, t, y)
            if use_cfg:
                cond, uncond = jnp.split(v, 2)
                v = uncond + cfg_scale * (cond - uncond)
            return z - v / ode_steps

        z = jax.lax.fori_loop(0, ode_steps, euler, z)
        return jnp.clip((z + 1) / 2, 0, 1)

    return sample


def save_checkpoint(path, state, metadata):
    """Atomic single-host checkpoint including optimizer, EMA, RNG and data position."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state": serialization.to_state_dict(jax.device_get(state)), "metadata": metadata}
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(serialization.msgpack_serialize(payload))
    temporary.replace(path)


def restore_checkpoint(path, state, metadata):
    payload = serialization.msgpack_restore(Path(path).read_bytes())
    if payload["metadata"] != metadata:
        raise ValueError("Checkpoint model/training/data configuration differs from this run")
    restored = serialization.from_state_dict(state, payload["state"])
    return jax.tree.map(jnp.asarray, restored)


def save_grid(images, path, columns=4):
    images = np.uint8(np.clip(np.asarray(images), 0, 1) * 255)
    n, h, w, c = images.shape
    grid = np.zeros(((n + columns - 1) // columns * h, columns * w, c), dtype=np.uint8)
    for i, image in enumerate(images):
        row, col = divmod(i, columns)
        grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = image
    Image.fromarray(grid).save(path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="butterflies")
    p.add_argument("--output-dir", default="runs/nanodit")
    p.add_argument("--resume", type=Path)
    p.add_argument("--download", action="store_true")
    p.add_argument("--require-tpu", action="store_true")
    p.add_argument("--sample-ema", action="store_true", help="Also save EMA sample grids; primary samples use live weights")
    for name, default in [("image-size", 64), ("patch-size", 2), ("hidden-size", 768),
        ("depth", 12), ("num-heads", 12), ("num-classes", 75), ("batch-size", 128),
        ("microbatch-size", 8), ("epochs", 600), ("max-steps", 0), ("seed", 42),
        ("warmup-steps", 0), ("log-every", 50), ("sample-every", 500),
        ("checkpoint-every", 1250), ("ode-steps", 150), ("workers", 8)]:
        p.add_argument(f"--{name}", type=int, default=default)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--cfg-scale", type=float, default=5.0)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--remat", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    for name in ["image_size", "patch_size", "hidden_size", "depth", "num_heads", "num_classes",
                 "batch_size", "microbatch_size", "epochs", "log_every", "ode_steps", "workers"]:
        if getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.batch_size % args.microbatch_size:
        p.error("--batch-size must be divisible by --microbatch-size")
    if min(args.max_steps, args.warmup_steps, args.sample_every, args.checkpoint_every, args.seed) < 0:
        p.error("steps, intervals and seed must be nonnegative")
    if not 0 <= args.ema_decay < 1:
        p.error("--ema-decay must be in [0, 1)")
    return args


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main(args=None):
    args = args or parse_args()
    if args.require_tpu and jax.default_backend() != "tpu":
        raise RuntimeError(f"A TPU was requested but JAX selected {jax.default_backend()}: {jax.devices()}")
    if jax.device_count() != 1:
        raise RuntimeError("This trainer targets a single device (v6e-1); multi-device sharding is not configured")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "latest.msgpack").exists() and args.resume is None:
        raise ValueError("Output already contains a checkpoint; use --resume or a new --output-dir")
    jax.config.update("jax_compilation_cache_dir", str(output / "jax_cache"))
    print(f"JAX {jax.__version__}; devices: {jax.devices()}; compute: {args.dtype}", flush=True)
    if args.download:
        from huggingface_hub import snapshot_download
        snapshot_download("sayakpaul/butteflies_with_classes", repo_type="dataset", local_dir=args.data_dir, max_workers=32)
    dataset = ImageFolder(args.data_dir, args.image_size, args.workers)
    classes = dataset.classes
    if len(classes) != args.num_classes:
        raise ValueError(f"Expected {args.num_classes} classes, found {len(classes)}: {classes}")
    steps_per_epoch = len(dataset.labels) // args.batch_size
    if not steps_per_epoch:
        raise ValueError("Dataset is smaller than --batch-size")
    metadata = {k: v for k, v in vars(args).items() if k not in {
        "output_dir", "resume", "download", "require_tpu", "epochs", "max_steps", "log_every",
        "sample_every", "checkpoint_every", "ode_steps", "cfg_scale", "workers", "data_dir", "sample_ema"}}
    metadata.update(classes=classes, num_images=len(dataset.labels))
    metadata["data_sha256"] = hashlib.sha256(memoryview(dataset.images)).hexdigest()
    model = NanoDiT(input_size=args.image_size, patch_size=args.patch_size, hidden_size=args.hidden_size,
                   depth=args.depth, num_heads=args.num_heads, num_classes=args.num_classes,
                   dtype=getattr(jnp, args.dtype), remat=args.remat)
    state = create_state(model, args.seed, args.learning_rate, args.warmup_steps, args.weight_decay)
    if args.resume:
        state = restore_checkpoint(args.resume, state, metadata)
    start_step = int(state.step)
    end_step = args.max_steps or args.epochs * steps_per_epoch
    if end_step <= start_step:
        raise ValueError("Training limit must be greater than the restored step")
    if args.resume and (output / "metrics.jsonl").exists():
        # Discard log entries after the restored checkpoint before replaying them.
        lines = []
        for line in (output / "metrics.jsonl").read_text().splitlines():
            try:
                if json.loads(line)["step"] <= start_step:
                    lines.append(line)
            except (json.JSONDecodeError, KeyError):
                continue
        (output / "metrics.jsonl").write_text("\n".join(lines) + "\n")
    write_json(output / "config.json", metadata)
    write_json(output / "run.json", {**vars(args), "resume": str(args.resume) if args.resume else None,
        "steps_per_epoch": steps_per_epoch, "total_steps": end_step, "start_step": start_step,
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("model.py", "train.py", "data.py", "plot_loss.py")}})
    print(f"Parameters: {sum(p.size for p in jax.tree.leaves(state.params)):,}; effective batch {args.batch_size}; "
          f"microbatch {args.microbatch_size}; steps {start_step}..{end_step}", flush=True)
    batches = dataset.batches(args.batch_size, args.seed, start_step, end_step)
    train_step = make_train_step(model, args.microbatch_size, args.ema_decay, steps_per_epoch)
    sampler = make_sampler(model, args.ode_steps, args.cfg_scale)
    sample_labels = jnp.repeat(jnp.arange(min(args.num_classes, 5), dtype=jnp.int32), 4)
    tick, last_log_step = time.perf_counter(), start_step
    checkpoint_step = start_step if args.resume else 0
    from plot_loss import plot_loss
    for step, (images, labels) in enumerate(prefetch(iter(batches)), start_step + 1):
        state, metrics = train_step(state, jax.device_put(images), jax.device_put(labels))
        epoch_end = step % steps_per_epoch == 0
        if step == start_step + 1 or step % args.log_every == 0 or epoch_end or step == end_step:
            metrics = jax.device_get(metrics)  # Synchronize only at logging/checkpoint boundaries.
            elapsed = time.perf_counter() - tick
            if not all(np.isfinite(v) for v in metrics.values()):
                raise FloatingPointError(f"Nonfinite training metrics at step {step}: {metrics}")
            record = dict(step=step, loss=float(metrics["loss"]), grad_norm=float(metrics["grad_norm"]),
                          epoch=step / steps_per_epoch, seconds_per_step=elapsed / (step - last_log_step))
            if epoch_end:
                record["epoch_loss"] = float(metrics["epoch_loss"])
            print(json.dumps(record), flush=True)
            with (output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            tick, last_log_step = time.perf_counter(), step
        if args.sample_every and (step % args.sample_every == 0 or step == end_step):
            images = sampler(state.params, jax.random.PRNGKey(args.seed + 1), sample_labels)
            save_grid(images, output / f"samples_{step:07d}.png")
            if args.sample_ema:
                images = sampler(state.ema_params, jax.random.PRNGKey(args.seed + 1), sample_labels)
                save_grid(images, output / f"samples_ema_{step:07d}.png")
            print(f"Saved sample grids at step {step} (epoch {step / steps_per_epoch:g})", flush=True)
            tick = time.perf_counter()
            last_log_step = step
        if step == steps_per_epoch or (args.sample_every and step % args.sample_every == 0) or step == end_step:
            plot_loss(output / "metrics.jsonl", output, total_epochs=end_step / steps_per_epoch)
        if (args.checkpoint_every and step % args.checkpoint_every == 0) or step == end_step:
            save_checkpoint(output / "latest.msgpack", state, metadata)
            checkpoint_step = step
            print(f"Saved checkpoint at step {step}", flush=True)
            tick = time.perf_counter()
            last_log_step = step
        if epoch_end or step == start_step + 1 or step == end_step:
            write_json(output / "status.json", dict(status="running", step=step, total_steps=end_step,
                epoch=step / steps_per_epoch, checkpoint_step=checkpoint_step))
    summary = dict(step=end_step, parameter_count=sum(p.size for p in jax.tree.leaves(state.params)),
                   jax_version=jax.__version__, device=jax.devices()[0].device_kind,
                   dtype=args.dtype, memory_stats=jax.devices()[0].memory_stats())
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    write_json(output / "status.json", dict(status="complete", step=end_step, total_steps=end_step,
        epoch=end_step / steps_per_epoch, checkpoint_step=checkpoint_step))
    print(json.dumps(summary), flush=True)
    print(f"Finished. Checkpoint: {output / 'latest.msgpack'}", flush=True)


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except Exception as error:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        write_json(Path(args.output_dir) / "status.json", dict(status="failed", error=str(error)))
        raise
