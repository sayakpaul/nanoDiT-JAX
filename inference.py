"""Generate butterfly grids from the published Flax EMA weights."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import serialization
from huggingface_hub import snapshot_download

from model import NanoDiT
from train import make_sampler, save_grid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default="sayakpaul/nanoDiT-JAX-butterflies")
    p.add_argument("--revision", default="main")
    p.add_argument("--model-dir", type=Path, help="Use a local exported model instead of the Hub")
    p.add_argument("--class-ids", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--samples-per-class", type=int, default=4)
    p.add_argument("--list-classes", action="store_true")
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--ode-steps", type=int, default=150)
    p.add_argument("--cfg-scale", type=float, default=5.0)
    p.add_argument("--dtype", choices=["bfloat16", "float32"])
    p.add_argument("--require-tpu", action="store_true")
    p.add_argument("--output", type=Path, default=Path("runs/inference.png"))
    args = p.parse_args()
    if min(args.samples_per_class, args.ode_steps) <= 0 or args.seed < 0:
        p.error("sample count and ODE steps must be positive; seed must be nonnegative")
    folder = args.model_dir or Path(snapshot_download(args.repo_id, revision=args.revision,
        allow_patterns=["config.json"] if args.list_classes else ["config.json", "model.msgpack"]))
    config = json.loads((folder / "config.json").read_text())
    if args.list_classes:
        for i, name in enumerate(config["classes"]):
            print(f"{i}: {name}")
        return
    if any(i < 0 or i >= config["num_classes"] for i in args.class_ids):
        p.error("class IDs are out of range; use --list-classes")
    if args.require_tpu and jax.default_backend() != "tpu":
        raise RuntimeError(f"Expected TPU, found {jax.devices()}")
    model = NanoDiT(input_size=config["image_size"], patch_size=config["patch_size"],
        hidden_size=config["hidden_size"], depth=config["depth"], num_heads=config["num_heads"],
        num_classes=config["num_classes"], dtype=getattr(jnp, args.dtype or config["dtype"]), remat=False)
    params = serialization.msgpack_restore((folder / "model.msgpack").read_bytes())
    params = jax.tree.map(jnp.asarray, params)
    labels = jnp.repeat(jnp.array(args.class_ids, dtype=jnp.int32), args.samples_per_class)
    print(f"{jax.devices()}; {config['weights']} weights at step {config['step']}; generating {len(labels)} images", flush=True)
    images = make_sampler(model, args.ode_steps, args.cfg_scale)(params, jax.random.PRNGKey(args.seed), labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_grid(images, args.output, columns=args.samples_per_class)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
