"""Export inference weights and the full resumable checkpoint for the Hub."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from flax import serialization


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("runs/hub"))
    p.add_argument("--weights", choices=["ema", "raw"], default="ema")
    args = p.parse_args()
    payload = serialization.msgpack_restore(args.checkpoint.read_bytes())
    state, metadata = payload["state"], payload["metadata"]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    params = state["ema_params" if args.weights == "ema" else "params"]
    (output / "model.msgpack").write_bytes(serialization.msgpack_serialize(params))
    config = dict(metadata, weights=args.weights, step=int(state["step"]), format="flax-msgpack")
    config["epochs"] = config["step"] / (metadata["num_images"] // metadata["batch_size"])
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copy2(args.checkpoint, output / "checkpoint.msgpack")
    checksums = {}
    for name in ("model.msgpack", "checkpoint.msgpack", "config.json"):
        with (output / name).open("rb") as f:
            checksums[name] = hashlib.file_digest(f, "sha256").hexdigest()
    (output / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
    print(json.dumps(dict(step=config["step"], epochs=config["epochs"], weights=args.weights,
                         output_dir=str(output), sha256=checksums), indent=2), flush=True)


if __name__ == "__main__":
    main()
