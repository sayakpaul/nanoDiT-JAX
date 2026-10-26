"""Render measured epoch-mean flow-matching loss as PNG and SVG."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_loss(metrics_path, output_dir, total_epochs=600):
    records = {}
    for line in Path(metrics_path).read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # A live log may end with a partial write.
        if "epoch_loss" in record:
            records[record["epoch"]] = record["epoch_loss"]
    if not records:
        return
    epochs = np.array(sorted(records))
    losses = np.array([records[e] for e in epochs])
    if not np.isfinite(losses).all():
        raise ValueError("Loss log contains nonfinite values")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "savefig.facecolor": "white"}):
        fig, ax = plt.subplots(figsize=(7.2, 4.3), layout="constrained")
        ax.plot(epochs, losses, color="#2563eb", linewidth=1.2, alpha=0.55 if len(epochs) >= 20 else 1,
                marker="o" if len(epochs) == 1 else None, label="Epoch mean")
        if len(epochs) >= 20:
            smooth = np.convolve(losses, np.ones(10) / 10, mode="valid")
            ax.plot(epochs[9:], smooth, color="#1e3a8a", linewidth=2, label="10-epoch trailing mean")
            ax.legend(frameon=False)
        complete = epochs[-1] >= total_epochs
        title = "nanoDiT-JAX · Butterflies"
        if not complete:
            title += f" · {epochs[-1]:g}/{total_epochs:g} epochs"
        ax.set(title=title, xlabel="Epoch", ylabel="Flow-matching MSE")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.set_xlim(left=0, right=max(epochs[-1], 2))
        for extension in ("png", "svg"):
            temporary = output / f".loss.tmp.{extension}"
            fig.savefig(temporary, dpi=200, format=extension)
            temporary.replace(output / f"loss.{extension}")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    parser.add_argument("--total-epochs", type=int, default=600)
    args = parser.parse_args()
    plot_loss(args.metrics, args.output_dir, args.total_epochs)
