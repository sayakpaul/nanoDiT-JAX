# nanoDiT-JAX

JAX/Flax port of [nanoDiT](https://github.com/sayakpaul/nanoDiT): class-conditional, pixel-space rectified-flow training on [butterflies](https://huggingface.co/datasets/sayakpaul/butteflies_with_classes). Architecture adapted from [DiT](https://github.com/facebookresearch/DiT).

Targets one Colab v6e-1 TPU. Uses bf16 computation, float32 optimizer state, gradient accumulation, activation checkpointing, EMA and classifier-free guidance.

## Run with Colab CLI

```bash
git clone https://github.com/sayakpaul/nanoDiT-JAX.git
cd nanoDiT-JAX
uv venv --python 3.12
uv pip install --python .venv/bin/python google-colab-cli==0.7.5
.venv/bin/python scripts/colab.py setup
.venv/bin/python scripts/colab.py train --download --max-steps 10000
.venv/bin/python scripts/colab.py fetch
.venv/bin/python scripts/colab.py stop
```

Complete Google sign-in when prompted. TPU runtimes have a fixed RAM configuration; Colab ignores `--high-mem`.

Resume before stopping the runtime:

```bash
.venv/bin/python scripts/colab.py train \
  --resume runs/nanodit/latest.msgpack --max-steps 20000
```

`fetch` saves `runs/colab-results.tar.gz` locally. Retrieve checkpoints before stopping: runtime storage is temporary. Training flags are listed by `python train.py --help` in the remote environment.

If allocation times out, check for an existing runtime before retrying:

```bash
.venv/bin/colab --auth oauth2 --config .colab/sessions.json sessions
.venv/bin/python scripts/colab.py --endpoint RUNTIME_ID attach
.venv/bin/python scripts/colab.py install
```

Validated on v6e-1: 129.6M parameters, 64×64 images, batch 128 (microbatch 8), approximately 0.85 seconds/update after compilation. Training, sampling and checkpoint resume verified.
