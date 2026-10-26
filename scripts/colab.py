"""Reproducible Colab CLI workflow; run from any directory with a local Python.

python scripts/colab.py setup
python scripts/colab.py train --download --max-steps 100
python scripts/colab.py fetch
python scripts/colab.py stop
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/content/nanodit"


def main():
    # TPU provisioning can outlast AuthorizedSession's 120-second default.
    # Respect an explicit override; the CLI reconciles timed-out allocations.
    os.environ.setdefault("COLAB_CLI_REQUEST_TIMEOUT", "300")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", default="nanodit-jax")
    p.add_argument("--auth", choices=["oauth2", "adc"], default="oauth2")
    p.add_argument("--endpoint", help="Existing runtime ID shown by colab sessions; used by attach")
    p.add_argument("--skip-checkpoints", action="store_true", help="Fetch logs and sample images without large weights")
    p.add_argument("action", choices=["setup", "attach", "install", "sync", "train", "fetch", "status", "stop"])
    args, extra = p.parse_known_args()
    binary = shutil.which("colab") or str(ROOT / ".venv/bin/colab")
    config = ROOT / ".colab/sessions.json"
    config.parent.mkdir(exist_ok=True)
    cli = [binary, "--auth", args.auth, "--config", str(config)]
    session = ["-s", args.session]

    def call(*command):
        subprocess.run([*cli, *command], check=True)

    def cli_state():
        from colab_cli.auth import AuthProvider
        from colab_cli.common import State
        state = State()
        state.config_path = str(config)
        state.auth_provider = AuthProvider(args.auth)
        return state

    def execute(code):
        # CLI 0.7.5 can exit zero after a remote Python exception. Require an
        # explicit success marker so setup/train failures propagate locally.
        marker = "NANODIT_OK_" + uuid.uuid4().hex
        process = subprocess.Popen(
            [*cli, "exec", *session, "--timeout", "86400"], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        process.stdin.write(code + f"\nprint({marker!r}, flush=True)\n")
        process.stdin.close()
        succeeded = False
        for line in process.stdout:
            if line.strip() == marker:
                succeeded = True
            else:
                print(line, end="", flush=True)
        if process.wait() or not succeeded:
            raise RuntimeError("Colab execution failed; see the remote output above")

    def run_remote(command):
        # Heartbeats keep CLI execution responsive while XLA compiles silently.
        execute(f'''
import pathlib, subprocess, time
root = pathlib.Path({REMOTE!r})
log = root / "command.log"
with log.open("w") as stream:
    process = subprocess.Popen({command!r}, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
with log.open() as stream:
    while process.poll() is None:
        output = stream.read()
        print(output if output else "[nanodit] Remote process running...", flush=True)
        time.sleep(5)
    print(stream.read(), flush=True)
if process.returncode:
    raise RuntimeError(f"Remote process exited with status {{process.returncode}}; see {{log}}")
''')

    def sync():
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "nanodit-source.tar.gz"
            files = [ROOT / name for name in ["model.py", "train.py", "data.py", "requirements.txt"]]
            with tarfile.open(archive, "w:gz") as tar:
                for path in files:
                    tar.add(path, arcname=path.relative_to(ROOT))
            call("upload", *session, str(archive), "/content/nanodit-source.tar.gz")
        execute(f'''
import pathlib, tarfile
pathlib.Path({REMOTE!r}).mkdir(exist_ok=True)
with tarfile.open("/content/nanodit-source.tar.gz") as archive:
    archive.extractall({REMOTE!r}, filter="data")
''')

    python = f"{REMOTE}/.venv/bin/python"
    if args.action == "attach":
        # Recovery for CLI 0.7.5: a timed-out POST may still provision a TPU,
        # which `colab sessions` shows as [?] without a local session name.
        from colab_cli.state import SessionState
        if not args.endpoint:
            p.error("attach requires --endpoint from colab sessions")
        state = cli_state()
        matches = [a for a in state.client.list_assignments() if a.endpoint == args.endpoint]
        if len(matches) != 1:
            raise RuntimeError("The specified runtime is not assigned to this account")
        a = matches[0]
        if a.accelerator.value.upper() != "V6E1":
            raise RuntimeError("Expected a v6e-1 TPU")
        existing = state.store.list().get(args.session)
        if existing and existing.endpoint != a.endpoint:
            raise RuntimeError("Session name already refers to a different runtime")
        if existing:
            print(f"Runtime is already attached as {args.session}")
            return
        info = a.runtime_proxy_info
        state.store.add(SessionState(name=args.session, token=info.token, url=info.url,
            endpoint=a.endpoint, token_expires_at=info.expires_at(), variant=a.variant.name,
            accelerator=a.accelerator.value, machine_shape=a.machine_shape.name))
        print(f"Attached existing TPU as {args.session}. Run install next.")
        return
    if args.action == "setup":
        tracked, assignments = cli_state().sync_sessions()
        if args.session in tracked:
            if tracked[args.session].accelerator.upper() != "V6E1":
                raise RuntimeError("Session name already refers to a different accelerator")
            print(f"Reusing existing session {args.session}")
        else:
            known = {s.endpoint for s in tracked.values()}
            orphaned = [a.endpoint for a in assignments if a.accelerator.value == "V6E1" and a.endpoint not in known]
            if orphaned:
                raise RuntimeError(f"A TPU already exists without a local name: {orphaned}. "
                                   "Use --endpoint RUNTIME_ID attach, then install.")
            call("new", *session, "--tpu", "v6e1", "--high-mem")
    if args.action in {"setup", "install"}:
        sync()
        # Colab's system Python may omit ensurepip. pip --python can populate
        # a venv created without pip, without changing the notebook environment.
        run_remote(["python3", "-m", "venv", "--without-pip", f"{REMOTE}/.venv"])
        run_remote(["python3", "-m", "pip", "--python", python, "install", "-r", "requirements.txt"])
        run_remote([python, "-c", "import jax, os; print('Devices:', jax.devices()); "
                    "print('Host RAM GiB:', os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30); "
                    "assert jax.default_backend() == 'tpu'; print(jax.devices()[0].memory_stats())"])
    elif args.action == "sync":
        sync()
    elif args.action == "train":
        sync()
        command = [python, "-u", "train.py", "--require-tpu"]
        run_remote([*command, *extra])
    elif args.action == "fetch":
        execute(f'''
import tarfile
with tarfile.open("/content/nanodit-results.tar.gz", "w:gz") as archive:
    archive.add({REMOTE + '/runs'!r}, arcname="runs", filter=lambda info: None if
        "/jax_cache" in info.name or ({args.skip_checkpoints!r} and info.name.endswith(".msgpack")) else info)
''')
        (ROOT / "runs").mkdir(exist_ok=True)
        call("download", *session, "/content/nanodit-results.tar.gz", str(ROOT / "runs/colab-results.tar.gz"))
    else:
        call(args.action, *session)


if __name__ == "__main__":
    main()
