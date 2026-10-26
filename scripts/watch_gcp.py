"""Mirror run artifacts locally and retrieve the final checkpoint before stopping the VM."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', required=True)
    p.add_argument('--zone', required=True)
    p.add_argument('--vm', default='nanodit-jax-v6e1')
    p.add_argument('--interval', type=int, default=120)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    local = root / 'runs/gcp'
    local.mkdir(parents=True, exist_ok=True)
    common = [f'--project={args.project}', f'--zone={args.zone}', '--quiet']

    def call(command, timeout=180):
        subprocess.run(command, check=True, timeout=timeout, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    while True:
        try:
            # Checkpoints and compilation cache are intentionally excluded from periodic copies.
            command = ('sudo tar --warning=no-file-changed --ignore-failed-read -czf /tmp/nanodit-progress.tar.gz '
                       '--exclude="*.msgpack" --exclude="*.tmp" --exclude="jax_cache" -C /opt/nanodit/runs full '
                       '&& sudo chmod 644 /tmp/nanodit-progress.tar.gz')
            call(['gcloud', 'compute', 'ssh', args.vm, *common, '--command', command])
            archive = local / 'progress.tar.gz'
            call(['gcloud', 'compute', 'scp', f'{args.vm}:/tmp/nanodit-progress.tar.gz', str(archive), *common])
            with tarfile.open(archive) as tar:
                tar.extractall(local, filter='data')
            output = local / 'full'
            status_path = output / 'status.json'
            status = json.loads(status_path.read_text()) if status_path.exists() else {'status': 'preparing'}
            print(json.dumps(status), flush=True)
            assets = root / 'assets'
            assets.mkdir(exist_ok=True)
            for name in ['loss.png', 'loss.svg']:
                source = output / name
                if source.exists():
                    temporary = assets / ('.' + name)
                    shutil.copy2(source, temporary)
                    temporary.replace(assets / name)
            for prefix, name in [('samples_0', 'samples.png'), ('samples_ema_', 'samples_ema.png')]:
                samples = sorted(output.glob(prefix + '*.png'))
                if samples:
                    shutil.copy2(samples[-1], assets / name)
            if status['status'] == 'failed':
                raise RuntimeError('Training failed; inspect runs/gcp/full/job.log')
            if status['status'] == 'complete':
                checkpoint = output / 'latest.msgpack.partial'
                call(['gcloud', 'compute', 'scp', f'{args.vm}:/opt/nanodit/runs/full/latest.msgpack', str(checkpoint), *common], timeout=720)
                checkpoint.replace(output / 'latest.msgpack')
                call(['gcloud', 'compute', 'instances', 'stop', args.vm, *common], timeout=300)
                print('Complete: final artifacts retrieved and VM stopped; persistent disk retained.', flush=True)
                return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, tarfile.TarError, json.JSONDecodeError) as error:
            # Avoid printing credential-bearing transport diagnostics.
            print(f'Artifact sync retry: {type(error).__name__}', flush=True)
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
