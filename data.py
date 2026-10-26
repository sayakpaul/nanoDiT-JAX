"""Small ImageFolder datasets cached as uint8 in host RAM; no torch/TF dependency."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


class ImageFolder:
    def __init__(self, root, image_size, workers=8):
        root = Path(root)
        self.classes = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        samples = []
        for label, name in enumerate(self.classes):
            paths = sorted(p for p in (root / name).rglob("*") if p.suffix.lower() in extensions)
            if not paths:
                raise ValueError(f"No images in {root / name}; pass the directory containing class folders")
            samples.extend((p, label) for p in paths)
        if not samples:
            raise ValueError(f"No class folders with images in {root}")

        def decode(sample):
            path, _ = sample
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                w, h = image.size
                size = (image_size, int(h * image_size / w)) if w <= h else (int(w * image_size / h), image_size)
                image = image.resize(size, Image.Resampling.BILINEAR)
                left, top = round((size[0] - image_size) / 2), round((size[1] - image_size) / 2)
                return np.asarray(image.crop((left, top, left + image_size, top + image_size)))

        self.images = np.empty((len(samples), image_size, image_size, 3), dtype=np.uint8)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, image in enumerate(pool.map(decode, samples)):
                self.images[i] = image
        self.labels = np.array([label for _, label in samples], dtype=np.int32)
        print(f"Cached {len(samples)} images / {len(self.classes)} classes in {self.images.nbytes / 2**20:.1f} MiB", flush=True)

    def batches(self, batch_size, seed, start_step, end_step):
        steps_per_epoch = len(self.labels) // batch_size
        if not steps_per_epoch:
            raise ValueError("Dataset must contain at least one full batch")
        previous_epoch = None
        for step in range(start_step, end_step):
            epoch, offset = divmod(step, steps_per_epoch)
            if epoch != previous_epoch:
                indices = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(self.labels))
                previous_epoch = epoch
            idx = indices[offset * batch_size:(offset + 1) * batch_size]
            yield self.images[idx], self.labels[idx]


def prefetch(iterator, depth=2):
    """Bounded host prefetch. Exceptions propagate on the consumer thread."""
    from collections import deque
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = deque()
        for _ in range(depth):
            pending.append(pool.submit(next, iterator, None))
        while pending:
            item = pending.popleft().result()
            if item is None:
                return
            pending.append(pool.submit(next, iterator, None))
            yield item
