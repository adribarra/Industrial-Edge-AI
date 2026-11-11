"""Downloads one MVTec AD category (screw by default) into datasets/mvtec/.

Dataset: MVTec AD, license CC BY-NC-SA 4.0 (non-commercial use only).
Citation: Bergmann et al., "MVTec AD - A Comprehensive Real-World Dataset
for Unsupervised Anomaly Detection", CVPR 2019.
It only uses the Python standard library, so there is nothing extra to install.
"""
import argparse
import tarfile
import urllib.request
from pathlib import Path

URLS = {
    "screw": "https://www.mydrive.ch/shares/163649/cfd394b4e3a49013745fab9e2084988a/"
             "download/420938130-1629960389/screw.tar.xz",
}


def download(url: str, dest: Path) -> None:
    """Download a file, showing progress, and check that it arrived completely."""
    with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
    print()
    if total and dest.stat().st_size != total:
        raise IOError("Incomplete download")


def main() -> int:
    """Download and unpack the category, unless it is already there."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--category", default="screw", choices=sorted(URLS))
    p.add_argument("--dataset", type=Path, default=Path("datasets/mvtec"))
    a = p.parse_args()

    target = a.dataset / a.category
    if (target / "test").is_dir():
        print(f"Already present: {target}")
        return 0
    a.dataset.mkdir(parents=True, exist_ok=True)
    archive = a.dataset / f"{a.category}.tar.xz"
    print(f"Downloading {a.category} (license CC BY-NC-SA 4.0) ...")
    download(URLS[a.category], archive)
    print("Extracting ...")
    with tarfile.open(archive) as t:
        t.extractall(a.dataset, filter="data")
    archive.unlink()
    print(f"Done: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
