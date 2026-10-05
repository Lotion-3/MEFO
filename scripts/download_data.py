"""Download external data into data/ and verify checksums.

The first successful download of a file records its SHA-256 in data/checksums.json; every later
download or `--verify` must match it. Commit checksums.json so collaborators verify the same bytes.

    uv run python scripts/download_data.py --model iML1515
    uv run python scripts/download_data.py --dataset haverkorn2011
    uv run python scripts/download_data.py --enzyme ecmpy_iML1515
    uv run python scripts/download_data.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

DATA = Path("data")
CHECKSUMS = DATA / "checksums.json"

# Source URLs. Verified against BiGG's API on 2026-09-30 (see docs/model_notes.md).
MODELS = {
    "iML1515": "https://bigg.ucsd.edu/static/models/iML1515.xml",
}

# Supplementary archives from Europe PMC's REST API (the server is slow: allow ~10 minutes).
# Only the listed members are extracted; see docs/dataset_notes.md for provenance.
DATASETS = {
    "haverkorn2011": (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC3094070/supplementaryFiles",
        ["msb20119-s1.xls", "msb20119-s2.xls", "msb20119-s3.xls"],
    ),
}

# Enzyme parameter files, pinned to a commit (ECMpy licence: MIT + Commons Clause, non-commercial).
ENZYME = {
    "ecmpy_iML1515": (
        "https://raw.githubusercontent.com/tibbdc/ECMpy/"
        "467e04057faca7834e97a634f7f1bec425adef74/model/eciML1515.json",
        "raw/ecmpy/eciML1515.json",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checksums() -> dict[str, str]:
    return json.loads(CHECKSUMS.read_text()) if CHECKSUMS.exists() else {}


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed https URLs above
    tmp.replace(dest)


def ensure(rel: str, url: str) -> bool:
    dest = DATA / rel
    sums = load_checksums()
    if not dest.exists():
        fetch(url, dest)
    digest = sha256(dest)
    if rel not in sums:
        sums[rel] = digest
        CHECKSUMS.write_text(json.dumps(sums, indent=2, sort_keys=True) + "\n")
        print(f"{rel}: recorded sha256 {digest}")
        return True
    if sums[rel] != digest:
        print(f"{rel}: CHECKSUM MISMATCH (expected {sums[rel]}, got {digest})", file=sys.stderr)
        return False
    print(f"{rel}: ok")
    return True


def ensure_dataset(name: str) -> bool:
    url, members = DATASETS[name]
    missing = [m for m in members if not (DATA / "raw" / name / m).exists()]
    if missing:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "suppl.zip"
            fetch(url, archive)
            with zipfile.ZipFile(archive) as z:
                for m in missing:
                    (DATA / "raw" / name).mkdir(parents=True, exist_ok=True)
                    (DATA / "raw" / name / m).write_bytes(z.read(m))
    ok = True
    for m in members:
        ok &= ensure(f"raw/{name}/{m}", url)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], choices=sorted(MODELS))
    ap.add_argument("--dataset", action="append", default=[], choices=sorted(DATASETS))
    ap.add_argument("--enzyme", action="append", default=[], choices=sorted(ENZYME))
    ap.add_argument("--verify", action="store_true", help="verify all recorded files")
    args = ap.parse_args()
    ok = True
    for m in args.model:
        ok &= ensure(f"models/{m}.xml", MODELS[m])
    for d in args.dataset:
        ok &= ensure_dataset(d)
    for e in args.enzyme:
        url, rel = ENZYME[e]
        ok &= ensure(rel, url)
    if args.verify:
        for rel, expected in load_checksums().items():
            path = DATA / rel
            good = path.exists() and sha256(path) == expected
            print(f"{rel}: {'ok' if good else 'MISSING OR MISMATCH'}")
            ok &= good
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
