"""Package the standalone active_mining adapter as s3_r2_adapter.zip + .sha256 without touching frozen source."""
from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_DIR = ROOT / "src" / "labyrinth_nn" / "active_mining"


def pack_adapter(out_zip: Path) -> tuple[Path, str]:
    out_zip = out_zip.resolve()
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in ADAPTER_DIR.rglob("*.py") if "__pycache__" not in p.parts)
    if not files:
        raise FileNotFoundError(f"no adapter modules found in {ADAPTER_DIR}")
    tmp_zip = out_zip.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            arcname = file_path.relative_to(ROOT / "src").as_posix()
            info = zipfile.ZipInfo(arcname, date_time=(2026, 11, 24, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, file_path.read_bytes())
    tmp_zip.replace(out_zip)
    digest = hashlib.sha256(out_zip.read_bytes()).hexdigest()
    sha_path = out_zip.with_name(f"{out_zip.name}.sha256")
    sha_path.write_text(f"{digest}  {out_zip.name}\n", encoding="utf-8", newline="\n")
    return out_zip, digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "adapters" / "s3_r2_adapter.zip")
    args = parser.parse_args(argv)
    zip_path, digest = pack_adapter(args.out)
    print(f"{zip_path}  sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
