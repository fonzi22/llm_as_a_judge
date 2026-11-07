"""Upload data lên Modal Volume 'hrvs-data' (chỉ upload file thiếu / khác size) rồi kiểm tra lại.
Dùng:  python prepare_data.py ["C:\\...\\CVPR_2027\\data"]
"""
import sys
from pathlib import Path

import modal

VOL_NAME = "hrvs-data"
LOCAL = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data"
CHUNK = 50


def remote_index(vol):
    try:
        return {e.path.lstrip("/"): e.size for e in vol.listdir("/", recursive=True)}
    except Exception:
        return {}


def main():
    if not (LOCAL / "annotations.json").exists():
        sys.exit(f"[X] Khong thay {LOCAL / 'annotations.json'}")
    files = [LOCAL / "annotations.json", LOCAL / "dataset_id_mapping.csv"] + sorted((LOCAL / "images").glob("*"))
    local = {p.relative_to(LOCAL).as_posix(): p for p in files if p.is_file()}
    n_img = sum(k.startswith("images/") for k in local)
    print(f"Local: {LOCAL}  ({len(local)} file, {n_img} anh, "
          f"{sum(p.stat().st_size for p in local.values()) / 1e9:.2f} GB)")

    vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
    remote = remote_index(vol)
    todo = [k for k, p in local.items() if remote.get(k) != p.stat().st_size]
    print(f"Volume '{VOL_NAME}': da co {len(local) - len(todo)}/{len(local)} file, can upload {len(todo)}")

    done_bytes, total_bytes = 0, sum(local[k].stat().st_size for k in todo)
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i:i + CHUNK]
        with vol.batch_upload(force=True) as batch:
            for k in chunk:
                batch.put_file(str(local[k]), "/" + k)
        done_bytes += sum(local[k].stat().st_size for k in chunk)
        print(f"  upload {i + len(chunk)}/{len(todo)} file  ({done_bytes / 1e9:.2f}/{total_bytes / 1e9:.2f} GB)")

    remote = remote_index(vol)
    bad = [k for k, p in local.items() if remote.get(k) != p.stat().st_size]
    if bad:
        sys.exit(f"[X] {len(bad)} file chua khop tren volume (vd: {bad[:3]}). Chay lai buoc nay.")
    print(f"[OK] Volume '{VOL_NAME}' du {len(local)} file ({n_img} anh), khop size voi may.")


if __name__ == "__main__":
    main()
