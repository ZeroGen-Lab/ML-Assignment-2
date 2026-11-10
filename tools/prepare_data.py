#!/usr/bin/env python3
"""Convert the downloaded datasets into tinydl .bin files.

Usage: python3 -I prepare_data.py RAW_DIR OUT_DIR

RAW_DIR must contain:
  fashion-{train,t10k}-{images-idx3,labels-idx1}-ubyte.gz   (Fashion-MNIST)
  cifar-10-binary.tar.gz                                     (CIFAR-10)
  uci-har.zip                                                (UCI HAR, may wrap an inner zip)
Archives are read in memory; nothing is extracted to disk.
Format: b"TDL1", int32 ndim, int32 dims[ndim], int32 dtype (0 f32, 1 u8), raw data.
"""
import gzip
import io
import struct
import sys
import tarfile
import zipfile
from array import array
from pathlib import Path


def write_bin(path: Path, dims, dtype: int, payload: bytes) -> None:
    n = 1
    for d in dims:
        n *= d
    assert len(payload) == n * (4 if dtype == 0 else 1), (path, len(payload), dims)
    with open(path, "wb") as f:
        f.write(b"TDL1" + struct.pack("<i", len(dims)) + struct.pack(f"<{len(dims)}i", *dims))
        f.write(struct.pack("<i", dtype) + payload)
    print(f"  {path.name}: {list(dims)} {'f32' if dtype == 0 else 'u8'}")


def fashion(raw: Path, out: Path) -> None:
    for split, name in (("train", "train"), ("t10k", "test")):
        img = gzip.decompress((raw / f"fashion-{split}-images-idx3-ubyte.gz").read_bytes())
        lab = gzip.decompress((raw / f"fashion-{split}-labels-idx1-ubyte.gz").read_bytes())
        magic, n, h, w = struct.unpack(">iiii", img[:16])
        magic2, n2 = struct.unpack(">ii", lab[:8])
        assert magic == 2051 and magic2 == 2049 and n == n2 and (h, w) == (28, 28)
        write_bin(out / f"fashion_{name}_x.bin", (n, 28, 28), 1, img[16:16 + n * 784])
        write_bin(out / f"fashion_{name}_y.bin", (n,), 1, lab[8:8 + n])


def cifar(raw: Path, out: Path) -> None:
    files = {}
    with tarfile.open(raw / "cifar-10-binary.tar.gz", "r:gz") as tar:
        for m in tar.getmembers():
            base = m.name.rsplit("/", 1)[-1]
            if m.isfile() and base.endswith(".bin") and ("data_batch_" in base or base == "test_batch.bin"):
                files[base] = tar.extractfile(m).read()
    for name, parts in (("train", [f"data_batch_{i}.bin" for i in range(1, 6)]), ("test", ["test_batch.bin"])):
        blob = b"".join(files[p] for p in parts)
        assert len(blob) % 3073 == 0
        n = len(blob) // 3073
        xs, ys = bytearray(n * 3072), bytearray(n)
        for i in range(n):
            rec = blob[i * 3073:(i + 1) * 3073]
            ys[i] = rec[0]
            xs[i * 3072:(i + 1) * 3072] = rec[1:]
        assert max(ys) <= 9
        write_bin(out / f"cifar_{name}_x.bin", (n, 3, 32, 32), 1, bytes(xs))
        write_bin(out / f"cifar_{name}_y.bin", (n,), 1, bytes(ys))


HAR_CHANNELS = ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z",
                "total_acc_x", "total_acc_y", "total_acc_z"]


def har(raw: Path, out: Path) -> None:
    z = zipfile.ZipFile(raw / "uci-har.zip")
    inner = [n for n in z.namelist() if n.endswith("UCI HAR Dataset.zip") and "__MACOSX" not in n]
    if inner:
        z = zipfile.ZipFile(io.BytesIO(z.read(inner[0])))
    names = {n.split("UCI HAR Dataset/", 1)[-1]: n for n in z.namelist() if "__MACOSX" not in n}

    def table(rel):
        rows = [list(map(float, line.split())) for line in z.read(names[rel]).decode().splitlines() if line.strip()]
        return rows

    data = {}
    for split in ("train", "test"):
        chans = [table(f"{split}/Inertial Signals/{c}_{split}.txt") for c in HAR_CHANNELS]
        labels = [int(r[0]) - 1 for r in table(f"{split}/y_{split}.txt")]
        n = len(labels)
        assert all(len(c) == n and len(c[0]) == 128 for c in chans) and set(labels) <= set(range(6))
        data[split] = (chans, labels)
    # per-channel standardisation with training statistics
    stats = []
    for ch in data["train"][0]:
        vals = [v for row in ch for v in row]
        m = sum(vals) / len(vals)
        s = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
        stats.append((m, s))
    for split, (chans, labels) in data.items():
        n = len(labels)
        x = array("f", [0.0]) * (n * 128 * 9)
        for c, (rows, (m, s)) in enumerate(zip(chans, stats)):
            for i, row in enumerate(rows):
                base = i * 128 * 9 + c
                for t, v in enumerate(row):
                    x[base + t * 9] = (v - m) / s
        if sys.byteorder != "little":
            x.byteswap()
        write_bin(out / f"har_{split}_x.bin", (n, 128, 9), 0, x.tobytes())
        write_bin(out / f"har_{split}_y.bin", (n,), 1, bytes(labels))


NEEDS = {
    "fashion": [f"fashion-{s}-{k}-ubyte.gz" for s in ("train", "t10k") for k in ("images-idx3", "labels-idx1")],
    "cifar": ["cifar-10-binary.tar.gz"],
    "har": ["uci-har.zip"],
}


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    raw, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    only = sys.argv[3:] or list(NEEDS)
    unknown = [n for n in only if n not in NEEDS]
    if unknown:
        sys.exit(f"未知的数据集：{'、'.join(unknown)}（可选 {'、'.join(NEEDS)}）")
    done = 0
    for name in only:
        missing = [f for f in NEEDS[name] if not (raw / f).is_file()]
        if missing:
            print(f"跳过 {name}：{raw} 里缺少 {'、'.join(missing)}（文件名见 README「数据」一节）")
            continue
        out.mkdir(parents=True, exist_ok=True)
        print(name)
        {"fashion": fashion, "cifar": cifar, "har": har}[name](raw, out)
        done += 1
    if not done:
        sys.exit("没有转换任何数据集。")


if __name__ == "__main__":
    main()
