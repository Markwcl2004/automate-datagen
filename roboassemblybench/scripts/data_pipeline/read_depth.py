"""Reference reader for the depth sidecar of the automate_fullproc LeRobot v3 datasets (needs numpy and zstandard).

Usage: python read_depth.py <dataset root> <episode_index> <camera key> <frame_index>
       e.g. python read_depth.py franka_panda_single_arm/00016/mixed 0 observation.images.front 100

Layout (also in meta/diversity/schema.json depth_format and in each index.json):
  file header  struct '<8sI'  = magic b'RABDU16\\0', version 1
  per frame    struct '<III'  = (frame_index, uncompressed_size, compressed_size), then one zstd frame of compressed_size bytes
  index.json   <camera key>.chunks[frame].offset = byte offset of that frame's 12-byte chunk header
  payload      16 bit planes (bit 0 first) of the row-major uint16 samples, each packed with numpy packbits(bitorder='little')
  values       uint16 millimetres, 0 = invalid, distance to the image plane
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np


def read_depth(dataset_root, episode_index, camera_key, frame_index):
    import zstandard as zstd

    root = Path(dataset_root)
    index = json.loads((root / f"sensors/depth/episode_{int(episode_index):06d}/index.json").read_text())
    meta = index[camera_key]
    chunk = meta["chunks"][int(frame_index)]
    with open(root / meta["path"], "rb") as f:
        magic, version = struct.unpack("<8sI", f.read(12))
        assert magic == b"RABDU16\0" and version == 1, "not a roboassemblybench_chunked_depth_v1 file"
        f.seek(int(chunk["offset"]))
        stored, raw_size, comp_size = struct.unpack("<III", f.read(12))
        assert stored == int(frame_index)
        payload = zstd.ZstdDecompressor().decompress(f.read(comp_size), max_output_size=raw_size)
    h, w = meta["shape"]
    n = h * w
    planes = np.frombuffer(payload, dtype=np.uint8).reshape(16, (n + 7) // 8)
    values = np.zeros(n, dtype=np.uint16)
    for bit in range(16):
        values |= np.unpackbits(planes[bit], bitorder="little")[:n].astype(np.uint16) << bit
    return values.reshape(h, w)  # millimetres, 0 = invalid


if __name__ == "__main__":
    d = read_depth(sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
    valid = d[d > 0]
    print(f"shape {d.shape}, valid {valid.size}/{d.size}, min {valid.min() if valid.size else 0} mm, "
          f"max {valid.max() if valid.size else 0} mm, centre {d[d.shape[0] // 2, d.shape[1] // 2]} mm")
