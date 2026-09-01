#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tải lẻ vài file audio của ViMD (nguyendv02/ViMD_Dataset) về để nghe.

    python3 download_samples.py                      # 5 mẫu lỗi mặc định
    python3 download_samples.py 73_0332 81_0303.wav  # utterance bất kỳ
    python3 download_samples.py --also-16k           # thêm bản 16 kHz mono model đã nghe

main.py chuẩn bị cả dataset (~60 GB parquet) rồi mới dùng được. Script này thì không:
parquet là định dạng cột, chia row group, nên nó mở shard *qua mạng*, chỉ đọc cột
`filename` để tìm utterance, rồi chỉ kéo đúng row group chứa nó. Tải vài chục MB thay
vì vài chục GB.

Nếu máy đã có cache 16 kHz do main.py sinh ra (data/vimd_16k/), script đọc thẳng từ đó
và không cần mạng.

Script cố ý độc lập - không import main.py - để copy sang máy khác chạy được ngay.
Chỉ cần: huggingface_hub, pyarrow, soundfile (hoặc không), numpy, scipy.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

LOGGER = logging.getLogger("vimd.samples")


# ===========================================================================
# 1. Cấu hình
# ===========================================================================

# 5 utterance trong phần phân tích lỗi: 2 ca repetition loop, 1 ca chèn đoạn mở đầu,
# 1 ca mất nửa đầu, 1 ca nghe nhầm phương ngữ. Số đầu của tên file là province_code,
# nên bảng in ra cuối script tự kiểm chứng được là đã lấy đúng file.
DEFAULT_FILENAMES: Tuple[str, ...] = (
    "73_0332.wav",  # Quảng Bình  · repetition loop · WER 452%
    "81_0303.wav",  # Gia Lai     · repetition loop · WER 398%
    "77_0282.wav",  # Bình Định   · chèn đoạn mở đầu · WER 95%
    "76_0327.wav",  # Quảng Ngãi  · mất nửa đầu · WER 73%
    "38_0284.wav",  # Hà Tĩnh     · nghe nhầm phương ngữ · WER 49%
)

# Cùng pattern main.py dùng để nhận shard trên Hub.
_SHARD_NAME_RE = re.compile(
    r"^data/(?P<split>train|valid|test)-(?P<index>\d+)-of-\d+\.parquet$"
)

# Tên split chính thức của ViMD là train / valid / test (không có "validation").
# Quét test trước: các mẫu WER đều đến từ split test.
SPLIT_ORDER: Tuple[str, ...] = ("test", "valid", "train")

METADATA_COLUMNS: Tuple[str, ...] = (
    "region",
    "province_code",
    "province_name",
    "filename",
    "text",
    "speakerID",
    "gender",
)

TARGET_RATE = 16000
INDEX_NAME = "_filename_index.json"
MANIFEST_NAME = "samples.json"


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value.strip()


def _key(filename: str) -> str:
    """Khoá so khớp: bỏ đuôi .wav, bỏ thư mục, hạ chữ thường.

    Không rõ cột `filename` upstream có kèm đuôi .wav hay không, và người dùng gõ kiểu
    nào cũng được - so theo stem thì cả hai đều khớp."""
    stem = Path(str(filename).strip().replace("\\", "/")).name
    if stem.lower().endswith(".wav"):
        stem = stem[:-4]
    return stem.lower()


# ===========================================================================
# 2. Tiện ích chung
# ===========================================================================

def _retry(operation, description: str, attempts: int):
    """Thử lại với backoff tăng dần. Ctrl-C không phải lỗi mạng nên không bị nuốt."""
    last_error: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:  # noqa: BLE001 - lỗi mạng vốn muôn hình vạn trạng
            last_error = exc
            wait = min(60.0, 5.0 * (2 ** (attempt - 1)))
            LOGGER.warning(
                "%s thất bại lần %d/%d (%s)%s",
                description, attempt, attempts, exc,
                f"; thử lại sau {wait:.0f}s" if attempt < attempts else "",
            )
            if attempt < attempts:
                time.sleep(wait)
    raise RuntimeError(f"{description} thất bại sau {attempts} lần thử") from last_error


def _read_json(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _write_wav_int16(path: Path, pcm: np.ndarray, rate: int) -> None:
    """Ghi WAV PCM 16-bit mono bằng stdlib - không phụ thuộc libsndfile."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(rate))
        handle.writeframes(np.asarray(pcm, dtype="<i2").tobytes())


def _decode_wav_bytes(raw: bytes) -> Tuple[np.ndarray, int]:
    """Giải mã WAV thành float32 trong [-1, 1], kèm sample rate gốc."""
    try:
        import soundfile as sf
        import io

        data, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        return np.asarray(data, dtype=np.float32), int(rate)
    except Exception:  # pragma: no cover - dự phòng khi libsndfile không mở được
        import io

        with wave.open(io.BytesIO(raw), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
        if width != 2:
            raise RuntimeError(f"độ rộng mẫu WAV không hỗ trợ: {width} byte")
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            samples = samples.reshape(-1, channels)
        return samples, rate


def _to_target_mono_int16(samples: np.ndarray, rate: int, target_rate: int) -> np.ndarray:
    """Trộn về mono và resample - đúng phép biến đổi main.py áp lên dữ liệu huấn luyện."""
    from scipy.signal import resample_poly

    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    samples = np.asarray(samples, dtype=np.float32)
    if int(rate) != int(target_rate):
        divisor = math.gcd(int(rate), int(target_rate))
        samples = resample_poly(
            samples, target_rate // divisor, int(rate) // divisor
        ).astype(np.float32)
    samples = np.clip(samples, -1.0, 1.0)
    return np.rint(samples * 32767.0).astype("<i2")


def _wav_shape(raw: bytes) -> Tuple[Optional[float], Optional[int], Optional[int]]:
    """(thời lượng giây, sample rate, số kênh) đọc từ header - không giải mã mẫu nào."""
    import io

    try:
        with wave.open(io.BytesIO(raw), "rb") as handle:
            rate = handle.getframerate()
            frames = handle.getnframes()
            channels = handle.getnchannels()
        return (float(frames) / rate if rate else None, int(rate), int(channels))
    except Exception:
        try:
            import soundfile as sf

            info = sf.info(io.BytesIO(raw))
            return (float(info.duration), int(info.samplerate), int(info.channels))
        except Exception:
            return (None, None, None)


def _audio_bytes_from_column(column: Any, row: int) -> bytes:
    """ViMD lưu audio dưới dạng HuggingFace Audio feature, tức struct {bytes, path}."""
    import pyarrow as pa

    # Đọc một row group trả về Table, nên cột là ChunkedArray - không có .field().
    # Gộp về một Array trước để helper nhận được cả hai dạng.
    if isinstance(column, pa.ChunkedArray):
        combined = column.combine_chunks()
        if isinstance(combined, pa.ChunkedArray):
            if combined.num_chunks == 0:
                raise RuntimeError("cột audio rỗng")
            combined = combined.chunk(0)
        column = combined
    if pa.types.is_struct(column.type):
        value = column.field("bytes")[row].as_py()
        if value is None:
            path = column.field("path")[row].as_py()
            raise RuntimeError(f"row audio không có bytes nội tuyến (path={path!r})")
        return value
    value = column[row].as_py()
    if value is None:
        raise RuntimeError("row audio rỗng")
    return value


# ===========================================================================
# 3. Đường nhanh: cache 16 kHz do main.py sinh ra
# ===========================================================================

def _collect_local(
    data_dir: Path, wanted: Sequence[str], splits: Sequence[str]
) -> Dict[str, Dict[str, Any]]:
    """Tìm các utterance trong data/vimd_16k/<split>/shard_*.jsonl.

    Trả về {khoá: bản ghi kèm pcm int16 16 kHz}. Đây chính xác là tín hiệu model đã
    nghe, và không tốn một byte mạng nào."""
    found: Dict[str, Dict[str, Any]] = {}
    remaining = set(wanted)
    if not data_dir.is_dir():
        return found

    for split in splits:
        if not remaining:
            break
        split_dir = data_dir / split
        if not split_dir.is_dir():
            continue
        for done in sorted(split_dir.glob("shard_*.done")):
            if not remaining:
                break
            jsonl_path = done.with_suffix(".jsonl")
            bin_path = done.with_suffix(".bin")
            if not jsonl_path.exists() or not bin_path.exists():
                continue
            hits: List[Dict[str, Any]] = []
            with open(jsonl_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    key = _key(record.get("filename") or "")
                    if key in remaining:
                        record["_key"] = key
                        hits.append(record)
            if not hits:
                continue
            mapped = np.memmap(bin_path, dtype="<i2", mode="r")
            for record in hits:
                start = int(record["offset"]) // 2
                pcm = np.asarray(
                    mapped[start:start + int(record["num_samples"])], dtype="<i2"
                )
                key = record.pop("_key")
                found[key] = {
                    "key": key,
                    "filename": record.get("filename"),
                    "split": split,
                    "shard": done.stem,
                    "source": "local-cache",
                    "sampling_rate": TARGET_RATE,
                    "channels": 1,
                    "duration": float(pcm.shape[0]) / TARGET_RATE,
                    "reference": record.get("text"),
                    "metadata": {
                        name: record.get(name)
                        for name in METADATA_COLUMNS
                        if name in record and name not in ("filename", "text")
                    },
                    "pcm16k": pcm,
                    "wav_bytes": None,
                }
                remaining.discard(key)
            del mapped
    return found


# ===========================================================================
# 4. Đường Hub: đọc parquet từ xa, chỉ kéo row group cần thiết
# ===========================================================================

class HubReader:
    """Đọc shard parquet trên Hub, ưu tiên streaming; tải cả file là phương án cuối."""

    def __init__(self, dataset_id: str, revision: str, token: Optional[str], retries: int,
                 scratch: Path) -> None:
        self.dataset_id = dataset_id
        self.revision = revision
        self.token = token
        self.retries = retries
        self.scratch = scratch
        self._fs = None
        self._streaming = True
        # Đường fallback: giữ shard đã tải cho tới khi xong hẳn shard đó, nếu không
        # việc quét filename và việc đọc row group sẽ tải cùng một file hai lần.
        self._local: Dict[str, Path] = {}

    def _filesystem(self):
        if self._fs is None:
            from huggingface_hub import HfFileSystem

            self._fs = HfFileSystem(token=self.token)
        return self._fs

    def open_parquet(self, filename: str):
        """Trả về (ParquetFile, dọn_dẹp). Streaming thì không có gì để dọn."""
        import pyarrow.parquet as pq

        if self._streaming:
            try:
                fs = self._filesystem()
                path = f"datasets/{self.dataset_id}@{self.revision}/{filename}"
                handle = _retry(
                    lambda: fs.open(path, "rb"), f"mở {filename} qua mạng", self.retries
                )
                parquet_file = pq.ParquetFile(handle)
                return parquet_file, handle.close
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:  # noqa: BLE001
                LOGGER.warning(
                    "không đọc trực tiếp được %s (%s); chuyển sang tải nguyên shard - "
                    "chậm hơn nhiều", filename, exc,
                )
                self._streaming = False

        from huggingface_hub import hf_hub_download

        local = self._local.get(filename)
        if local is None or not local.exists():
            local = Path(
                _retry(
                    lambda: hf_hub_download(
                        repo_id=self.dataset_id,
                        filename=filename,
                        repo_type="dataset",
                        revision=self.revision,
                        local_dir=str(self.scratch),
                    ),
                    f"tải {filename}",
                    self.retries,
                )
            )
            self._local[filename] = local
        # File tải về được giữ lại cho lần mở sau của cùng shard; chỉ đóng reader.
        parquet_file = pq.ParquetFile(str(local))
        return parquet_file, parquet_file.close

    def release(self, filename: Optional[str] = None) -> None:
        """Xoá shard đã tải về. Không làm gì ở chế độ streaming vì không có file nào."""
        for name in ([filename] if filename else list(self._local)):
            local = self._local.pop(name, None)
            if local is None:
                continue
            try:
                local.unlink()
            except OSError:
                pass


def _list_shards(dataset_id: str, revision: str, token: Optional[str], retries: int,
                 splits: Sequence[str]) -> List[Tuple[str, str]]:
    """[(split, tên file parquet)] theo đúng thứ tự quét mong muốn."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    files = _retry(
        lambda: api.list_repo_files(dataset_id, repo_type="dataset", revision=revision),
        f"liệt kê file của {dataset_id}",
        retries,
    )
    by_split: Dict[str, List[Tuple[int, str]]] = {}
    for name in files:
        match = _SHARD_NAME_RE.match(name)
        if match:
            by_split.setdefault(match.group("split"), []).append(
                (int(match.group("index")), name)
            )
    ordered: List[Tuple[str, str]] = []
    for split in splits:
        for _, name in sorted(by_split.get(split, [])):
            ordered.append((split, name))
    if not ordered:
        raise RuntimeError(
            f"{dataset_id}@{revision} không có shard parquet nào khớp "
            f"data/<split>-NNNNN-of-NNNNN.parquet. Kiểm tra VIMD_DATASET_ID."
        )
    return ordered


def _shard_filenames(reader: HubReader, shard: str, index: Dict[str, Any]) -> Dict[str, Any]:
    """Danh sách filename của shard, kèm kích thước từng row group.

    Chỉ đọc cột `filename` - vài chục KB - nên quét cả split vẫn rất rẻ. Kết quả được
    cache lại để lần chạy sau tìm utterance khác là tức thì."""
    cached = index.get(shard)
    if isinstance(cached, dict) and "names" in cached and "rg_sizes" in cached:
        return cached

    parquet_file, cleanup = reader.open_parquet(shard)
    try:
        if "filename" not in parquet_file.schema_arrow.names:
            raise RuntimeError(
                f"{shard} không có cột `filename`; schema là "
                f"{sorted(parquet_file.schema_arrow.names)}"
            )
        names: List[str] = []
        rg_sizes: List[int] = []
        for group in range(parquet_file.num_row_groups):
            column = parquet_file.read_row_group(group, columns=["filename"]).column("filename")
            values = column.to_pylist()
            names.extend("" if v is None else str(v) for v in values)
            rg_sizes.append(len(values))
    finally:
        cleanup()

    entry = {"names": names, "rg_sizes": rg_sizes}
    index[shard] = entry
    return entry


def _locate(entry: Dict[str, Any], wanted: Iterable[str]) -> Dict[str, Tuple[int, int, str]]:
    """{khoá: (row group, row trong row group, filename gốc)} cho các khoá có trong shard."""
    targets = set(wanted)
    hits: Dict[str, Tuple[int, int, str]] = {}
    if not targets:
        return hits
    names: List[str] = entry["names"]
    rg_sizes: List[int] = entry["rg_sizes"]
    position = 0
    for group, size in enumerate(rg_sizes):
        for row in range(size):
            if position + row >= len(names):
                break
            name = names[position + row]
            key = _key(name)
            if key in targets and key not in hits:
                hits[key] = (group, row, name)
        position += size
        if len(hits) == len(targets):
            break
    return hits


def _collect_hub(
    dataset_id: str,
    revision: str,
    token: Optional[str],
    retries: int,
    scratch: Path,
    index_path: Path,
    wanted: Sequence[str],
    splits: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    """Tìm và lấy các utterance từ parquet trên Hub, quét theo thứ tự split đã cho."""
    found: Dict[str, Dict[str, Any]] = {}
    remaining = set(wanted)
    if not remaining:
        return found

    cache = _read_json(index_path)
    if not (isinstance(cache, dict)
            and cache.get("dataset_id") == dataset_id
            and cache.get("revision") == revision
            and isinstance(cache.get("shards"), dict)):
        cache = {"dataset_id": dataset_id, "revision": revision, "shards": {}}
    index: Dict[str, Any] = cache["shards"]
    index_dirty = False

    reader = HubReader(dataset_id, revision, token, retries, scratch)
    shards = _list_shards(dataset_id, revision, token, retries, splits)
    LOGGER.info("quét %d shard trên Hub, dừng ngay khi đủ %d file", len(shards), len(remaining))

    try:
        for split, shard in shards:
            if not remaining:
                break
            try:
                if shard not in index:
                    index_dirty = True
                entry = _shard_filenames(reader, shard, index)
                hits = _locate(entry, remaining)
                if not hits:
                    LOGGER.info("%s: không có utterance nào cần tìm", shard)
                    continue

                LOGGER.info("%s: tìm thấy %s", shard, ", ".join(sorted(hits)))
                _extract_hits(reader, split, shard, hits, found)
                remaining.difference_update(hits)
            finally:
                # Ở chế độ streaming không có gì để xoá; ở chế độ fallback thì shard vừa
                # tải về được giữ đúng trong phạm vi vòng lặp này rồi bỏ đi.
                reader.release(shard)
    finally:
        reader.release()
        if index_dirty:
            cache["shards"] = index
            _write_json_atomic(index_path, cache)
    return found


def _extract_hits(
    reader: "HubReader",
    split: str,
    shard: str,
    hits: Dict[str, Tuple[int, int, str]],
    found: Dict[str, Dict[str, Any]],
) -> None:
    """Đọc đúng những row group chứa hit và bóc audio + metadata của từng row."""
    parquet_file, cleanup = reader.open_parquet(shard)
    try:
        columns = parquet_file.schema_arrow.names
        if "audio" not in columns:
            raise RuntimeError(f"{shard} không có cột `audio`")
        present = [name for name in METADATA_COLUMNS if name in columns]

        # Gom theo row group: mỗi row group chỉ đọc một lần dù có nhiều hit trong đó.
        by_group: Dict[int, List[Tuple[str, int, str]]] = {}
        for key, (group, row, name) in hits.items():
            by_group.setdefault(group, []).append((key, row, name))

        for group, items in sorted(by_group.items()):
            table = parquet_file.read_row_group(group, columns=["audio", *present])
            audio_column = table.column("audio")
            metadata = {name: table.column(name).to_pylist() for name in present}
            for key, row, name in items:
                raw = _audio_bytes_from_column(audio_column, row)
                duration, rate, channels = _wav_shape(raw)
                found[key] = {
                    "key": key,
                    "filename": name,
                    "split": split,
                    "shard": shard,
                    "source": "hub-parquet",
                    "sampling_rate": rate,
                    "channels": channels,
                    "duration": duration,
                    "reference": metadata["text"][row] if "text" in metadata else None,
                    "metadata": {
                        n: metadata[n][row] for n in present if n not in ("filename", "text")
                    },
                    "pcm16k": None,
                    "wav_bytes": raw,
                }
    finally:
        cleanup()


def _resolve_revision(dataset_id: str, data_dir: Path, override: str,
                      token: Optional[str], retries: int) -> str:
    """Ưu tiên revision main.py đã ghim, để audio khớp đúng lần eval đã sinh ra bảng WER."""
    if override:
        LOGGER.info("dataset %s @ %s (VIMD_DATASET_REVISION)", dataset_id, override)
        return override
    pinned = _read_json(data_dir / "dataset_revision.json")
    if (isinstance(pinned, dict) and pinned.get("dataset_id") == dataset_id
            and pinned.get("revision")):
        revision = str(pinned["revision"])
        LOGGER.info("dataset %s @ %s (ghim sẵn trong dataset_revision.json)", dataset_id, revision)
        return revision

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    info = _retry(
        lambda: api.dataset_info(dataset_id), f"tra cứu {dataset_id}", retries
    )
    LOGGER.info("dataset %s @ %s (HEAD trên Hub)", dataset_id, info.sha)
    return str(info.sha)


# ===========================================================================
# 5. Ghi file và báo cáo
# ===========================================================================

def _write_outputs(record: Dict[str, Any], out_dir: Path, also_16k: bool) -> Dict[str, Any]:
    stem = record["key"]
    wav_path = out_dir / f"{stem}.wav"
    out_dir.mkdir(parents=True, exist_ok=True)

    if record["wav_bytes"] is not None:
        # Ghi nguyên bytes trong parquet: không giải mã, không mã hoá lại, không mất chất lượng.
        wav_path.write_bytes(record["wav_bytes"])
    else:
        _write_wav_int16(wav_path, record["pcm16k"], TARGET_RATE)

    entry = {
        "filename": record["filename"],
        "split": record["split"],
        "shard": record["shard"],
        "source": record["source"],
        "audio_path": str(wav_path.relative_to(ROOT)) if _under(wav_path, ROOT) else str(wav_path),
        "sampling_rate": record["sampling_rate"],
        "channels": record["channels"],
        "duration": record["duration"],
        "reference": record["reference"],
    }
    entry.update(record["metadata"])

    # Cache local vốn đã là mono 16 kHz, nên bản .16k.wav sẽ giống hệt - không ghi trùng.
    if also_16k and record["wav_bytes"] is not None:
        path_16k = out_dir / f"{stem}.16k.wav"
        samples, rate = _decode_wav_bytes(record["wav_bytes"])
        _write_wav_int16(path_16k, _to_target_mono_int16(samples, rate, TARGET_RATE), TARGET_RATE)
        entry["audio_16k_path"] = (
            str(path_16k.relative_to(ROOT)) if _under(path_16k, ROOT) else str(path_16k)
        )
    return entry


def _under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _report(entries: Dict[str, Dict[str, Any]], order: Sequence[str]) -> None:
    print()
    print("=" * 78)
    for key in order:
        entry = entries.get(key)
        if entry is None:
            continue
        duration = entry.get("duration")
        head = "  ".join(
            part for part in (
                str(entry.get("filename") or key),
                entry.get("split") or "",
                str(entry.get("province_name") or ""),
                str(entry.get("region") or ""),
                f"{duration:.1f}s" if isinstance(duration, (int, float)) else "",
                f"{entry.get('sampling_rate')} Hz" if entry.get("sampling_rate") else "",
            ) if part
        )
        print(head)
        print(f"    -> {entry.get('audio_path')}")
        if entry.get("audio_16k_path"):
            print(f"    -> {entry['audio_16k_path']}")
        reference = entry.get("reference")
        if reference:
            print(f"    reference: {reference}")
        print("-" * 78)


# ===========================================================================
# 6. main
# ===========================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tải lẻ vài file audio ViMD về để nghe.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "filenames", nargs="*",
        help="tên utterance, có hoặc không có đuôi .wav (mặc định: 5 mẫu lỗi trong phần phân tích)",
    )
    parser.add_argument(
        "--out", default=None,
        help="thư mục đích (mặc định: <VIMD_OUTPUT_DIR>/audio_samples)",
    )
    parser.add_argument(
        "--split", action="append", choices=list(SPLIT_ORDER), default=None,
        help="chỉ tìm trong split này (lặp lại được; mặc định: test, valid, train)",
    )
    parser.add_argument(
        "--source", choices=("auto", "local", "hub"), default="auto",
        help="auto: thử cache 16 kHz trước rồi mới lên Hub (mặc định)",
    )
    parser.add_argument(
        "--also-16k", action="store_true",
        help="ghi thêm bản mono 16 kHz - đúng tín hiệu model đã nghe",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="tải lại kể cả khi file đã có sẵn trong thư mục đích",
    )
    parser.add_argument(
        "--retries", type=int, default=int(_env_str("VIMD_DOWNLOAD_RETRIES", "5")),
        help="số lần thử lại cho mỗi thao tác mạng",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    args = parse_args(argv)

    dataset_id = _env_str("VIMD_DATASET_ID", "nguyendv02/ViMD_Dataset")
    data_dir = Path(_env_str("VIMD_DATA_DIR", str(ROOT / "data" / "vimd_16k")))
    output_dir = Path(_env_str("VIMD_OUTPUT_DIR", str(ROOT / "outputs")))
    out_dir = Path(args.out) if args.out else output_dir / "audio_samples"
    scratch = Path(_env_str("VIMD_RAW_DIR", str(ROOT / "data" / "_raw_parquet"))) / ".vimd_scratch"
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    splits = tuple(args.split) if args.split else SPLIT_ORDER

    requested = list(args.filenames) or list(DEFAULT_FILENAMES)
    order: List[str] = []
    for name in requested:
        key = _key(name)
        if key and key not in order:
            order.append(key)
    if not order:
        LOGGER.error("không có tên file nào để tải")
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / MANIFEST_NAME
    existing = _read_json(manifest_path)
    entries: Dict[str, Dict[str, Any]] = {}
    if isinstance(existing, dict) and isinstance(existing.get("samples"), dict):
        entries = {str(k): v for k, v in existing["samples"].items() if isinstance(v, dict)}

    todo: List[str] = []
    for key in order:
        entry = entries.get(key)
        # Bản lấy từ cache local vốn đã là 16 kHz nên không có file .16k.wav riêng;
        # đòi file đó tồn tại sẽ khiến mọi lần chạy sau đều tải lại vô ích.
        from_local = isinstance(entry, dict) and entry.get("source") == "local-cache"
        has_16k = (not args.also_16k) or from_local or (out_dir / f"{key}.16k.wav").exists()
        if (not args.force) and isinstance(entry, dict) \
                and (out_dir / f"{key}.wav").exists() and has_16k:
            LOGGER.info("%s.wav đã có sẵn, bỏ qua (dùng --force để tải lại)", key)
            continue
        todo.append(key)

    collected: Dict[str, Dict[str, Any]] = {}
    if todo and args.source in ("auto", "local"):
        collected.update(_collect_local(data_dir, todo, splits))
        if collected:
            LOGGER.info("lấy từ cache 16 kHz: %s", ", ".join(sorted(collected)))
        elif args.source == "local":
            LOGGER.error(
                "không tìm thấy utterance nào trong cache %s. Cache chỉ tồn tại sau khi "
                "main.py đã chuẩn bị dataset. Bỏ --source local để tải từ Hub.", data_dir,
            )
            return 1

    missing = [key for key in todo if key not in collected]
    if missing and args.source in ("auto", "hub"):
        revision = _resolve_revision(
            dataset_id, data_dir, _env_str("VIMD_DATASET_REVISION", ""), token, args.retries
        )
        scratch.mkdir(parents=True, exist_ok=True)
        collected.update(
            _collect_hub(
                dataset_id, revision, token, args.retries, scratch,
                out_dir / INDEX_NAME, missing, splits,
            )
        )

    for key in order:
        record = collected.get(key)
        if record is not None:
            entries[key] = _write_outputs(record, out_dir, args.also_16k)

    _write_json_atomic(
        manifest_path,
        {"dataset_id": dataset_id, "samples": {k: entries[k] for k in entries}},
    )
    _report(entries, order)

    not_found = [key for key in order if key not in entries]
    if not_found:
        where = {"local": f"cache {data_dir}", "hub": f"Hub ({dataset_id})"}.get(
            args.source, f"cache {data_dir} lẫn Hub ({dataset_id})"
        )
        LOGGER.error(
            "không tìm thấy trong %s, split %s: %s",
            where, "/".join(splits), ", ".join(f"{k}.wav" for k in not_found),
        )
        return 1
    LOGGER.info("xong: %d file trong %s (manifest: %s)", len(order), out_dir, MANIFEST_NAME)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        LOGGER.warning("bị ngắt")
        sys.exit(130)
