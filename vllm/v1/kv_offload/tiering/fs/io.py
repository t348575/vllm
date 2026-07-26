# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os
import random
import threading
import time

from simple_profiler import profiler

logger = logging.getLogger(__name__)

# O_DIRECT is Linux-specific and not available on macOS
O_DIRECT = getattr(os, "O_DIRECT", 0)

# Thread-local storage for unique temporary file suffixes
_thread_local = threading.local()


def _get_tmp_suffix() -> str:
    """Generate a thread-local unique suffix for temporary files."""
    try:
        return _thread_local.tmp_suffix
    except AttributeError:
        _thread_local.tmp_suffix = f"_{random.randint(0, 2**63 - 1)}.tmp"
        return _thread_local.tmp_suffix


def _ensure_dirs(path: str) -> None:
    """Create parent directories of *path* if they don't exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def store_block(
    dest_path: str,
    buffer: memoryview,
    offset: int,
    block_size: int,
    job_id: int = 0,
    req_id: str = "",
    file_index: int = 0,
    num_files: int = 1,
) -> None:
    """
    Store callback: Writes to a temp file then atomically replaces the destination.
    """
    # Check if block already exists to avoid redundant writes
    if os.path.exists(dest_path):
        return

    tmp_path = dest_path + _get_tmp_suffix()
    # Ensure parent directories exist
    _ensure_dirs(dest_path)

    # Write block atomically. Cast to a flat byte view so the slice uses byte
    # indices; the raw memoryview may be multi-dimensional with itemsize > 1.
    view_slice = buffer.cast("B")[offset : offset + block_size]
    try:
        fd = os.open(
            tmp_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_TRUNC | O_DIRECT,
            0o644,
        )
        try:
            start_ns = time.perf_counter_ns()
            written = os.write(fd, view_slice)
            duration_ns = time.perf_counter_ns() - start_ns
            if written < len(view_slice):
                raise OSError(
                    f"Short write: expected {len(view_slice)} bytes, "
                    f"wrote {written}"
                )
            profiler.add_event(
                "py_kvcache.file_write",
                "fs",
                start_ns,
                duration_ns,
                tid=req_id or None,
                args={
                    "job_id": job_id,
                    "req_id": req_id,
                    "file_index": file_index,
                    "num_files": num_files,
                    "num_bytes": block_size,
                },
            )
        finally:
            os.close(fd)
        os.replace(tmp_path, dest_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError as cleanup_exc:
            logger.warning(
                "Failed to remove temp file %s: %s", tmp_path, cleanup_exc
            )
        raise


def load_block(
    source_path: str,
    view: memoryview,
    offset: int,
    block_size: int,
    job_id: int = 0,
    req_id: str = "",
    file_index: int = 0,
    num_files: int = 1,
) -> None:
    """
    Load callback: read one KV block from disk. Remove the file on failure.
    """
    fd: int | None = None
    view_slice = view.cast("B")[offset : offset + block_size]
    try:
        fd = os.open(source_path, os.O_RDONLY | O_DIRECT)
        start_ns = time.perf_counter_ns()
        bytes_read = os.readv(fd, [view_slice])
        duration_ns = time.perf_counter_ns() - start_ns
        if bytes_read < block_size:
            raise OSError(
                f"Short read: expected {block_size} bytes, read {bytes_read}"
            )
        profiler.add_event(
            "py_kvcache.file_read",
            "fs",
            start_ns,
            duration_ns,
            tid=req_id or None,
            args={
                "job_id": job_id,
                "req_id": req_id,
                "file_index": file_index,
                "num_files": num_files,
                "num_bytes": block_size,
            },
        )
    except Exception:
        try:
            os.remove(source_path)
        except OSError as cleanup_exc:
            logger.warning(
                "Failed to remove unreadable file %s: %s", source_path, cleanup_exc
            )
        raise
    finally:
        if fd is not None:
            os.close(fd)
