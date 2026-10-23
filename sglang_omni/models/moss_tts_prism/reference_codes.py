# SPDX-License-Identifier: Apache-2.0
"""Prism reference codes stored in Lance tables, imported only for lance:// references."""

from __future__ import annotations

import functools
from urllib.parse import parse_qs, unquote, urlsplit

import lance
import pyarrow as pa
import torch

MAX_REFERENCE_FRAMES = 1250  # 100 s at the codec's 12.5 Hz, the reference audio cap
MAX_OPEN_TABLES = 1024


@functools.lru_cache(maxsize=MAX_OPEN_TABLES)
def open_reference_table(path: str, version: int) -> lance.LanceDataset:
    return lance.dataset(path, version=version)


def load_reference_codes(
    uri: str, *, n_vq: int, speech_vocab_size: int
) -> torch.Tensor:
    """Read lance:///table?version=V&row=R&column=C as [frames, n_vq] codes."""
    location = urlsplit(uri)
    query = parse_qs(location.query, keep_blank_values=True)
    if (
        location.netloc
        or not location.path.startswith("/")
        or location.fragment
        or sorted(query) != ["column", "row", "version"]
        or any(len(values) != 1 for values in query.values())
        or not query["version"][0].isdigit()
        or not query["row"][0].isdigit()
        or not query["column"][0]
    ):
        raise ValueError(
            f"Prism reference {uri!r} must be "
            "lance:///<table>?version=<version>&row=<row>&column=<column>"
        )
    column = query["column"][0]
    table = open_reference_table(unquote(location.path), int(query["version"][0]))
    frames = (
        table.take([int(query["row"][0])], columns=[column])
        .column(column)
        .combine_chunks()
    )
    frame_type = frames.type.value_type if pa.types.is_list(frames.type) else None
    if (
        frame_type is None
        or not pa.types.is_fixed_size_list(frame_type)
        or not pa.types.is_integer(frame_type.value_type)
        or frames.null_count
        or frames.flatten().null_count
        or frames.flatten().flatten().null_count
    ):
        raise ValueError(
            f"Prism reference {uri!r} must hold a list of fixed-size integer frames"
        )
    codes = torch.tensor(
        frames.flatten().flatten().to_numpy(), dtype=torch.long
    ).reshape(-1, frame_type.list_size)
    if frame_type.list_size < n_vq:
        raise ValueError(
            f"Prism reference {uri!r} has {frame_type.list_size} quantizers, "
            f"fewer than n_vq={n_vq}"
        )
    elif not 1 <= len(codes) <= MAX_REFERENCE_FRAMES:
        raise ValueError(
            f"Prism reference {uri!r} has {len(codes)} frames, "
            f"outside 1..{MAX_REFERENCE_FRAMES}"
        )
    elif codes[:, :n_vq].min() < 0 or codes[:, :n_vq].max() >= speech_vocab_size:
        raise ValueError(
            f"Prism reference {uri!r} has codes outside [0, {speech_vocab_size})"
        )
    else:
        return codes[:, :n_vq].contiguous()
