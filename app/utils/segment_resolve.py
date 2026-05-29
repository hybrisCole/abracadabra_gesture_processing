"""Resolve overlapping gesture segments by movement precedence."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

# Higher rank wins when intervals overlap.
MOVEMENT_PRECEDENCE: Dict[str, int] = {
    "wrist_rotation": 4,
    "double_tap": 3,
    "tap": 2,
    "still": 1,
    "silence": 1,
}

NON_PASSWORD_MOVEMENTS = frozenset({"still", "silence"})

# Gestures below this confidence are excluded from resolve + password sequence.
MIN_GESTURE_CONFIDENCE = 0.5


def segment_meets_confidence(
    segment: Dict[str, Any],
    min_confidence: float = MIN_GESTURE_CONFIDENCE,
) -> bool:
    return float(segment.get("confidence", 0.0)) >= min_confidence


def filter_segments_by_confidence(
    segments: Sequence[Dict[str, Any]],
    min_confidence: float = MIN_GESTURE_CONFIDENCE,
) -> List[Dict[str, Any]]:
    return [segment for segment in segments if segment_meets_confidence(segment, min_confidence)]


def _precedence_rank(movement_type: str) -> int:
    return MOVEMENT_PRECEDENCE.get(movement_type, 0)


def _active_segments(
    segments: Sequence[Dict[str, Any]], start_ms: float, end_ms: float
) -> List[Dict[str, Any]]:
    return [
        segment
        for segment in segments
        if float(segment["start_ms"]) < end_ms and float(segment["end_ms"]) > start_ms
    ]


def _pick_winner(active: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return max(
        active,
        key=lambda segment: (
            _precedence_rank(str(segment["movement_type"])),
            float(segment.get("confidence", 0.0)),
        ),
    )


def resolve_segments_by_precedence(
    segments: Sequence[Dict[str, Any]],
    *,
    min_confidence: float = MIN_GESTURE_CONFIDENCE,
) -> List[Dict[str, Any]]:
    """Merge overlapping raw segments into a non-overlapping timeline.

    When multiple segments cover the same time range, the label with the highest
    precedence wins (wrist_rotation > double_tap > tap > still).
    Segments below ``min_confidence`` are ignored.
    """
    segments = filter_segments_by_confidence(segments, min_confidence)
    if not segments:
        return []

    boundaries = sorted(
        {float(segment["start_ms"]) for segment in segments}
        | {float(segment["end_ms"]) for segment in segments}
    )

    slices: List[Dict[str, Any]] = []
    for index in range(len(boundaries) - 1):
        start_ms = boundaries[index]
        end_ms = boundaries[index + 1]
        if end_ms <= start_ms:
            continue

        active = _active_segments(segments, start_ms, end_ms)
        if not active:
            continue

        winner = _pick_winner(active)
        slices.append(
            {
                "movement_type": winner["movement_type"],
                "start_ms": start_ms,
                "end_ms": end_ms,
                "center_ms": (start_ms + end_ms) / 2.0,
                "duration_ms": end_ms - start_ms,
                "confidence": float(winner.get("confidence", 0.0)),
            }
        )

    merged: List[Dict[str, Any]] = []
    for piece in slices:
        if merged and merged[-1]["movement_type"] == piece["movement_type"]:
            merged[-1]["end_ms"] = piece["end_ms"]
            merged[-1]["duration_ms"] = merged[-1]["end_ms"] - merged[-1]["start_ms"]
            merged[-1]["center_ms"] = (
                merged[-1]["start_ms"] + merged[-1]["end_ms"]
            ) / 2.0
            merged[-1]["confidence"] = max(
                float(merged[-1]["confidence"]), float(piece["confidence"])
            )
        else:
            merged.append(dict(piece))

    merged = _merge_leading_tap_into_double_tap(merged)
    return filter_segments_by_confidence(merged, min_confidence)


def _merge_leading_tap_into_double_tap(
    segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Absorb a tap slice that immediately precedes double_tap (first hit artifact)."""
    if len(segments) < 2:
        return segments

    merged: List[Dict[str, Any]] = []
    index = 0
    while index < len(segments):
        current = segments[index]
        if (
            index + 1 < len(segments)
            and current["movement_type"] == "tap"
            and segments[index + 1]["movement_type"] == "double_tap"
            and float(current["end_ms"]) == float(segments[index + 1]["start_ms"])
        ):
            double_tap = dict(segments[index + 1])
            double_tap["start_ms"] = float(current["start_ms"])
            double_tap["duration_ms"] = float(double_tap["end_ms"]) - float(
                double_tap["start_ms"]
            )
            double_tap["center_ms"] = (
                float(double_tap["start_ms"]) + float(double_tap["end_ms"])
            ) / 2.0
            double_tap["confidence"] = max(
                float(current.get("confidence", 0.0)),
                float(double_tap.get("confidence", 0.0)),
            )
            merged.append(double_tap)
            index += 2
            continue

        merged.append(dict(current))
        index += 1

    return merged


def segments_to_password_sequence(
    segments: Sequence[Dict[str, Any]],
    *,
    include_still: bool = False,
    min_confidence: float = MIN_GESTURE_CONFIDENCE,
) -> List[str]:
    """Gesture password order: one label per contiguous resolved segment, time order."""
    sequence: List[str] = []
    for segment in segments:
        if not segment_meets_confidence(segment, min_confidence):
            continue
        movement = str(segment["movement_type"])
        if not include_still and movement in NON_PASSWORD_MOVEMENTS:
            continue
        sequence.append(movement)
    return sequence
