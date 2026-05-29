"""Tests for overlapping segment precedence resolution."""

from app.utils.segment_resolve import (
    resolve_segments_by_precedence,
    segments_to_password_sequence,
)

# Segments from a real full-recording analysis (overlapping tap / double_tap / still).
SCREENSHOT_SEGMENTS = [
    {
        "movement_type": "still",
        "start_ms": 0,
        "end_ms": 895,
        "confidence": 1.0,
    },
    {
        "movement_type": "tap",
        "start_ms": 600,
        "end_ms": 1045,
        "confidence": 0.84,
    },
    {
        "movement_type": "double_tap",
        "start_ms": 750,
        "end_ms": 1345,
        "confidence": 0.76,
    },
    {
        "movement_type": "tap",
        "start_ms": 1050,
        "end_ms": 1495,
        "confidence": 0.70,
    },
    {
        "movement_type": "wrist_rotation",
        "start_ms": 1200,
        "end_ms": 2395,
        "confidence": 0.79,
    },
    {
        "movement_type": "still",
        "start_ms": 2100,
        "end_ms": 2545,
        "confidence": 0.81,
    },
    {
        "movement_type": "tap",
        "start_ms": 2250,
        "end_ms": 3145,
        "confidence": 0.70,
    },
]


def test_screenshot_recording_resolves_to_double_tap_wrist_rotation_tap():
    resolved = resolve_segments_by_precedence(SCREENSHOT_SEGMENTS)
    sequence = segments_to_password_sequence(resolved)

    assert sequence == ["double_tap", "wrist_rotation", "tap"]

    assert resolved[0]["movement_type"] == "still"
    assert resolved[1]["movement_type"] == "double_tap"
    assert resolved[1]["start_ms"] == 600
    assert resolved[1]["end_ms"] == 1200
    assert resolved[2]["movement_type"] == "wrist_rotation"
    assert resolved[2]["start_ms"] == 1200
    assert resolved[2]["end_ms"] == 2395
    assert resolved[3]["movement_type"] == "tap"
    assert resolved[3]["start_ms"] == 2395


def test_leading_tap_merged_into_adjacent_double_tap():
    segments = [
        {"movement_type": "tap", "start_ms": 600, "end_ms": 750, "confidence": 0.8},
        {"movement_type": "double_tap", "start_ms": 750, "end_ms": 1200, "confidence": 0.76},
    ]
    resolved = resolve_segments_by_precedence(segments)
    assert len(resolved) == 1
    assert resolved[0]["movement_type"] == "double_tap"
    assert resolved[0]["start_ms"] == 600
    assert resolved[0]["end_ms"] == 1200
