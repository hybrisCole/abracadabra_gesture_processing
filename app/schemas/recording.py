from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


MovementType = Literal["tap", "double_tap", "still", "silence", "wrist_rotation"]


class ImuSampleIn(BaseModel):
    """One decoded IMU sample from abracadabra-rnapp / BLE payload."""

    t_ms: int = Field(..., ge=0)
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float


class RecordingWindowIn(BaseModel):
    """A cropped gesture window or a full recording from the RN app."""

    window_id: Optional[int] = None
    recording_id: Optional[str] = None
    samples: List[ImuSampleIn] = Field(..., min_length=10)
    sample_rate_hz: Optional[float] = Field(default=None, gt=0)


class LabeledRecordingWindowIn(RecordingWindowIn):
    """A crop saved as training data for one movement label."""

    movement_type: MovementType

    @field_validator("movement_type")
    @classmethod
    def normalize_movement_type(cls, value: str) -> str:
        value = value.strip().lower()
        return "still" if value == "silence" else value


class AnalyzeRecordingIn(RecordingWindowIn):
    """A full 3-4 second recording to segment into timed gestures."""

    window_size_ms: int = Field(default=450, ge=100, le=1500)
    overlap_ms: int = Field(default=300, ge=0, le=1400)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_segment_windows: int = Field(default=1, ge=1, le=20)
    include_still: bool = True

    @field_validator("overlap_ms")
    @classmethod
    def overlap_must_leave_step(cls, value: int, info) -> int:
        window_size = info.data.get("window_size_ms")
        if window_size is not None and value >= window_size:
            raise ValueError("overlap_ms must be smaller than window_size_ms")
        return value


class ExpectedGestureIn(BaseModel):
    movement_type: MovementType
    min_start_ms: Optional[int] = Field(default=None, ge=0)
    max_start_ms: Optional[int] = Field(default=None, ge=0)
    max_gap_ms: Optional[int] = Field(default=None, ge=0)

    @field_validator("movement_type")
    @classmethod
    def normalize_movement_type(cls, value: str) -> str:
        value = value.strip().lower()
        return "still" if value == "silence" else value


class VerifyGesturePasswordIn(AnalyzeRecordingIn):
    """Analyze a recording and compare its non-still sequence to an expected one."""

    expected_sequence: List[ExpectedGestureIn] = Field(..., min_length=1)
