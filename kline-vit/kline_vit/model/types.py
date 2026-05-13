from dataclasses import dataclass


@dataclass
class InferenceResult:
    image_path: str
    buy_probability: float
    label: int
    inference_time_ms: float
