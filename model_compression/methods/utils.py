"""압축 방법론 공용 유틸리티."""
from __future__ import annotations
from typing import Any


def extract_logits(output: Any) -> Any:
    """모델 출력에서 logits 추출. HuggingFace 출력과 순수 텐서 모두 지원."""
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)):
        return output[0]
    return output
