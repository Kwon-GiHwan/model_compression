"""압축 방법론 공용 유틸리티."""
import torch


def extract_logits(output: object) -> torch.Tensor:
    """모델 출력에서 logits 추출. HuggingFace 출력과 순수 텐서 모두 지원."""
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def unpack_batch(
    batch: dict | list | tuple, device: str, label_key: str = "label"
) -> tuple[dict | torch.Tensor, torch.Tensor]:
    """batch에서 inputs/labels를 분리하고 device로 이동.

    이미지: (tensor, labels) → tensor, labels
    NLP: {"input_ids": ..., "label": ...} → {k: v}, labels
    """
    if isinstance(batch, (list, tuple)):
        return batch[0].to(device), batch[1].to(device)
    inputs = {k: v.to(device) for k, v in batch.items() if k != label_key}
    labels = batch[label_key].to(device)
    return inputs, labels


def forward_and_extract_logits(model: torch.nn.Module, inputs: dict | torch.Tensor) -> torch.Tensor:
    """inputs 타입에 따라 model(inputs) 또는 model(**inputs) 호출 후 logits 추출."""
    if isinstance(inputs, dict):
        return extract_logits(model(**inputs))
    return extract_logits(model(inputs))
