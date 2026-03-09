import copy

import torch

from model_compression.methods.base_method import BaseMethod


class AttentionHeadPruner(BaseMethod):
    """
    Transformer Attention Head Pruning.
    중요도가 낮은 attention head를 제거.
    NLP / Vision Transformer 모두 적용 가능.
    """

    def __init__(self, pruning_ratio: float):
        self.pruning_ratio = pruning_ratio

    def apply(self, student, teacher=None, dataloader=None):
        model = copy.deepcopy(student).to("cpu").eval()

        pruned_heads = {}
        for layer_idx, layer in enumerate(model.encoder.layer):
            attn = layer.attention.self
            num_heads = attn.num_attention_heads
            num_to_prune = max(1, int(num_heads * self.pruning_ratio))

            # 각 head의 중요도 = weight norm 합산
            head_importance = torch.zeros(num_heads)
            for i in range(num_heads):
                head_size = attn.attention_head_size
                start = i * head_size
                end = (i + 1) * head_size
                head_importance[i] = (
                    attn.query.weight[start:end].norm()
                    + attn.key.weight[start:end].norm()
                    + attn.value.weight[start:end].norm()
                )

            # 중요도 낮은 head 선택
            heads_to_prune = head_importance.argsort()[:num_to_prune].tolist()
            pruned_heads[layer_idx] = heads_to_prune

        model.prune_heads(pruned_heads)
        print(f"[AttentionHeadPruner] 완료: {pruned_heads}")
        return model

    def validate(self, config):
        if config.MODEL_TYPE != "huggingface":
            raise ValueError(
                "AttentionHeadPruner는 HuggingFace Transformer 모델만 지원합니다"
            )
