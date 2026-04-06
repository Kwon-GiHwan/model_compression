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

    def _get_attention_layers(self, model):
        """다양한 Transformer 아키텍처의 attention layer를 탐지."""
        # BERT-style: model.encoder.layer[i].attention.self
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            bert_layers = [
                (i, layer.attention.self)
                for i, layer in enumerate(model.encoder.layer)
                if hasattr(layer, 'attention') and hasattr(layer.attention, 'self')
            ]
            if bert_layers:
                return bert_layers
        # GPT2-style: model.h[i].attn
        if hasattr(model, 'h'):
            return [
                (i, layer.attn)
                for i, layer in enumerate(model.h)
                if hasattr(layer, 'attn')
            ]
        # ViT-style: model.encoder.layer[i].attention.attention
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            layers = []
            for i, layer in enumerate(model.encoder.layer):
                if hasattr(layer, 'attention') and hasattr(layer.attention, 'attention'):
                    layers.append((i, layer.attention.attention))
            if layers:
                return layers
        raise ValueError(
            "지원하지 않는 모델 아키텍처입니다. "
            "encoder.layer[].attention.self (BERT) 또는 h[].attn (GPT2) 구조가 필요합니다."
        )

    def _get_head_params(self, attn):
        """attention 모듈에서 head 수와 QKV weight를 추출."""
        num_heads = getattr(attn, 'num_attention_heads', None) or getattr(attn, 'num_heads', None)
        if num_heads is None:
            raise ValueError("attention 모듈에서 num_heads를 찾을 수 없습니다")

        head_size = getattr(attn, 'attention_head_size', None) or getattr(attn, 'head_dim', None)
        if head_size is None:
            # Fallback: hidden_size / num_heads
            hidden = None
            for name in ('query', 'q_proj', 'c_attn'):
                proj = getattr(attn, name, None)
                if proj is not None and hasattr(proj, 'weight'):
                    hidden = proj.weight.shape[0] if name != 'c_attn' else proj.weight.shape[1]
                    break
            if hidden:
                head_size = hidden // num_heads
            else:
                raise ValueError("head_size를 결정할 수 없습니다")

        # QKV weights
        qkv_weights = []
        # BERT: query, key, value
        for name in ('query', 'key', 'value'):
            proj = getattr(attn, name, None)
            if proj is not None and hasattr(proj, 'weight'):
                qkv_weights.append(proj.weight)
        # GPT2: c_attn (combined QKV)
        if not qkv_weights:
            c_attn = getattr(attn, 'c_attn', None)
            if c_attn is not None and hasattr(c_attn, 'weight'):
                w = c_attn.weight
                chunk_size = w.shape[-1] // 3
                qkv_weights = list(w.split(chunk_size, dim=-1))
        # q_proj, k_proj, v_proj (LLaMA, etc.)
        if not qkv_weights:
            for name in ('q_proj', 'k_proj', 'v_proj'):
                proj = getattr(attn, name, None)
                if proj is not None and hasattr(proj, 'weight'):
                    qkv_weights.append(proj.weight)

        return num_heads, head_size, qkv_weights

    def apply(self, student, teacher=None, dataloader=None):
        model = copy.deepcopy(student).to("cpu").eval()

        pruned_heads = {}
        for layer_idx, attn in self._get_attention_layers(model):
            num_heads, head_size, qkv_weights = self._get_head_params(attn)
            num_to_prune = max(1, int(num_heads * self.pruning_ratio))

            # 각 head의 중요도 = weight norm 합산
            head_importance = torch.zeros(num_heads)
            for i in range(num_heads):
                start = i * head_size
                end = (i + 1) * head_size
                importance = sum(w[start:end].norm() for w in qkv_weights)
                head_importance[i] = importance

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
