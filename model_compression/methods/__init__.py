"""압축 방법론 구현체 모음.

디렉토리 구조:
    methods/
      <category>/              # pruning, quantization, distillation, ...
        <name>_<type>.py       # 구현체 (예: magnitude_pruner.py)
      base_method.py           # BaseMethod ABC
      registry.py              # 등록 및 팩토리
      utils.py                 # unpack_batch, forward_and_extract_logits 등

새 방법론 추가:
    1. 적절한 <category>/ 디렉토리에 구현체 작성 (또는 새 카테고리 디렉토리 생성).
       Registry는 카테고리명을 특별 취급하지 않으므로 자유롭게 확장 가능.
    2. methods/registry.py 의 `_registry.register("<category>.<name>")(Class)` 에 한 줄 추가.
    3. .env 에서 `METHOD=<category>.<name>` 로 사용.
"""
