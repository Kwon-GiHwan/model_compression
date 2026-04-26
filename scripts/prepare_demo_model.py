"""작은 사전학습 PyTorch 모델을 받아 .pt로 저장 (demo/검증용).

torchvision의 MobileNetV3-Small을 사용. fpga_simulator의 vela 컴파일 흐름까지
end-to-end 동작 가능 여부 검증을 위한 가장 작은 합리적 모델.
"""
import sys
from pathlib import Path

import torch
from torchvision import models

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/demo_model.pt")
OUT.parent.mkdir(parents=True, exist_ok=True)

print(f"[prepare_demo_model] downloading mobilenet_v3_small (pretrained)...")
m = models.mobilenet_v3_small(weights="DEFAULT")
m.eval()
torch.save(m, str(OUT))
print(f"[prepare_demo_model] saved → {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")
