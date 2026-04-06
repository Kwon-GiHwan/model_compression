import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection, ImageFolder

from model_compression.data.base_dataloader import BaseDataLoader


class ImageDataLoader(BaseDataLoader):
    """
    torchvision 기반 이미지 데이터 로더.
    로컬 ImageFolder 구조 또는 COCO 형식 지원.
    """

    def __init__(
        self,
        dataset_path: str,
        task: str = "classification",
        input_size: int = 224,
        batch_size: int = 16,
        split: str = "train",
        normalize_mean: list[float] | None = None,
        normalize_std: list[float] | None = None,
    ):
        self.dataset_path = dataset_path
        self.task = task
        self.input_size = input_size
        self.batch_size = batch_size
        self.split = split
        self.normalize_mean = normalize_mean or [0.485, 0.456, 0.406]
        self.normalize_std = normalize_std or [0.229, 0.224, 0.225]

    def get_dataloader(self) -> DataLoader:
        transform = T.Compose(
            [
                T.Resize((self.input_size, self.input_size)),
                T.ToTensor(),
                T.Normalize(mean=self.normalize_mean, std=self.normalize_std),
            ]
        )

        dataset = ImageFolder(root=self.dataset_path, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
