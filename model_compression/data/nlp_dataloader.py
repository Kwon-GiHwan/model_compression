from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import Config
from model_compression.data.base_dataloader import BaseDataLoader


class NLPDataLoader(BaseDataLoader):
    """
    HuggingFace datasets 기반 NLP 데이터 로더.
    dataset name과 config만 있으면 자동 로드.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer_path: str,
        dataset_config: str = None,
        split: str = "train",
        batch_size: int = 16,
        max_length: int = 128,
        text_column: str | None = None,
        label_column: str = "label",
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.tokenizer_path = tokenizer_path
        self.split = split
        self.batch_size = batch_size
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column

    @classmethod
    def from_config(cls, config: Config) -> "NLPDataLoader":
        return cls(
            dataset_name=config.data.name,
            tokenizer_path=config.MODEL_PATH,
            dataset_config=config.data.config,
            split=config.data.split,
            batch_size=config.data.batch_size,
            max_length=config.data.max_length,
        )

    def get_dataloader(self) -> DataLoader:
        dataset = load_dataset(self.dataset_name, self.dataset_config, split=self.split)
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        def tokenize(batch):
            if self.text_column:
                text_col = self.text_column
            else:
                # 텍스트 컬럼 자동 탐지
                text_col = next(
                    (c for c in ["text", "sentence", "document", "question"] if c in batch),
                    list(batch.keys())[0],
                )
            return tokenizer(
                batch[text_col],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )

        dataset = dataset.map(tokenize, batched=True)
        columns = ["input_ids", "attention_mask"]
        try:
            if self.label_column in dataset.column_names:
                columns.append(self.label_column)
        except TypeError:
            columns.append(self.label_column)
        dataset.set_format(type="torch", columns=columns)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
