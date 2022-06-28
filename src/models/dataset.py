import torch
import pytorch_lightning as pl
from typing import List
from torch.utils.data import Dataset, DataLoader
from data_preparation import create_attention_masks, pad_sequence, truncate_sequence, pad_sequence_2
from utils import find_max_length_in_list


class CustomDataset(Dataset):
    def __init__(self, texts: List[list], targets: List[list],
                 subtoken_checks: List[list], max_length: int, tokenizer, target_indexer):
        self.texts = texts
        self.targets = targets
        self.subtoken_checks = subtoken_checks
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.target_indexer = target_indexer
        self.position = [str(p) for p in range(self.max_length)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item_index):
        data = self.tokenizer.encode_plus(text=self.texts[item_index],
                                          add_special_tokens=True,
                                          max_length=self.max_length,
                                          return_tensors="pt",
                                          padding="max_length",
                                          truncation=True)
        subtoken_check = self.tokenizer.encode_plus(text=self.subtoken_checks[item_index],
                                                    add_special_tokens=True,
                                                    max_length=self.max_length,
                                                    return_tensors="pt",
                                                    padding="max_length",
                                                    truncation=True).input_ids

        target = self.target_indexer.convert_samples_to_indexes([self.targets[item_index]])[0]

        return {"input_ids": data["input_ids"].flatten(),
                "target": torch.LongTensor(target),
                "attention_mask": data["attention_mask"].flatten(),
                "subtoken_check": subtoken_check.flatten()}


class LstmDataset(Dataset):
    def __init__(self, texts: List[list], targets: List[list], max_length: int,
                 tokenizer, target_indexer):
        self.texts = texts
        self.targets = targets
        self.max_length = max_length
        self.token_indexer = tokenizer
        self.target_indexer = target_indexer

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item_index):
        attention_masks = create_attention_masks(data=[self.texts[item_index]],
                                                 pad_item="[PAD]")[0]

        text = self.token_indexer.convert_samples_to_indexes([self.texts[item_index]])[0]

        target = self.target_indexer.convert_samples_to_indexes([self.targets[item_index]])[0]

        return {"input_ids": torch.LongTensor(text), "target": torch.LongTensor(target),
                "attention_mask": torch.LongTensor(attention_masks)}


class DataModule(pl.LightningDataModule):

    def __init__(self, data: dict, batch_size, max_length, tokenizer, target_indexer):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.target_indexer = target_indexer
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def setup(self):
        self.train_dataset = CustomDataset(texts=self.data["train_data"][0],
                                           targets=self.data["train_data"][1],
                                           subtoken_checks=self.data["train_data"][2],
                                           max_length=self.max_length,
                                           tokenizer=self.tokenizer,
                                           target_indexer=self.target_indexer)

        self.val_dataset = CustomDataset(texts=self.data["val_data"][0],
                                         targets=self.data["val_data"][1],
                                         subtoken_checks=self.data["val_data"][2],
                                         max_length=self.max_length,
                                         tokenizer=self.tokenizer,
                                         target_indexer=self.target_indexer)

        self.test_dataset = CustomDataset(texts=self.data["val_data"][0],
                                          targets=self.data["val_data"][1],
                                          subtoken_checks=self.data["val_data"][2],
                                          max_length=self.max_length,
                                          tokenizer=self.tokenizer,
                                          target_indexer=self.target_indexer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=10)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=10)
