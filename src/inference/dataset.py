# -*- coding: utf-8 -*-
"""
    Complex NER Project:
"""

# ============================ Third Party libs ============================

from torch.utils.data import Dataset
from typing import List


class InferenceDataset(Dataset):
    def __init__(self, texts: List[list], subtoken_checks: List[list], tokenizer, max_length: int):
        self.texts = texts
        self.subtoken_checks = subtoken_checks
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item_index):
        inputs = self.tokenizer.encode_plus(
            text=self.texts[item_index],
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True
        )

        subtoken_check = self.tokenizer.encode_plus(
            text=self.subtoken_checks[item_index],
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True
        ).input_ids

        return {"input_ids": inputs["input_ids"].flatten(),
                "attention_mask": inputs["attention_mask"].flatten(),
                "subtoken_check": subtoken_check.flatten()}
