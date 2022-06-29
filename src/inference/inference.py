# -*- coding: utf-8 -*-
"""
This module create to write Inference class. With Inference class we
can inference with proposed complex ner model
"""

# ============================ Third Party libs ============================
from typing import List
import numpy

# ============================ My packages ============================
from configuration import BaseConfig

CONFIG_CLASS = BaseConfig()
CONFIG = CONFIG_CLASS.get_config()


class Inference:
    """
    class to make inference for complex ner model

    Arguments:
        model: instance of model class
        tokenizer: tokenizer object
    """

    def __init__(self, model, tokenizer):
        """

        Args:
            model: instance of model class
            tokenizer: tokenizer object
        """

        self.model = model
        self.tokenizer = tokenizer

    def tokenizing_sentences(self, sentence: str, max_length: int):
        """
        method to tokenize input sentence
        Args:
            sentence: input sentence
            max_length: maximum length for sentence

        Returns:
            output of tokenizer (batch with size 1)

        """
        inputs = self.tokenizer.encode_plus(
            text=sentence,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            return_attention_mask=True,
            truncation=True,
            return_tensors="pt"
        )

        return inputs

    def predict(self, inputs):
        """
        method to make prediction with input batch
        Args:
            inputs: input batch

        Returns:

        """
        outputs = self.model(inputs)
        outputs = numpy.argmax(outputs.cpu().detach().numpy(), axis=2)
        return list(outputs)

    def convert_ids_to_entities(self, predicted_tags: List[list]) -> List[list]:
        """
        method to convert each output id to ner tag
        Args:
            predicted_tags: predicted tag for list of samples

        Returns:

        """
        """

        :param predicted_tags: [[ t1, t2, ..., tn][t1, t2, ..., tn]]
        :return:
        """
        outputs = [self.model.hparams["idx2tag"][tag] for tag in predicted_tags[0]]
        # outputs = [[self.model.hparams["idx2tag"][tag] for tag in item]
        #            for item in predicted_tags]
        return outputs

    def convert_token_id_to_token(self, batched_sample: dict) -> List[list]:
        """
        function to convert id to tokens
        Args:
            batched_sample: bacth of samples

        Returns:
            list of sentence

        """

        sentence = [self.tokenizer.convert_ids_to_tokens(idx) for idx in
                    batched_sample["input_ids"]]
        return sentence
