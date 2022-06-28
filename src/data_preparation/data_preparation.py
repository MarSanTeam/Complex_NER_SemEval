# -*- coding: utf-8 -*-
"""
    Complex NER Project:
        data preparation:
            data_prepration.py

"""

# ============================ Third Party libs ============================
from typing import List


def prepare_conll_data(data: list) -> [List[list], List[list]]:
    """
    prepare_conll_data function is written for loading data in conll format
    :param data:
    :return:
    """
    sentences, labels, tokens, tags = [], [], [], []
    for line in data:
        if not line.startswith("# id"):
            if line == "\n":
                if len(tokens) != 0:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens, tags = [], []
            else:
                line = line.strip().split()
                tokens.append(line[0].strip())
                tags.append(line[3].strip())
    return sentences, labels


def tokenize_and_keep_labels(sentences: List[list], labels: List[list], tokenizer,
                             mode: str = "same") -> [List[list], List[list], List[list]]:
    """
    Function to tokenize and preserve labels
    :param sentences: [['تغییرات', 'قیمت', 'رمز', 'ارز',
                            'اتریوم', 'در', 'یک', 'هفته', 'قبل'], ... ]
    :param labels: [['O', 'O', 'B-ENT', 'I-ENT',
                        'I-ENT', 'O','B-TIM', 'I-TIM', 'I-TIM'], ...]
    :param indexed_sentences:
    :param tokenizer:
    :param mode: ["same", "x_mode"]
    :return: [['تغییر', '##ات', 'قیمت', 'رمز', 'ا', '##رز', 'ا', '##تری', '##وم'
                , 'در', 'یک', 'هفت', '##ه', 'قبل'], ... ]
            [['O', 'O', 'O', 'B-ENT', 'I-ENT', 'I-ENT', 'I-ENT',
            'I-ENT', 'I-ENT', 'O', 'B-TIM', 'I-TIM', 'I-TIM', 'I-TIM'], ... ]
    """
    assert len(sentences) == len(labels), "Sentences and labels should have " \
                                          "the same number of samples"
    subtoken_checks = []
    for idx, (sentence, label) in enumerate(zip(sentences, labels)):
        tokenized_sentence, labels_, checks_ = [], [], []
        for word, tag in zip(sentence, label):
            checks_.append("1")
            # Tokenize each word and count number of its subwords
            tokenized_word = tokenizer.tokenize(str(word))
            n_subwords = len(tokenized_word)
            # The tokenized word is added to the resulting tokenized word list
            tokenized_sentence.extend(tokenized_word)
            checks_.extend(["0"] * (n_subwords - 1))

            # The same label is added to the new list of labels `n_subwords` times
            if mode == "same":
                labels_.extend([tag] * n_subwords)
            elif mode == "x_mode":
                labels_.append(tag)
                labels_.extend(["X"] * (n_subwords - 1))
        sentences[idx], labels[idx] = tokenized_sentence, labels_
        subtoken_checks.append(checks_)
    return sentences, labels, subtoken_checks


def add_special_tokens(sentences: List[list], labels: List[list], cls_token: str,
                       sep_token: str) -> [List[list], List[list]]:
    """
    add_special_tokens function is written for add special tokens for samples
    :param sentences:
    :param labels:
    :param cls_token:
    :param sep_token:
    :return:
    """
    for idx, (sentence, label) in enumerate(zip(sentences, labels)):
        sentence.insert(0, cls_token)
        label.insert(0, cls_token)
        sentence.append(sep_token)
        label.append(sep_token)

        sentences[idx] = sentence
        labels[idx] = label
    return sentences, labels


def pad_sequence(texts: List[list], max_length: int, pad_item: str = "[PAD]") -> List[list]:
    """
    pad_sequence function is written for pad list of samples
    :param texts: [["item_1", "item_2", "item_3"], ["item_1", "item_2"]]
    :param max_length: 4
    :param pad_item: pad_item
    :return: [["item_1", "item_2", "item_3", pad_item],
                    ["item_1", "item_2", pad_item, pad_item]]
    """
    for idx, text in enumerate(texts):
        text_length = len(text)
        texts[idx].extend([pad_item] * (max_length - text_length))
    return texts


def truncate_sequence(texts: List[list], max_length: int) -> list:
    """
    truncate_sequence function is written for truncate list of samples
    :param texts: [["item_1", "item_2", "item_3"], ["item_1", "item_2"]]
    :param max_length: 2
    :return: [["item_1", "item_2"], ["item_1", "item_2"]]
    """
    for idx, text in enumerate(texts):
        if len(text) > max_length:
            texts[idx] = text[: max_length - 1]
            texts[idx].append("[SEP]")
    return texts


def create_test_samples(data: List[list], tokenizer) -> [List[list], List[list], List[list]]:
    """

    :param data:
    :param tokenizer:
    :return:
    """
    subtoken_checks = []
    for idx, item in enumerate(data):
        tokenized_item = []
        subtoken_checks_temp = []
        for tok in item:
            subtoken_checks_temp.append("1")
            tokenized_word = tokenizer.tokenize(tok)
            n_subwords = len(tokenized_word)
            tokenized_item.extend(tokenized_word)
            subtoken_checks_temp.extend(["0"] * (n_subwords - 1))

        data[idx] = tokenized_item
        subtoken_checks.append(subtoken_checks_temp)
    return data, subtoken_checks
