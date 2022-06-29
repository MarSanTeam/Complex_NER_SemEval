# -*- coding: utf-8 -*-
"""
This module is written to write data preparation function
"""

# ============================ Third Party libs ============================
from typing import List


def prepare_conll_data(data: list) -> [List[list], List[list]]:
    """
    prepare_conll_data function is written for loading data in conll format

    Args:
        data: named entity recognition data format

    Returns:
        list of tokenized sentences, list of tokens tags for eact sentence

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
    function to handle subtoken labels and add subtoken check feature
    Args:
        sentences: list of input tokenized sentence
        labels: list of ner tag for each input tokenized sentence
        tokenizer: tokenizer objective
        mode: if same, ner tag for all subtoken are equal.
              if x_mode, ner tag for all subtoken except first one is X

    Returns:sentences, labels, subtoken_checks
        list of tokenized sentences
        list of ner tag for each token in sentence
        list of subtoken check tag for each token in sentence

    Examples:
        sentences = [["تغییرات", "قیمت", "رمز", "ارز", "اتریوم", "در", "یک", "هفته", "قبل"]]
        labels = [["O", "O", "B-ENT", "I-ENT", "I-ENT", "O","B-TIM", "I-TIM", "I-TIM"]]
        [['تغییر', '##ات', 'قیمت', 'رمز', 'ا', '##رز', 'ا', '##تری', '##وم'
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
    function to add special tokens for samples

    Args:
        sentences: list of input tokenized sentences
        labels: list of ner tag for each input tokenized sentences
        cls_token: cls token
        sep_token: sep token

    Returns:
        list of tokenized sentences with special tokens
        list of ner tag for each tokenized sentences with special tokens

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
    function to padded input list to maximum length

    Args:
        texts: list of tokenized sentences
        max_length: maximum length for each sentence
        pad_item: pad item

    Returns:
        list of padded sentences

    """

    for idx, text in enumerate(texts):
        text_length = len(text)
        texts[idx].extend([pad_item] * (max_length - text_length))
    return texts


def truncate_sequence(texts: List[list], max_length: int) -> list:
    """
    function to truncate input sentences
    Args:
        texts: list of input tokenized sentences
        max_length: maximum length of sentence

    Returns:
        list of truncated sentences

    """

    for idx, text in enumerate(texts):
        if len(text) > max_length:
            texts[idx] = text[: max_length - 1]
            texts[idx].append("[SEP]")
    return texts


def create_test_samples(data: List[list], tokenizer) -> [List[list], List[list]]:
    """
    function to make test sample to inference
    Args:
        data: list of input sentence
        tokenizer: tokenizer object

    Returns:
        list of tokenized sentences
        list of subtoken check for each sentence

    """
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
