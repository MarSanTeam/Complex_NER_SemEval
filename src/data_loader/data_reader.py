# -*- coding: utf-8 -*-
# ========================================================
"""
    Complex NER Project:
        data loader:
            data_reader
"""

# ============================ Third Party libs ============================
import pickle


# ==========================================================================

def read_text(path: str) -> list:
    """
    read_text function for  reading text file
    :param path:
    :return:
    """
    with open(path, "r", encoding="utf8") as file:
        data = file.readlines()
    return data


def read_pickle(path: str) -> list:
    """
    read_pickle function for  reading pickle file
    :param path:
    :return:
    """
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data
