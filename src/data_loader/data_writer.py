# -*- coding: utf-8 -*-
# ==========================================================================

"""
    Complex NER Project:
        data loader:
            writing data
"""

# ============================ Third Party libs ============================
import json
import pickle

import pandas as pd


# ==========================================================================


def write_json(data: dict, path: str) -> None:
    """
    write_json function is written for write in json files
    :param path:
    :param data:
    :return:
    """
    with open(path, "w", encoding="utf8") as outfile:
        json.dump(data, outfile, separators=(",", ":"), indent=4)


def write_pickle(path: str, data: list) -> None:
    """
    write_pickle function is written for write data in pickle file
    :param path:
    :param data:
    :return:
    """
    with open(path, "wb") as outfile:
        pickle.dump(data, outfile)
