# -*- coding: utf-8 -*-
# ========================================================
"""data_writer module is written for write data in files"""
# ========================================================


# ========================================================
# Imports
# ========================================================

import json


def write_json(path: str, data: dict) -> None:
    """
    write_json function is written for write in json files
    :param path:
    :param data:
    :return:
    """
    with open(path, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, separators=(",", ":"), indent=4)
