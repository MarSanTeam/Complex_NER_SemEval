# -*- coding: utf-8 -*-
# ========================================================

"""
    Complex NER Project:
        configuration:
                Get data and parameter from the user
"""

# ============================ Third Party libs ============================
import argparse
from pathlib import Path


# ========================================================


class BaseConfig:
    """
    BaseConfig class is written to write configs in it
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--model_name", type=str, default="english")

        self.parser.add_argument("--processed_data_dir", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/"
                                                                               "public_data/")

        self.parser.add_argument("--assets_dir", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/assets/")

        self.parser.add_argument("--saved_model_path", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/assets/saved_models/")

        self.parser.add_argument("--language_model_path", type=str,
                                 default=Path(__file__).parents[3].__str__()
                                         + "/LanguageModels/t5_en_large")
        self.parser.add_argument("--language_model_tokenizer_path", type=str,
                                 default=Path(__file__).parents[3].__str__()
                                         + "/LanguageModels/t5_en_large")

        self.parser.add_argument("--csv_logger_path", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/assets")

        self.parser.add_argument("--train_data", type=str, default="EN-English/en_train.conll")
        self.parser.add_argument("--test_data", type=str, default="test_data.csv")
        self.parser.add_argument("--dev_data", type=str, default="EN-English/en_dev.conll")

        self.parser.add_argument("--save_top_k", type=int, default=1, help="...")

        self.parser.add_argument("--num_workers", type=int,
                                 default=10,
                                 help="...")

        self.parser.add_argument("--n_epochs", type=int,
                                 default=1,
                                 help="...")

        self.parser.add_argument("--batch_size", type=int,
                                 default=32,
                                 help="...")

        self.parser.add_argument("--dropout", type=float,
                                 default=0.15,
                                 help="...")

        self.parser.add_argument("--lr", default=2e-5,
                                 help="...")

    def get_config(self):
        """

        :return:
        """
        return self.parser.parse_args()
