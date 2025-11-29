import numpy as np
from sentence_transformers import SentenceTransformer
from semdisc.lib import table_embeddings
from semdisc.algorithm import semantic_hash_index
from custom_lib import db
from semdisc.algorithm.query_builder import joinability
from semdisc.algorithm import semantic_type_extractor
from semdisc.algorithm import column_is_numeric
from custom_lib import utils, console
from semdisc.lib import file_path_manager as fpm

def compute_is_numeric_of_columns(all_dataframes):
    is_numeric_dictionary = {}
    for table, dataframe in all_dataframes.items():
        is_numeric_dictionary[table] = {}
        for column in dataframe.columns:
            is_numeric_dictionary[table][column] = db.get_numeric_portion_of_column(dataframe, column) > 0.5
    return is_numeric_dictionary

def get_column_is_numeric_json_path():
    return f"{fpm.get_datalake_path()}/column_is_numeric.json"

def load_column_is_numeric():
    return utils.file_load(get_column_is_numeric_json_path())

def save_column_is_numeric(column_is_numeric_dict):
    utils.file_dump(column_is_numeric_dict, get_column_is_numeric_json_path())