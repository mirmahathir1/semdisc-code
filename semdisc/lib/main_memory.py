import gc
from custom_lib import db

join_graph = None
def get_join_graph_from_memory():
    return join_graph

def is_join_graph_in_memory():
    return join_graph is not None

def load_join_graph_to_memory(new_join_graph):
    global join_graph
    join_graph = new_join_graph

def delete_join_graph_from_memory():
    global join_graph
    join_graph = None


all_tables_with_joinable_columns = {}
def is_table_loaded_in_memory(table):
    global all_tables_with_joinable_columns
    return table in all_tables_with_joinable_columns

def save_table_to_memory(dataframe, table):
    global all_tables_with_joinable_columns
    all_tables_with_joinable_columns[table] = dataframe

def get_table_from_memory(table):
    return all_tables_with_joinable_columns[table]

def clear_tables_from_memory():
    global all_tables_with_joinable_columns
    all_tables_with_joinable_columns = {}

all_tables = {}
def get_dataframe_from_csv_by_table_name(table):
    global all_tables
    if not table in all_tables:
        all_tables[table] = db.get_dataframe_from_csv_by_table_name(table)
    return all_tables[table]

joined_dataframes = {}
joined_metadatas = {}
def is_joined_dataframe_found_for_join_order_in_memory(join_order_hash):
    return join_order_hash in joined_dataframes
def clear_joined_dataframes_from_memory():
    global joined_dataframes
    global joined_metadatas
    for key in list(joined_dataframes.keys()):  # Create a copy of keys with list()
        del joined_dataframes[key]
        del joined_metadatas[key]

    gc.collect()
    joined_dataframes = {}
    joined_metadatas = {}

def get_joined_dataframe_for_join_order_from_memory(join_order_hash):
    return joined_dataframes[join_order_hash], joined_metadatas[join_order_hash]

def save_joined_dataframe_for_join_order_to_memory(join_order_hash, dataframe, metadata):
    global joined_dataframes
    global joined_metadatas
    joined_dataframes[join_order_hash] = dataframe
    joined_metadatas[join_order_hash] = metadata
