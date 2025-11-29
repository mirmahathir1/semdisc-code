from custom_lib import utils
from custom_lib import parallel
from custom_lib import console

from custom_lib import db
from semdisc.lib import file_path_manager as fpm
from semdisc.lib import tqdm_threaded

import json
from datasketch import MinHash
import pandas as pd
import os

def get_minhash_edge_value(edges, column1, column2):
    matching_edge = list(filter(lambda x: x[1] == column1 and x[2] == column2, edges))
    if len(matching_edge) == 0:
        return 0
    return matching_edge[0][0]

def materialize_minhash_graph(column_name_dictionary):
    if len(os.listdir(fpm.get_minhashes_folder())) == 0:
        console.log("ERROR: minhash graph not materialized yet. Program will exit")
        os._exit(1)
    if is_minhash_json_created():
        console.log("minhash json loading from cache")
        return load_minhash_graph_from_json()
    minhash_graph = {}
    console.log("materializing minhash graph")
    combinations = db.get_all_pairs_of_tables()
    for [table1, table2] in tqdm_threaded.TqdmNonThreaded(combinations):
        if table1 not in minhash_graph:
            minhash_graph[table1] = {}
        if table2 not in minhash_graph:
            minhash_graph[table2] = {}
        if table1 not in minhash_graph[table2]:
            minhash_graph[table2][table1] = {}
        if table2 not in minhash_graph[table1]:
            minhash_graph[table1][table2] = {}

        if not os.path.exists(f"{fpm.get_minhashes_folder()}/{table1}/{table2}.json"):
            continue
        edges = json.load(open(f"{fpm.get_minhashes_folder()}/{table1}/{table2}.json",'r'))
        table1_columns = column_name_dictionary[table1]
        table2_columns = column_name_dictionary[table2]
        all_possible_pairs = [(i,j) for i in table1_columns for j in table2_columns]
        for (column1, column2) in all_possible_pairs:
            if column1 not in minhash_graph[table1][table2]:
                minhash_graph[table1][table2][column1] = {}
            if column2 not in minhash_graph[table2][table1]:
                minhash_graph[table2][table1][column2] = {}

            minhash_value = get_minhash_edge_value(edges, column1, column2)
            minhash_graph[table1][table2][column1][column2] = minhash_value
            minhash_graph[table2][table1][column2][column1] = minhash_value
    save_minhash_graph_in_single_file(minhash_graph)
    return minhash_graph

def save_minhash_graph_in_single_file(minhash_graph):
    console.log(f"creating file {fpm.get_minhash_file_path()}")
    json.dump(obj=minhash_graph, fp=open(fpm.get_minhash_file_path(), 'w'), indent=4)

def is_minhash_json_created():
    return os.path.exists(fpm.get_minhash_file_path())

def delete_minhash_json():
    os.unlink(fpm.get_minhash_file_path())

def load_minhash_graph_from_json(override_path=None):
    path = fpm.get_minhash_file_path()
    if override_path is not None:
        path = override_path
    return json.load(fp=open(path))

def column_min_hashes(argument, common_argument, common_argument_for_batch):
    table, dataframe = argument

    col_min_hash = []
    columns = []
    for column in dataframe.columns:
        col_min_hash += [(column,MinHash())]
        columns += ["\"" + column + "\""]

    for row_index, row in dataframe.iterrows():
        for index in range(len(row)):
            if pd.notna(row[index]):
                col_min_hash[index][1].update(str(row[index]).encode('utf8'))
    column_minhashes = {}
    for single_col_hash in col_min_hash:
        column_name = single_col_hash[0]
        minhash = single_col_hash[1]
        column_minhashes[column_name] = minhash
    return {table: column_minhashes}

def compute_jaccard_similarity_for_single_column_pair(argument, common_argument, common_argument_for_batch):
    table1 = argument['table1']
    table2 = argument['table2']
    column1 = argument['column1']
    column2 = argument['column2']

    column1_minhash = argument['column1_minhash']
    column2_minhash = argument['column2_minhash']

    jaccard = column1_minhash.jaccard(column2_minhash)
                
    return {
        'table1': table1,
        'table2': table2,
        'column1': column1,
        'column2': column2,
        'jaccard': jaccard
    }

def compute_all_minhashes_of_columns_of_all_tables(all_dataframes):
    results = parallel.execute(
        func=column_min_hashes,
        argument_list=list(all_dataframes.items()),
        common_arguments={},
        function_for_batch=utils.empty_function
        )

    table_minhashes_cache = {}
    for result in results:
        table_minhashes_cache.update(result)

    return table_minhashes_cache

def get_all_minhash_jaccard(all_dataframes, table_minhashes_cache):
    console.log("Calculating jaccard similarities:")

    column_headers = db.get_all_column_headers(all_dataframes=all_dataframes)

    all_pair_column_header = utils.get_combinations(column_headers, 2)

    parallel_data = []
    for (column_1_info, column_2_info) in all_pair_column_header:
        parallel_data.append({
            'table1': column_1_info['table'],
            'table2': column_2_info['table'],
            'column1': column_1_info['column'],
            'column2': column_2_info['column'],
            'column1_minhash': table_minhashes_cache[column_1_info['table']][column_1_info['column']],
            'column2_minhash': table_minhashes_cache[column_2_info['table']][column_2_info['column']]
        })

    results = parallel.execute(
        func=compute_jaccard_similarity_for_single_column_pair,
        argument_list=parallel_data
        )
    
    all_minhash_jaccard_similarities = utils.nested_dict()
    for result in results:
        table1 = result['table1']
        table2 = result['table2']
        column1 = result['column1']
        column2 = result['column2']
        jaccard = result['jaccard']
        all_minhash_jaccard_similarities[table1][table2][column1][column2] = jaccard
        all_minhash_jaccard_similarities[table2][table1][column2][column1] = jaccard

    console.log(f"completed calculating jaccard similarities")
    return all_minhash_jaccard_similarities

def get_minhash_jaccard_similarities_path(path):
    if path is not None:
        return path
    return f"{fpm.get_datalake_path()}/jaccard_minhash.json"

def save_minhash_jaccard_similarities(minhash_jaccard_similarities, path=None):
    utils.file_dump(minhash_jaccard_similarities, get_minhash_jaccard_similarities_path(path))

def load_minhash_jaccard_similarities(path = None):
    return utils.file_load(get_minhash_jaccard_similarities_path(path))    

def compute_minhash_jaccard_similarities(all_dataframes):
    console.log("---compute minhash graph: started")
    table_minhashes_cache = compute_all_minhashes_of_columns_of_all_tables(all_dataframes=all_dataframes)
    console.log("Minhash calculation complete")

    minhash_jaccard_similarities = get_all_minhash_jaccard(all_dataframes=all_dataframes, table_minhashes_cache=table_minhashes_cache)

    console.log("compute minhash graph: complete")

    return minhash_jaccard_similarities