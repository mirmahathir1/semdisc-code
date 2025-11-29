from custom_lib import utils, console, parallel, text_encoder
from semdisc.lib import table_embeddings, file_path_manager as fpm, tqdm_threaded, constants, table_embeddings
from custom_lib import db
import numpy as np
from datasketch import MinHash
import os
import json
import pickle
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

# number_of_hash = 128
# embedding_size = 768

# Function to compute SimHash using given hyperplanes
def extract_simhash_using_hyperplanes(embeddings, hyperplanes):
    dot_products = np.dot(embeddings, hyperplanes.T)
    simhash_values = (dot_products > 0).astype(int)
    simhash_values = [''.join(map(str, map(int, row))) for row in simhash_values]
    return simhash_values

def extract_angular_lsh_using_hyperplanes(embeddings, hyperplanes):
    dot_products = np.dot(embeddings, hyperplanes.T)
    simhash_values = (dot_products > 0).astype(int)
    simhash_values = [''.join(map(str, map(int, row))) for row in simhash_values]
    return simhash_values

def extract_sign_threshold_strings(embeddings, bit_count):
    simhash_values = (np.array(embeddings) > 0).astype('uint8')
    simhash_values = [''.join(map(str, row))[:bit_count] for row in simhash_values]
    return simhash_values

# Function to compute MinHash signatures using datasketch
def compute_minhash(signature, num_hashes):
    # Create a MinHash object
    m = MinHash(num_perm=num_hashes, seed=1)
    # printbroken(signature.shape)
    # Update MinHash object with the binary SimHash signature
    for i in range(len(signature)):
        # printbroken(string_to_be_hashed)
        if signature[i] is not None:
            m.update(signature[i].encode('utf8'))

    return m

# Function to compute Jaccard similarity between MinHash signatures
def jaccard_similarity(minhash1, minhash2):
    return minhash1.jaccard(minhash2)

def extract_simhash_for_single_column(argument, common_argument, common_argument_for_batch):
    column_value_info = argument
    hyperplanes = common_argument['hyperplanes']
    table = column_value_info['table']
    column = column_value_info['column']
    values = column_value_info['values']
    embeddings = column_value_info['embeddings']
    simhashes = extract_simhash_using_hyperplanes(embeddings=embeddings, hyperplanes=hyperplanes)

    # convert simhash strings to nan where the original value is nan
    # simhashes = np.where(values.isna(), np.nan, simhashes)
    simhashes = np.where(values.isna(), None, simhashes)

    return {
        'table': table,
        'column': column,
        'simhashes': simhashes
    }


def extract_simhash_for_column_list(column_value_info_list, hyperplanes, tracker_submitter):
    for column_value_info in column_value_info_list:
        table = column_value_info['table']
        column = column_value_info['column']

        embeddings = table_embeddings.load_embedding_table_column_from_cache(table, column)
        simhashes = extract_simhash_using_hyperplanes(embeddings=embeddings, hyperplanes=hyperplanes)
        pickle.dump(obj=simhashes, file=open(f"{fpm.get_all_column_simhash_values_directory()}/{table}/{column}.pickle", "wb"))
        tracker_submitter.check_and_submit_tracker_file()

simhash_values_of_columns = defaultdict(dict)
def get_simhash_values_of_column_cached(table, column):
    if column not in simhash_values_of_columns[table]:
        simhash_values_of_columns[table][column] = pickle.load(open(f"{fpm.get_all_column_simhash_values_directory()}/{table}/{column}.pickle", "rb"))
    return simhash_values_of_columns[table][column]

simhash_string_of_columns = defaultdict(dict)
def get_simhash_strings_of_column_cached(table, column):
    if column not in simhash_string_of_columns[table]:
        simhashes = pickle.load(open(f"{fpm.get_all_column_simhash_values_directory()}/{table}/{column}.pickle", "rb"))
        simhash_string_of_columns[table][column] = [''.join(map(str, map(int, row))) for row in simhashes]
    return simhash_string_of_columns[table][column]

def load_simhash_of_all_tables(override_path=None):
    path = fpm.get_all_column_simhash_values_directory()
    if override_path is not None:
        path = override_path
    simhash_string_of_columns = defaultdict(dict)
    column_name_dictionary = db.get_column_name_dictionary()
    for table, columns in column_name_dictionary.items():
        for column in columns:
            simhashes = pickle.load(open(f"{path}/{table}/{column}.pickle", "rb"))
            simhash_string_of_columns[table][column] = [''.join(map(str, map(int, row))) for row in simhashes]
    return simhash_string_of_columns

def compute_simhash_for_tables(simhash_size, all_dataframes, all_table_embeddings):
    console.log(f"extract_simhash_for_all_tables for simhash size {simhash_size} start")

    all_column = []
    for table, dataframe in tqdm(all_dataframes.items()):
        for column in dataframe.columns:
            all_column.append({
                "table": table,
                "column": column,
                "values": dataframe[column],
                "embeddings": all_table_embeddings[table][column]
            })
    all_column = utils.shuffle_list(all_column)
    hyperplanes = get_hyperplanes_for_simhash(simhash_size, text_encoder.get_output_dim())

    results = parallel.execute(func=extract_simhash_for_single_column,
                     argument_list=all_column,
                     common_arguments={'hyperplanes': hyperplanes},
                     function_for_batch=utils.empty_function)
    
    all_simhashes = utils.nested_dict()
    for single_column_simhash in results:
        table = single_column_simhash['table']
        column = single_column_simhash['column']
        simhashes = single_column_simhash['simhashes']
        all_simhashes[table][column] = simhashes

    """
    convert the dictionary into dataframes    
    """
    for table in all_simhashes:
        all_simhashes[table] = pd.DataFrame(all_simhashes[table])

    console.log("extract_simhash_for_all_tables end")
    return all_simhashes, hyperplanes


def get_hyperplanes_for_simhash(num_hyperplanes, embedding_size):
    return np.random.randn(num_hyperplanes, embedding_size)

def get_semantic_hash_similarities_for_single_table_pair(argument, common_argument, common_argument_for_batch):
    table_pair = argument
    numeric_portion_dictionary = common_argument['numeric_portion_dictionary']
    simhash_minhashes = common_argument['simhash_minhashes']
    enable_numeric_columns_for_semantic_graph = common_argument['enable_numeric_columns_for_semantic_graph']
    [table1, table2] = table_pair
    table1_columns = db.get_column_names_from_csv(table1)
    table2_columns = db.get_column_names_from_csv(table2)
    os.makedirs(f"{fpm.get_semantic_hash_similarity_path()}/{table1}", exist_ok = True)
    os.makedirs(f"{fpm.get_semantic_hash_similarity_path()}/{table2}", exist_ok = True)
    table1_to_table2_similarities = []
    table2_to_table1_similarities = []
    for table1_column in table1_columns:
        for table2_column in table2_columns:
            if enable_numeric_columns_for_semantic_graph == False and\
                (numeric_portion_dictionary[table1][table1_column] > 0.5 or numeric_portion_dictionary[table2][table2_column] > 0.5):
                continue
            similarity = simhash_minhashes[table1][table1_column].jaccard(simhash_minhashes[table2][table2_column])
            table1_to_table2_similarities.append({"column1": table1_column, "column2": table2_column, "similarity": similarity})
            table2_to_table1_similarities.append({"column1": table2_column, "column2": table1_column, "similarity": similarity})
    
    table1_to_table2_similarities.sort(key=lambda x: x["similarity"], reverse=True)
    table2_to_table1_similarities.sort(key=lambda x: x["similarity"], reverse=True)
    json.dump(table1_to_table2_similarities, open(f"{fpm.get_semantic_hash_similarity_path()}/{table1}/{table2}.json", "w"), indent=4)
    json.dump(table2_to_table1_similarities, open(f"{fpm.get_semantic_hash_similarity_path()}/{table2}/{table1}.json", "w"), indent=4)

def save_semantic_hash_graph_in_single_file(semantic_hash_graph):
    console.log(f"creating file {fpm.get_semantic_hash_graph_json_path()}")
    json.dump(obj=semantic_hash_graph, fp=open(fpm.get_semantic_hash_graph_json_path(), 'w'), indent=4)

def is_semantic_hash_json_created():
    return os.path.exists(fpm.get_semantic_hash_graph_json_path())

def delete_semantic_hash_json():
    os.unlink(fpm.get_semantic_hash_graph_json_path())

def get_semantic_hash_edge_value(edges, column1, column2):
    matching_edge = list(filter(lambda x: x['column1'] == column1 and x['column2'] == column2, edges))
    if len(matching_edge) == 0:
        return 0
    return matching_edge[0]['similarity']

def get_semantic_matches_by_simhashes(
    column1_values,
    column2_values,
    column1_simhashes,
    column2_simhashes,
):
    """
    Find all pairs of values from two columns where their corresponding simhashes are equal,
    ignoring any pairs where either value is NaN.
    
    Args:
        column1_values: pandas Series of values from the first column
        column2_values: pandas Series of values from the second column
        column1_simhashes: list of strings representing simhashes for column1_values
        column2_simhashes: list of strings representing simhashes for column2_values
    
    Returns:
        tuple: (matches_1_to_2, matches_2_to_1) where each is a list of matches:
              Each match is a list of 4 elements:
              [column1_value, column2_value, column1_index, column2_index]
    """
    matches_1_to_2 = []
    matches_2_to_1 = []
    
    # Create a dictionary to map simhashes to their positions in column2 for faster lookup
    simhash_to_indices = {}
    for idx, simhash in enumerate(column2_simhashes):
        # Skip NaN values in column2
        if pd.isna(column2_values.iloc[idx]):
            continue
        if simhash not in simhash_to_indices:
            simhash_to_indices[simhash] = []
        simhash_to_indices[simhash].append(idx)
    
    # Iterate through column1 and find matching simhashes in column2
    for idx1, (value1, simhash1) in enumerate(zip(column1_values, column1_simhashes)):
        # Skip NaN values in column1
        if pd.isna(value1):
            continue
        if simhash1 in simhash_to_indices:
            for idx2 in simhash_to_indices[simhash1]:
                value2 = column2_values.iloc[idx2]
                # Additional check (though we already skipped column2 NaN values)
                if not pd.isna(value2):
                    matches_1_to_2.append([value1, value2, idx1, idx2])
                    matches_2_to_1.append([value2, value1, idx2, idx1])
    
    return matches_1_to_2, matches_2_to_1


def compute_minhash_simhash_jaccard_for_one_column_pair(argument, common_arguments, common_arguments_for_batch):
    table1_name = argument['table1_name']
    table2_name = argument['table2_name']
    column1_name = argument['column1_name']
    column2_name = argument['column2_name']

    column1_minhash = argument['column1_minhash']
    column2_minhash = argument['column2_minhash']


    jaccard_similarity_minhash = jaccard_similarity(column1_minhash, column2_minhash)
    return {
        'table1_name': table1_name,
        'table2_name': table2_name,
        'column1_name': column1_name,
        'column2_name': column2_name,
        'jaccard_similarity_minhash': jaccard_similarity_minhash
    }

def get_minhash_for_single_column(argument, common_arguments, common_arguments_for_batch):
    table_name = argument['table_name']
    column_name = argument['column_name']

    column_simhashes = argument['column_simhashes']

    number_of_hashes_for_minhash = argument['number_of_hashes_for_minhash']
    minhash_simhash = compute_minhash(column_simhashes, number_of_hashes_for_minhash)

    return {
        'table': table_name,
        'column': column_name,
        'minhash_simhash': minhash_simhash
    }

def compute_minhash_simhash_for_all_columns(all_simhash, number_of_hashes_for_minhash):
    console.log("computing minhashes for all columns")
    results = parallel.execute(
        func=get_minhash_for_single_column,
        argument_list=[
            {
                'table_name': table_name,
                'column_name': column_name,
                'column_simhashes': all_simhash[table_name][column_name],
                'number_of_hashes_for_minhash': number_of_hashes_for_minhash
            }
            for table_name in all_simhash
            for column_name in all_simhash[table_name]
        ]
    )

    all_minhash_simhashes = utils.nested_dict()
    for result in results:
        all_minhash_simhashes[result['table']][result['column']] = result['minhash_simhash']
    return all_minhash_simhashes

def compute_minhash_simhash_jaccard_for_all_column_pair(all_dataframes, all_minhash_simhashes):
    all_column_headers = db.get_all_column_headers(all_dataframes=all_dataframes)
    all_column_pairs = utils.get_combinations(all_column_headers, 2)
    parallel_arguments = []
    for column_pair in tqdm(all_column_pairs):
        table1_name = column_pair[0]['table']
        table2_name = column_pair[1]['table']
        column1_name = column_pair[0]['column']
        column2_name = column_pair[1]['column']
        column1_minhash = all_minhash_simhashes[table1_name][column1_name]
        column2_minhash = all_minhash_simhashes[table2_name][column2_name]
        parallel_arguments.append({
            'table1_name': table1_name,
            'table2_name': table2_name,
            'column1_name': column1_name,
            'column2_name': column2_name,
            'column1_minhash': column1_minhash,
            'column2_minhash': column2_minhash
        })

    results = parallel.execute(
        func=compute_minhash_simhash_jaccard_for_one_column_pair,
        argument_list=parallel_arguments
    )

    all_minhash_simhash_jaccard = utils.nested_dict()

    for result in results:
        table1_name= result['table1_name']
        table2_name= result['table2_name']
        column1_name= result['column1_name']
        column2_name= result['column2_name']
        jaccard_similarity_minhash = result['jaccard_similarity_minhash']

        all_minhash_simhash_jaccard[table1_name][table2_name][column1_name][column2_name] = jaccard_similarity_minhash
        all_minhash_simhash_jaccard[table2_name][table1_name][column2_name][column1_name] = jaccard_similarity_minhash
    
    return all_minhash_simhash_jaccard

def get_all_simhashes_pickle_path():
    return f"{fpm.get_datalake_path()}/all_simhashes.pickle"

def load_all_simhashes():
    return utils.file_load(get_all_simhashes_pickle_path())

def save_all_simhashes(all_simhashes):
    utils.file_dump(all_simhashes, get_all_simhashes_pickle_path())

def get_simhash_hyperplanes_pickle_path():
    return f"{fpm.get_datalake_path()}/simhash_hyperplanes.pickle"

def load_simhash_hyperplanes():
    return utils.file_load(get_simhash_hyperplanes_pickle_path())

def save_simhash_hyperplanes(simhash_hyperplanes):
    utils.file_dump(simhash_hyperplanes, get_simhash_hyperplanes_pickle_path())

def get_minhash_simhash_jaccard_json_path():
    return f"{fpm.get_datalake_path()}/jaccard_minhash_simhash.json"

def save_minhash_simhash_jaccard(minhash_simhash_jaccard):
    utils.file_dump(minhash_simhash_jaccard, get_minhash_simhash_jaccard_json_path())

def load_minhash_simhash_jaccard():
    return utils.file_load(get_minhash_simhash_jaccard_json_path())

def get_minhash_simhashes_pickle_path():
    return f"{fpm.get_datalake_path()}/minhash_simhashes.pickle"

def save_minhash_simhashes(minhash_simhashes):
    utils.file_dump(minhash_simhashes, get_minhash_simhashes_pickle_path())

def load_minhash_simhashes():
    return utils.file_load(get_minhash_simhashes_pickle_path())
