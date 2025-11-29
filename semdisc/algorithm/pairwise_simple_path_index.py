from custom_lib import utils
from custom_lib import console
from custom_lib import parallel

from semdisc.lib import constants
from semdisc.lib import file_path_manager as fpm
from semdisc.lib import tqdm_threaded

from semdisc.algorithm import join_graph
from semdisc.algorithm import join_dataframes

import networkx as nx
import math
import numpy as np
from tqdm import tqdm

import itertools
import os
import json
from networkx.exception import NetworkXNoPath
from collections import defaultdict
import pickle

from semdisc.algorithm import cms_cardinality

def initialize_nx_graph(join_graph_dictionary):
    all_node_pairs = join_graph.get_all_node_pairs(join_graph_dictionary)
    nx_graph_edges = []
    for [node1, node2] in tqdm_threaded.TqdmNonThreaded(all_node_pairs):
        edges = join_graph.get_edges(join_graph_dictionary = join_graph_dictionary, source_table=node1, destination_table=node2)
        if len(edges) == 0:
            continue
        best_edge = edges[0]
        if best_edge[2] > 0:
            best_edge_weight = -math.log2(best_edge[2])
        else:
            best_edge_weight = constants.MAX_VALUE_FOR_EDGE
        nx_graph_edges.append((node1, node2, best_edge_weight))

    G = nx.Graph()
    G.add_weighted_edges_from(nx_graph_edges)
    return G

def get_all_simple_paths_generators(table_pairs, G):
    return [nx.shortest_simple_paths(G, node1, node2) for [node1, node2] in table_pairs]

def get_join_order_from_path_using_best_edges_only(path, join_graph_dictionary):
    join_order = []
    for i in range(1, len(path)):
        parent_table = path[i-1]
        child_table = path[i]
        edges = join_graph.get_edges(
            join_graph_dictionary=join_graph_dictionary,
            source_table=parent_table,
            destination_table=child_table
            )
        best_edge = edges[0]
        join_type = best_edge[3]
        parent_column = best_edge[0]
        child_column = best_edge[1]
        join_order.append({"parent": parent_table,
                           "child": child_table,
                           "parent_column": parent_column,
                           "child_column": child_column,
                           "join_type": join_type})
    return join_order


def get_pairwise_simple_path_tracker_directory():
    return f"{fpm.get_debug_folder_path()}/pairwise_simple_path_tracker"

def get_pairwise_simple_paths_pickle_directory():
    return f"{fpm.get_debug_folder_path()}/pairwise_simple_paths_pickle"

def generate_random_simple_paths_for_one_pair(argument, common_argument, common_argument_for_batch):
    table1 = argument[0]
    table2 = argument[1]
    simple_path_max_length = common_argument['simple_path_max_length']
    path_generator = nx.shortest_simple_paths(common_argument["G"], table1, table2)
    all_generated_paths = []
    while True:
        try:
            next_path = next(path_generator, None)
        except NetworkXNoPath as e:
            break
        if next_path is None:
            break
        if len(next_path) > simple_path_max_length:
            break
        # print(f"{len(all_generated_paths) % 10}", end='')
        if len(all_generated_paths) % 1000 == 0: 
            print(".", end="", flush=True)
        all_generated_paths.append(next_path)
    return all_generated_paths

def get_unevaluated_join_orders_from_join_paths(paths, join_graph_dictionary):
    join_orders = utils.nested_dict()
    for path in tqdm(paths):
        path_hash = utils.hash_dicts(path)
        join_order = get_join_order_from_path_using_best_edges_only(path=path, join_graph_dictionary=join_graph_dictionary)
        join_orders[path_hash]['join_order'] = join_order
        join_orders[path_hash]['path'] = path
    return join_orders

def evaluate_single_simple_path(argument, common_arguments, common_argument_for_batch):
    join_order_hash, join_order_info = argument
    join_order = join_order_info['join_order']
    path = join_order_info['path']
    tables_ready_for_join = join_dataframes.prepare_tables_with_only_join_columns_with_nan_dropped_for_cardinality_estimation(
        join_order=join_order,
        all_dataframes=common_arguments['all_dataframes'],
        all_simhashes=common_arguments['all_simhashes']
    )
    # cardinality = cms_cardinality.calculate_cms_join_cardinality(
    #     tables_ready_for_join=tables_ready_for_join,
    #     join_order=join_order
    # )
    cardinality = join_dataframes.calculate_true_join_cardinality(
        tables_ready_for_join=tables_ready_for_join,
        join_order=join_order
    )
    return {'join_order': join_order, 'path': path, 'hash': join_order_hash, 'cardinality': cardinality['cardinality'], 'time': cardinality['time']}

def save_paths_to_join_order_map(all_join_orders_with_cardinality):
    console.log("processing pairwise_simple_path_to_join_order_map")
    all_paths = []
    path_to_join_order_map = {}

    console.log("building paths from join order")

    for join_order_info in tqdm(all_join_orders_with_cardinality):
        if join_order_info is None:
            continue
        join_order = join_order_info['join_order']
        cardinality = join_order_info['cardinality']
        path = [join_order[0]['parent']]
        path += [order_element['child'] for order_element in join_order]
        path_hash = utils.hash_dicts(path)
        all_paths.append({'path': path, 'cardinality': cardinality, 'hash': path_hash})
        path_to_join_order_map[path_hash] = join_order
    console.log(f"total finalized join path after evaluation: {len(all_paths)}")
    console.log("saving pairwise_simple_path_to_join_order_map to file")
    json.dump(all_paths, open(fpm.get_pairwise_simple_path_all_paths_json(), 'w'), indent=4)
    json.dump(path_to_join_order_map, open(fpm.get_pairwise_simple_path_to_join_order_map(), 'w'), indent=4)

all_pairwise_simple_paths = None
pairwise_simple_path_to_join_order_map = None
def load_paths_to_join_order_map():
    global all_pairwise_simple_paths, pairwise_simple_path_to_join_order_map
    console.log("loading pairwise_simple_path_to_join_order_map to memory")
    all_pairwise_simple_paths = json.load(open(fpm.get_pairwise_simple_path_all_paths_json()))
    pairwise_simple_path_to_join_order_map = json.load(open(fpm.get_pairwise_simple_path_to_join_order_map()))

def reset_all_pairwise_simple_paths():
    global all_pairwise_simple_paths
    all_pairwise_simple_paths = None

def get_all_join_orders(override_path=None):
    path = fpm.get_pairwise_simple_path_to_join_order_map()
    if override_path is not None:
        path = override_path

    pairwise_simple_path_to_join_order_map = utils.json_load(path)
    return pairwise_simple_path_to_join_order_map

def get_join_order_from_simple_path(pairwise_simple_path_to_join_order_map, path):
    path_hash = utils.hash_dicts(path)
    if path_hash not in pairwise_simple_path_to_join_order_map:
        path_hash = utils.hash_dicts(list(reversed(path)))
    if path_hash not in pairwise_simple_path_to_join_order_map:
        utils.crash_code(f"join order not found for path: {path}.")
    return pairwise_simple_path_to_join_order_map[path_hash]


def load_all_pairwise_simple_paths(override_path = None):
    path = fpm.get_pairwise_simple_path_all_paths_json()
    if override_path is not None:
        path = override_path
    
    return utils.json_load(path)

def reset_join_order_from_pairwise_simple_path():
    global all_pairwise_simple_paths
    all_pairwise_simple_paths = None

def trim_path_for_path_index(main_list, sub_list):
    indices = [main_list.index(i) for i in sub_list if i in main_list]
    if indices:
        start, end = min(indices), max(indices)
        return main_list[start:end+1]
    else:
        return []
    
def sort_path_infos(path_infos):
    # put (-) for larger to smaller
    # put (+) for smaller to larger
    return sorted(path_infos, key=lambda x: (-len(x['matched_tables']), len(x['trimmed_path']), -x['contains_exact_contiguous_subset'], -x['cardinality']))

def sort_path_infos_with_length(path_infos):
    return sorted(path_infos, key=lambda x: (len(x)))

def get_best_path_from_tables(initial_tables, all_paths, tqdm_enabled=True):
    path_infos = []

    if tqdm_enabled:
        console.log(f"scanning {len(all_paths)} paths for best match")
        iterator = tqdm_threaded.TqdmNonThreaded(all_paths)
    else:
        iterator = all_paths
    for path in iterator:
        matched_tables = [table for table in path['path'] if table in initial_tables]
        trimmed_path = trim_path_for_path_index(path['path'], matched_tables)
        is_trimmed_path_exact_matched_with_initial_tables = 1 if trimmed_path == initial_tables else 0
        path_infos.append({
            'path': path['path'],
            'cardinality': path['cardinality'],
            'matched_tables': matched_tables, #TODO: deprecate this
            'trimmed_path': trimmed_path,
            "contains_exact_contiguous_subset": is_trimmed_path_exact_matched_with_initial_tables
        })
    if tqdm_enabled:
        console.log("sorting paths")
    return sort_path_infos(path_infos)

def get_best_path_indexes_from_tables_using_table_path_inverted_index(initial_tables, all_paths, all_table_indices, table_to_path_index, tqdm_enabled=True):
    initial_table_indices = [all_table_indices.index(item) for item in initial_tables]
    path_indices_with_initial_tables = np.where(np.all(table_to_path_index[initial_table_indices] == 1, axis=0))[0]
    # paths_with_initial_tables = [all_paths[idx] for idx in path_indices_with_initial_tables]

    # return sort_path_infos_with_length(paths_with_initial_tables)
    return path_indices_with_initial_tables

def gather_pickled_paths_of_node_pairs():
    pickle_root_path = get_pairwise_simple_paths_pickle_directory()
    all_paths = []
    for pickle_path in os.listdir(pickle_root_path):
        all_paths += pickle.load(open(f"{pickle_root_path}/{pickle_path}", 'rb'))
    return all_paths

def compute_all_unevaluated_pairwise_simple_paths(join_graph_dictionary, simple_path_max_length):
    nx_graph = initialize_nx_graph(join_graph_dictionary=join_graph_dictionary)
    nx_graph_node_pairs = list(itertools.combinations(list(nx_graph.nodes), 2))
    nx_graph_node_pairs = utils.shuffle_list(nx_graph_node_pairs)
    console.log(f'max path length: {simple_path_max_length}')
    console.log("generate_random_simple_paths started")
    results = parallel.execute(func=generate_random_simple_paths_for_one_pair,
                     argument_list=nx_graph_node_pairs,
                     common_arguments={'G': nx_graph, 'simple_path_max_length': simple_path_max_length},
                     function_for_batch=utils.empty_function
                     )
    
    all_generated_paths = []
    for result in results:
        all_generated_paths += result

    console.log("build_pairwise_simple_path_index ended")
    return all_generated_paths

def get_all_simple_path_json_path():
    return f"{fpm.get_datalake_path()}/all_simple_paths.json"

def save_all_simple_paths(all_simple_paths):
    utils.file_dump(all_simple_paths, get_all_simple_path_json_path())

def load_all_simple_paths():
    return utils.file_load(get_all_simple_path_json_path())

def compute_cardinality_and_inverted_index(all_unevaluated_simple_paths, join_graph_dictionary, all_dataframes, all_simhashes):
    console.log("---compute_evaluation_of_pairwise_simple_paths")

    console.log(f"total all_unevaluated_simple_paths: {len(all_unevaluated_simple_paths)}")

    console.log(f"getting all unevaluated join orders")
    all_join_orders = get_unevaluated_join_orders_from_join_paths(
        paths=all_unevaluated_simple_paths,
        join_graph_dictionary=join_graph_dictionary
        )

    console.log("evaluating all generated simple paths")

    all_join_orders_with_cardinality = parallel.execute(
        func=evaluate_single_simple_path,
        argument_list=list(all_join_orders.items()),
        common_arguments={
            'all_dataframes': all_dataframes,
            'all_simhashes': all_simhashes
        }
    )

    all_join_orders_indexed_by_hash = {}
    all_evaluated_simple_paths = []

    for join_order_with_cardinality in all_join_orders_with_cardinality:
        if join_order_with_cardinality['cardinality'] == 0:
            continue
        hash_val = join_order_with_cardinality['hash']
        all_join_orders_indexed_by_hash[hash_val] = join_order_with_cardinality
        all_evaluated_simple_paths.append(join_order_with_cardinality['path'])

    all_evaluated_simple_paths.sort(key=lambda x: len(x))

    table_to_path_index, table_index_map = compute_table_to_path_index(
        all_simple_paths=all_evaluated_simple_paths,
        all_dataframes=all_dataframes
    )



    return all_join_orders_indexed_by_hash, table_to_path_index, table_index_map, all_evaluated_simple_paths

def get_table_index_map_json_path():
    return f"{fpm.get_datalake_path()}/table_index_map.json"

def load_table_index_map():
    return utils.file_load(get_table_index_map_json_path())

def save_table_index_map(table_index_map):
    utils.file_dump(table_index_map, get_table_index_map_json_path())

def get_table_to_path_index_pickle_path():
    return f"{fpm.get_datalake_path()}/table_to_path_index.pickle"

def load_table_to_path_index():
    return utils.file_load(get_table_to_path_index_pickle_path())

def save_table_to_path_index(table_to_path_index):
    utils.file_dump(table_to_path_index, get_table_to_path_index_pickle_path())

def get_all_join_order_json_path():
    return f"{fpm.get_datalake_path()}/all_join_orders.json"

def load_all_join_order():
    return utils.file_load(get_all_join_order_json_path())

def save_all_join_order(all_join_order):
    utils.file_dump(all_join_order, get_all_join_order_json_path())

def compute_table_to_path_index(all_simple_paths, all_dataframes):
    table_index_map = list(all_dataframes.keys())
    table_to_path_index = np.zeros((len(table_index_map),len(all_simple_paths)), dtype=int)
    for table_index, table in tqdm(enumerate(table_index_map)):
        for path_index, path in enumerate(all_simple_paths):
            if table in path:
                table_to_path_index[table_index][path_index] = 1

    return table_to_path_index, table_index_map

def evaluate_candidate_path_based_on_row_matches(candidate_path):

    candidate_path_grouped_by_table = defaultdict(list)
    for table_col_info in candidate_path:
        table = table_col_info['table']
        candidate_path_grouped_by_table[table].append(table_col_info)

    indices_validated_in_single_table_per_query_row = {}
    rows_matched_in_single_table = {}
    for table, table_col_info_list in candidate_path_grouped_by_table.items():
        if len(table_col_info_list) == 1:
            continue
        # printbroken("table: "+table)
        all_example_value_indices = [table_column_info["example_match_indices"] for table_column_info in table_col_info_list]
        
        table_row_indices_containing_example_row = {}
        for example_row_index, list_of_example_value_and_index in enumerate(zip(*all_example_value_indices)):
            index_set_list = []
            for single_query_column_example_and_match in list_of_example_value_and_index:
                index_set_list.append(set(single_query_column_example_and_match['indices']))
            

            common_rows = set.intersection(*index_set_list)
            
            table_row_indices_containing_example_row[f"query row {example_row_index}"] = list(common_rows)
            
        
        indices_validated_in_single_table_per_query_row[table] = table_row_indices_containing_example_row
        rows_matched_in_single_table[table] = sum([1 for query_row_name, common_row_indices in table_row_indices_containing_example_row.items() if len(common_row_indices) > 0])

    return rows_matched_in_single_table, indices_validated_in_single_table_per_query_row

def sort_candidate_paths_based_on_row_matches(candidate_path_infos, sort_enabled):
    for candidate_path_info in candidate_path_infos:
        rows_matched_in_single_table, indices_validated_in_single_table_per_query_row = evaluate_candidate_path_based_on_row_matches(candidate_path_info['candidate_join_path'])

        candidate_path_info['matched_rows_all_table'] = sum(rows_matched_in_single_table.values())
        candidate_path_info['matched_rows_per_table'] = rows_matched_in_single_table
        candidate_path_info['indices_validated_in_single_table_per_query_row'] = indices_validated_in_single_table_per_query_row

    if sort_enabled:
        candidate_path_infos.sort(key=lambda x: x['matched_rows_all_table'], reverse = True)

    return candidate_path_infos

# def get_join_orders_from_candidate_paths_using_table_path_inverted_index(
#         candidate_path_infos,
#         all_paths,
#         all_table_indices,
#         table_to_path_index,
#         all_join_orders,
#         K_top_paths,
#         ):


#     best_join_orders_with_detected_columns = []

#     unique_join_path_hashes = []

#     for candidate_path_info in candidate_path_infos:
#         candidate_path = candidate_path_info['candidate_join_path']
#         candidate_path_only_tables = [table_column_info['table'] for table_column_info in candidate_path]

#         if len(unique_join_path_hashes) > K_top_paths:
#             break
#         only_tables_hash = utils.hash_dicts(candidate_path_only_tables)
#         if only_tables_hash in unique_join_path_hashes:
#             continue
#         unique_join_path_hashes.append(only_tables_hash)

#         best_path_indexes = get_best_path_indexes_from_tables_using_table_path_inverted_index(
#             initial_tables=candidate_path_only_tables,
#             all_paths=all_paths,
#             all_table_indices=all_table_indices,
#             table_to_path_index=table_to_path_index,
#         )
#         if len(best_path_indexes) == 0:
#             # console.debug("path not found in inverted index")
#             continue
#         best_path_index = best_path_indexes[0]
#         best_path = all_paths[best_path_index]

#         best_join_order_with_detected_column = utils.deep_copy(get_join_order_from_simple_path(
#                 pairwise_simple_path_to_join_order_map=all_join_orders,
#                 path=best_path
#             ))
        
#         if len(candidate_path) == 1:
#             console.log("found one detected column after interleaving list")
#             console.log(candidate_path)
#             console.log(best_path)
#             utils.crash_code("found one detected column after interleaving list")

#         best_join_order_with_detected_column["detected_columns"] = candidate_path
#         best_join_order_with_detected_column["path"] = best_path
#         best_join_order_with_detected_column["candidate_path_info"] = candidate_path_info

#         best_join_orders_with_detected_columns.append(best_join_order_with_detected_column)

#     return best_join_orders_with_detected_columns

def get_join_orders_from_candidate_paths_using_table_path_inverted_index(
        candidate_path_infos,
        all_paths,
        all_table_indices,
        table_to_path_index,
        all_join_orders,
        K_top_paths,
        ):


    best_join_orders_with_detected_columns = []
    unique_join_order_hashes = []

    for candidate_path_info in tqdm(candidate_path_infos):
        candidate_path = candidate_path_info['candidate_join_path']
        best_path_indexes = get_best_path_indexes_from_tables_using_table_path_inverted_index(
            initial_tables=[table_column_info['table'] for table_column_info in candidate_path],
            all_paths=all_paths,
            all_table_indices=all_table_indices,
            table_to_path_index=table_to_path_index,
        )
        if len(best_path_indexes) == 0:
            # console.debug("path not found in inverted index")
            continue
        best_path_index = best_path_indexes[0]
        best_path = all_paths[best_path_index]

        best_join_order_with_detected_column = utils.deep_copy(get_join_order_from_simple_path(
                pairwise_simple_path_to_join_order_map=all_join_orders,
                path=best_path
            ))
        
        join_order_hash = utils.hash_dicts(best_join_order_with_detected_column)
        if join_order_hash in unique_join_order_hashes:
            # console.debug("path already searched in inverted index")
            continue
        else:
            unique_join_order_hashes.append(join_order_hash)
        
        if len(candidate_path) == 1:
            console.log("found one detected column after interleaving list")
            console.log(candidate_path)
            console.log(best_path)
            utils.crash_code("found one detected column after interleaving list")

        best_join_order_with_detected_column["detected_columns"] = candidate_path
        best_join_order_with_detected_column["path"] = best_path
        best_join_order_with_detected_column["candidate_path_info"] = candidate_path_info

        best_join_orders_with_detected_columns.append(best_join_order_with_detected_column)

        if len(unique_join_order_hashes) >= K_top_paths:
            break

    return best_join_orders_with_detected_columns

def compute_tables_without_paths(inverted_index, table_index_map):
    row_mask   = inverted_index.sum(axis=1) == 0
    zero_rows  = np.nonzero(row_mask)[0] 
    tables_not_having_any_paths = [table_index_map[index] for index in zero_rows]
    return tables_not_having_any_paths