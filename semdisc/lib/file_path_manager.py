import json
import argparse
from custom_lib import console

datalake_name = None

semdisc_data_directory = './data'

all_datalakes_directory = f"{semdisc_data_directory}/all_datalakes"

def get_all_datalakes_directory():
    return all_datalakes_directory

def is_current_datalake_name_set():
    return not datalake_name is None

def set_current_datalake_name(datalake):
    global datalake_name
    datalake_name = datalake

def parse_datalake_name_from_commandline_argument():
    parser = argparse.ArgumentParser(description="Process the --datalake argument.")
    parser.add_argument('--datalake', required=True, help='Path to the datalake')
    args, unknown = parser.parse_known_args()
    return args.datalake

def get_current_datalake_name():
    return datalake_name

current_datalake_name_json_path = 'active_datalake.json'
def write_current_datalake_name_to_file(datalake):
    json.dump({"active_datalake": datalake}, fp=open(current_datalake_name_json_path,'w'))

def read_current_datalake_name_from_file():
    return json.load(open(current_datalake_name_json_path,'r'))["active_datalake"]

def get_datalake_path():
    return f'{all_datalakes_directory}/{datalake_name}'

def get_join_graph_root_directory():
    return f'{all_datalakes_directory}/{datalake_name}/join_graph'

def get_join_graph_json_file():
    return f'{all_datalakes_directory}/{datalake_name}/join_graph.json'

def get_joined_table_cache_root():
    return f'{all_datalakes_directory}/{datalake_name}/joined_table_cache'

def get_joined_table_log_root():
    return f'{all_datalakes_directory}/{datalake_name}/joined_table_cache/logs'

def get_joined_table_fail_log_root():
    return f'{all_datalakes_directory}/{datalake_name}/joined_table_cache/fail_logs'

def get_annotation_root_folder():
    return f'{all_datalakes_directory}/{datalake_name}/annotations'

def get_semantic_relation_pair_path():
    return f'{all_datalakes_directory}/{datalake_name}/llm_semantic_relation_pairs'

def get_path_to_join_order_mapping_folder():
    return f'{all_datalakes_directory}/{datalake_name}/path_to_join_order_mapping'

def get_minhashes_folder():
    return f'{all_datalakes_directory}/{datalake_name}/minhashes'

def get_uploaded_datasets_path():
    return f'{all_datalakes_directory}/{datalake_name}/uploaded_datasets'

def get_original_csv_path():
    return f'{all_datalakes_directory}/{datalake_name}/original'

def get_damaged_datasets_path():
    return f'{all_datalakes_directory}/{datalake_name}/damaged_datasets'

def get_semantic_types_path():
    return f'{all_datalakes_directory}/{datalake_name}/semantic_types'

def get_pairwise_column_embedding_similarities_path():
    return f'{all_datalakes_directory}/{datalake_name}/pairwise_column_embedding_similarities'

def get_embedding_graph_json_path():
    return f'{all_datalakes_directory}/{datalake_name}/graph_embedding.json'

def get_sample_embedding_directory():
    return f'{all_datalakes_directory}/{datalake_name}/sample_embedding_directory'

def get_minimal_llm_graph_path():
    return f"{all_datalakes_directory}/{datalake_name}/semantic_graph.json"

def get_llm_graph_json_path():
    return f'{all_datalakes_directory}/{datalake_name}/graph_llm.json'

def get_table_embeddings_path():
    return f'{all_datalakes_directory}/{datalake_name}/table_embeddings'

def get_table_embeddings_without_nan_path():
    return f'{all_datalakes_directory}/{datalake_name}/table_embeddings_without_nan'

def get_pairwise_simple_path_join_order_collection_path():
    return f'{all_datalakes_directory}/{datalake_name}/pairwise_simple_path_join_order_collection'

def get_pairwise_simple_path_all_paths_json():
    return f'{all_datalakes_directory}/{datalake_name}/pairwise_simple_path_all_paths.json'

def get_pairwise_simple_path_to_join_order_map():
    return f'{all_datalakes_directory}/{datalake_name}/pairwise_simple_path_to_join_order_map.json'

def get_debug_folder_path():
    return f'{all_datalakes_directory}/{datalake_name}/debug_folder'

def get_semantic_hash_similarity_path():
    return f'{all_datalakes_directory}/{datalake_name}/semantic_hash_similarities'

def get_semantic_hash_path():
    return f'{all_datalakes_directory}/{datalake_name}/semantic_hash.pickle'

def get_semantic_hash_graph_json_path():
    return f'{all_datalakes_directory}/{datalake_name}/graph_semantic_hash.json'

def get_semantic_hash_hyperplanes_path():
    return f'{all_datalakes_directory}/{datalake_name}/semantic_hash_hyperplanes'

def get_semantic_matches_directory():
    return f"{all_datalakes_directory}/{datalake_name}/semantic_join_matches"

def get_semantic_type_simililarity_pickle_path():
    return f"{all_datalakes_directory}/{datalake_name}/semantic_type_simililarity.pickle"

def get_all_column_simhash_values_directory():
    return f"{all_datalakes_directory}/{datalake_name}/all_column_simhash_values"

def get_diversity_index_file_path():
    return f"{all_datalakes_directory}/{datalake_name}/diversity_index.json"

# locations which are not related to datalakes
def get_dump_directory():
    return f"{semdisc_data_directory}/dump"

def startup(overwrite_datalake_name=None, verbose=True):
    console.log("code is not running under flask")

    if overwrite_datalake_name is None:
        current_datalake = parse_datalake_name_from_commandline_argument()
    else:
        current_datalake = overwrite_datalake_name
    
    set_current_datalake_name(current_datalake)
    if verbose:
        console.log("_"*40)
        console.log(f"active datalake: {current_datalake}")
    console.log("startup done")