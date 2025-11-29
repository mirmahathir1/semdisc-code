from semdisc.algorithm import precompute_all_join_paths_minhashes
from semdisc.lib import table_embeddings
from semdisc.algorithm import join_graph
from semdisc.algorithm import pairwise_simple_path_index
from semdisc.algorithm import semantic_hash_index
from custom_lib import utils
from semdisc.algorithm import semantic_hash_index
from semdisc.algorithm import diversity_index
from semdisc.algorithm import sanitize
from semdisc.algorithm import semantic_type_extractor
from semdisc.algorithm import online_query_table_processor
from semdisc.algorithm import column_is_numeric
from custom_lib import db
from semdisc.lib import file_path_manager as fpm
from semdisc.stages.datalake_config import get_config

fpm.startup()

datalake = fpm.get_current_datalake_name()

datalake_config = get_config(datalake_name=datalake)

args = utils.parse_arg([
    {'name': 'startstep', 'type': float, 'default': None},
    {'name': 'endstep', 'type': float, 'default': None}
    ])

start_step = args['startstep']
end_step = args['endstep']

if start_step is None or end_step is None:
    utils.crash_code("input a valid step")

def check_routine(routine):
    if end_step < routine:
        exit(0)

    return start_step <= routine

if check_routine(0):
    sanitize.delete_all_processed_data()

if check_routine(1):
    sanitize.copy_datalakes_from_original()

if check_routine(2):
    sanitize.santize_data_function()

all_dataframes = db.load_all_dataframes(fpm.get_uploaded_datasets_path())

if check_routine(3):
    semantic_type_extractor.get_and_save_semantic_types(all_dataframes=all_dataframes)

if check_routine(4):
    all_table_embeddings = table_embeddings.compute_all_table_embeddings(all_dataframes=all_dataframes, normalize=datalake_config['normalize_embeddings'])
    table_embeddings.save_all_table_embeddings(all_table_embeddings)


all_semantic_type_infos=semantic_type_extractor.load_all_semantic_types_single_file()

if check_routine(5):
    hnsw_index = online_query_table_processor.compute_hnsw_index_from_semantic_types(
        all_semantic_type_infos = all_semantic_type_infos
    )
    online_query_table_processor.save_semantic_type_hnsw_index(hnsw_index=hnsw_index)

if check_routine(6):
    minhash_jaccard_similarities = precompute_all_join_paths_minhashes.compute_minhash_jaccard_similarities(all_dataframes=all_dataframes)
    precompute_all_join_paths_minhashes.save_minhash_jaccard_similarities(minhash_jaccard_similarities=minhash_jaccard_similarities)

all_table_embeddings = table_embeddings.load_all_table_embeddings()

if check_routine(7):
    all_simhashes, hyperplanes = semantic_hash_index.compute_simhash_for_tables(
        simhash_size=datalake_config['simhash_size'],
        all_dataframes=all_dataframes,
        all_table_embeddings=all_table_embeddings
    )

    semantic_hash_index.save_all_simhashes(all_simhashes=all_simhashes)
    semantic_hash_index.save_simhash_hyperplanes(simhash_hyperplanes=hyperplanes)

    all_minhash_simhashes = semantic_hash_index.compute_minhash_simhash_for_all_columns(
        all_simhash=all_simhashes, number_of_hashes_for_minhash=datalake_config['number_of_hashes_for_minhash']
        )
    semantic_hash_index.save_minhash_simhashes(minhash_simhashes=all_minhash_simhashes)
    

    minhash_simhash_jaccard_similarities = semantic_hash_index.compute_minhash_simhash_jaccard_for_all_column_pair(
        all_dataframes=all_dataframes,
        all_minhash_simhashes=all_minhash_simhashes
        )
    semantic_hash_index.save_minhash_simhash_jaccard(
        minhash_simhash_jaccard=minhash_simhash_jaccard_similarities
        )
all_simhashes = semantic_hash_index.load_all_simhashes()

if check_routine(8):
    all_diversity = diversity_index.compute_diversity_index(all_dataframes=all_dataframes, all_simhashes=all_simhashes)
    diversity_index.save_all_diversity(all_diversity=all_diversity)

minhash_jaccard_similarities = precompute_all_join_paths_minhashes.load_minhash_jaccard_similarities()
minhash_simhash_jaccard_similarities = semantic_hash_index.load_minhash_simhash_jaccard()
all_diversity = diversity_index.load_all_diversity()

column_name_dictionary = db.get_column_name_dictionary(all_dataframes=all_dataframes)

if check_routine(9):
    semantic_type_similarities = semantic_type_extractor.compute_semantic_type_similarities(all_semantic_types=all_semantic_type_infos)
    semantic_type_extractor.save_semantic_type_similarities(
        semantic_type_similarities=semantic_type_similarities
    )
semantic_type_similarities = semantic_type_extractor.load_semantic_type_similarities()

if check_routine(10):
    is_numeric_of_columns = column_is_numeric.compute_is_numeric_of_columns(all_dataframes=all_dataframes)
    column_is_numeric.save_column_is_numeric(column_is_numeric_dict=is_numeric_of_columns)

is_numeric_of_columns = column_is_numeric.load_column_is_numeric()

if check_routine(11):
    join_graph_dictionary = join_graph.compute_join_graph(
        all_dataframes=all_dataframes,
        minhash_jaccard=minhash_jaccard_similarities,
        minhash_simhash_jaccard=minhash_simhash_jaccard_similarities,
        all_diversity=all_diversity,
        semantic_type_similarities=semantic_type_similarities,
        column_name_dictionary=column_name_dictionary,
        is_numeric_of_columns=is_numeric_of_columns,
        natural_join_enabled=True,
        semantic_join_enabled=True,
        diversity_enabled=datalake_config['diversity_enabled'],
        join_edge_threshold=datalake_config['join_edge_threshold'],
        diversity_multiplier_threshold=datalake_config['diversity_multiplier_threshold'],
        semantic_type_similarity_threshold=datalake_config['semantic_type_similarity_threshold'],
    )

    join_graph.save_join_graph(join_graph_dictionary=join_graph_dictionary)

join_graph_dictionary = join_graph.load_join_graph()

if check_routine(12):
    all_unevaluated_simple_paths = pairwise_simple_path_index.compute_all_unevaluated_pairwise_simple_paths(
        join_graph_dictionary=join_graph_dictionary,
        simple_path_max_length=datalake_config['simple_path_max_length']
        )

    all_join_orders_with_cardinality, table_to_path_index, table_index_map, all_evaluated_simple_paths = pairwise_simple_path_index.compute_cardinality_and_inverted_index(
        all_unevaluated_simple_paths=all_unevaluated_simple_paths,
        join_graph_dictionary=join_graph_dictionary,
        all_dataframes=all_dataframes,
        all_simhashes=all_simhashes
        )

    pairwise_simple_path_index.save_all_join_order(all_join_order=all_join_orders_with_cardinality)
    pairwise_simple_path_index.save_table_index_map(table_index_map=table_index_map)
    pairwise_simple_path_index.save_table_to_path_index(table_to_path_index=table_to_path_index)
    pairwise_simple_path_index.save_all_simple_paths(all_simple_paths=all_evaluated_simple_paths)
    
