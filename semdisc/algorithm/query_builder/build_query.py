from custom_lib import utils
from custom_lib import db
from semdisc.lib import file_path_manager as fpm
from semdisc.algorithm.query_builder import simple_induced_paths
from semdisc.algorithm.query_builder import simple_dfs_path
from semdisc.algorithm import semantic_type_extractor
from semdisc.algorithm import semantic_hash_index
from semdisc.algorithm import pairwise_simple_path_index
from semdisc.algorithm import join_dataframes
from semdisc.lib import constants
from custom_lib import console
from semdisc.lib import table_embeddings
from semdisc.algorithm.query_builder import gt_graph
from semdisc.algorithm import column_is_numeric
from semdisc.algorithm import diversity_index
from tqdm import tqdm
from semdisc.algorithm.query_builder import config_gt_graph
from semdisc.algorithm.query_builder import lib

max_cardinality_for_row_eval = 10_000_000
number_of_vertices = 3

def get_query_table_json_path():
    return fpm.get_datalake_path() + "/query_tables.json"

if __name__ == "__main__":
    simhash_size = 18

    fpm.startup()

    datalake_name = fpm.get_current_datalake_name()

    cosine_similarity_threshold = config_gt_graph.get_config(datalake_name)['cosine_similarity_threshold']
    semantic_type_similarity_threshold = config_gt_graph.get_config(datalake_name)['semantic_similarity_threshold']
    joinability_threshold = config_gt_graph.get_config(datalake_name)['joinability_threshold']
    diversity_threshold = config_gt_graph.get_config(datalake_name)['diversity_threshold']
    induced_path_flag = config_gt_graph.get_config(datalake_name)['induced_path_flag']


    console.log(f"cosine_similarity_threshold: {cosine_similarity_threshold}, \
                semantic_type_similarity_threshold: {semantic_type_similarity_threshold}, \
                    joinability_threshold: {joinability_threshold}, \
                        diversity_threshold: {diversity_threshold}, \
                            induced_path_flag: {induced_path_flag}")


    all_dataframes = db.load_all_dataframes(path=fpm.get_uploaded_datasets_path())
    all_embeddings = table_embeddings.load_all_table_embeddings()
    column_name_dictionary = db.get_column_name_dictionary(all_dataframes=all_dataframes)
    # semantic_type_similarities = semantic_type_extractor.load_semantic_type_similarities()
    # all_diversity = diversity_index.load_all_diversity()
    is_numeric_of_columns = column_is_numeric.load_column_is_numeric()

    all_simhashes, hyperplanes = semantic_hash_index.compute_simhash_for_tables(
        simhash_size=simhash_size,
        all_dataframes=all_dataframes,
        all_table_embeddings=all_embeddings
    )

    # prepared_embeddings = gt_graph.build_prepared_embeddings(
    #     all_dataframes=all_dataframes,
    #     all_embeddings=all_embeddings
    # )

    # joinability_dict = gt_graph.build_joinability_dict(
    #     all_dataframes=all_dataframes,
    #     prepared_embeddings=prepared_embeddings,
    #     is_numeric_of_columns=is_numeric_of_columns,
    #     semantic_type_similarities=semantic_type_similarities,
    #     all_diversity=all_diversity,

    #     cosine_similarity=cosine_similarity_threshold,
    #     semantic_similarity_threshold=semantic_type_similarity_threshold,
    #     diversity_threshold=diversity_threshold
    # )

    # join_graph_gt_dict = gt_graph.build_join_graph_dict(joinability_dict, threshold=joinability_threshold)

    # gt_graph.save_join_graph_gt(join_graph_gt=join_graph_gt_dict)

    join_graph_gt_dict = gt_graph.load_join_graph_gt()

    semantic_type_dictionary = semantic_type_extractor.compute_semantic_type_dictionary(all_dataframes=all_dataframes)
    
    all_queries = lib.generate_queries(
        join_graph_dict=join_graph_gt_dict,
        all_dataframes=all_dataframes,
        all_simhashes=all_simhashes,
        semantic_type_dictionary=semantic_type_dictionary,
        column_name_dictionary=column_name_dictionary,
        is_numeric_of_columns=is_numeric_of_columns,
        max_cardinality_for_row_eval=max_cardinality_for_row_eval,
        number_of_vertices=number_of_vertices
    )

    if len(all_queries) == 0:
        console.log("No queries generated")

    utils.file_dump(all_queries, get_query_table_json_path())