from custom_lib import db
from semdisc.algorithm import semantic_hash_index
from semdisc.algorithm import semantic_type_extractor
from semdisc.algorithm import online_query_table_processor
from semdisc.algorithm import pairwise_simple_path_index
from custom_lib import console
from custom_lib import utils
from semdisc.algorithm.query_builder import build_query
from semdisc.algorithm import join_dataframes
from tqdm import tqdm
from collections import defaultdict
from semdisc.algorithm.query_builder.build_query import max_cardinality_for_row_eval
from semdisc.algorithm import column_is_numeric
from semdisc.lib import file_path_manager as fpm

fpm.startup()

K_top_paths = 5
K_top_semantic_types = 10
sort_by_rows = True


console.log(f"K_top_paths: {K_top_paths}, K_top_semantic_types: {K_top_semantic_types}, sort_by_rows: {sort_by_rows}, max_cardinality_for_row_eval: {max_cardinality_for_row_eval}")

all_queries = utils.file_load(build_query.get_query_table_json_path())

console.log("Loading all data")
all_dataframes = db.load_all_dataframes(fpm.get_uploaded_datasets_path())

all_simhashes = semantic_hash_index.load_all_simhashes()
all_semantic_type_infos=semantic_type_extractor.load_all_semantic_types_single_file()
all_paths = pairwise_simple_path_index.load_all_simple_paths()
table_index_map = pairwise_simple_path_index.load_table_index_map()
table_to_path_index = pairwise_simple_path_index.load_table_to_path_index()
all_join_orders = pairwise_simple_path_index.load_all_join_order()
is_numeric_of_columns = column_is_numeric.load_column_is_numeric()
hnsw_index = online_query_table_processor.load_semantic_type_hnsw_index()
simhash_hyperplanes = semantic_hash_index.load_simhash_hyperplanes()
console.log("finished loading all data")

column_value_indices, column_simhash_indices = online_query_table_processor.compute_column_value_and_simhash_indices_for_all_column(
    all_dataframes=all_dataframes,
    all_simhashes=all_simhashes
)

found_all_examples = 0
found_one_example = 0
no_join_path_found = 0

found_query_count = 0

found_all_examples_in_a_table = 0
found_one_example_in_a_table = 0

# debug_query_id = 'Query 44'
debug_query_id = None

for query_number, query_info in tqdm(enumerate(list(all_queries))):
    user_query = query_info['query']

    example_row_count = len(user_query[0]['examples'])

    query_id = query_info['query_id']

    if not debug_query_id is None and not query_id == debug_query_id:
        continue

    console.log(f"\nQuery ID: {query_id}")

    tables_without_path = pairwise_simple_path_index.compute_tables_without_paths(inverted_index=table_to_path_index, table_index_map=table_index_map)


    start_time = utils.get_current_time()
    user_query_with_detected_columns = online_query_table_processor.get_most_relevant_columns_based_on_search_index_and_example_value_match_count(
        user_query=user_query,
        K_semantic_type=K_top_semantic_types,
        search_index=hnsw_index,
        all_semantic_type_infos=all_semantic_type_infos,
        column_value_indices=column_value_indices,
        column_simhash_indices=column_simhash_indices,
        simhash_hyperplanes=simhash_hyperplanes,
        is_column_numeric=is_numeric_of_columns,
        tables_without_path=tables_without_path
    )
    


    console.log(f"get_most_relevant_columns_based_on_search_index_and_example_value_match_count took {utils.get_current_time()-start_time}")
    # utils.file_dump(console.reduce_data(data=utils.deep_copy(user_query_with_detected_columns), depth=3), f"debug_folder/user_query_with_detected_columns.json")
    # console.log("user_query_with_detected_columns")
    # console.log(user_query_with_detected_columns, depth=3)

    # utils.file_dump(console.reduce_data(data=utils.deep_copy(user_query_with_detected_columns), depth=3), f"user_query_with_detected_columns.json")
    start_time = utils.get_current_time()
    candidate_path_infos = online_query_table_processor.get_candidate_paths(user_query_with_detected_columns)
    console.log(f"get_candidate_paths took {utils.get_current_time()-start_time}")

    if len(candidate_path_infos) == 0:
        console.log("WARNING: At least one query table column had zero candidate columns by semdisc")

    # utils.file_dump(console.reduce_data(data=utils.deep_copy(candidate_path_infos), depth=3), f"debug_folder/candidate_path_infos.json")

    # utils.file_dump(candidate_path_infos, f"debug_folder/query_id__{query_id}.json")
    # exit(0)
    start_time = utils.get_current_time()
    candidate_path_infos = pairwise_simple_path_index.sort_candidate_paths_based_on_row_matches(
        candidate_path_infos=candidate_path_infos,
        sort_enabled=sort_by_rows
        )
    console.log(f"sort_candidate_paths_based_on_row_matches took {utils.get_current_time()-start_time}")

    for candidate_path_info in candidate_path_infos:
        if sum(candidate_path_info['matched_rows_per_table'].values()) == 5:
            found_all_examples_in_a_table += 1
            break

    for candidate_path_info in candidate_path_infos:
        if sum(candidate_path_info['matched_rows_per_table'].values()) > 0:
            found_one_example_in_a_table += 1
            break

    # console.log("candidate_path_infos")
    # utils.file_dump(console.reduce_data(data=utils.deep_copy(candidate_path_infos), depth=3), f"candidate_path_infos.json")
    start_time = utils.get_current_time()
    best_join_orders = pairwise_simple_path_index.get_join_orders_from_candidate_paths_using_table_path_inverted_index(
        candidate_path_infos=candidate_path_infos,
        all_paths=all_paths,
        all_table_indices=table_index_map,
        table_to_path_index=table_to_path_index,
        all_join_orders=all_join_orders,
        K_top_paths=K_top_paths,
    )
    console.log(f"get_join_orders_from_candidate_paths_using_table_path_inverted_index took {utils.get_current_time()-start_time}")

    if len(best_join_orders) == 0:
        no_join_path_found += 1

    enriched_best_join_orders = []
    for old_join_order_info in best_join_orders:
        try:
            join_order_info = utils.deep_copy(old_join_order_info)
        except Exception as e:
            console.log(old_join_order_info)
            exit(0)
        join_order = join_order_info['join_order']
        console.log(f'evaluating join order {utils.hash_dicts(join_order)}')

        detected_columns = join_order_info['detected_columns']

        tables_ready_for_join = join_dataframes.prepare_tables_with_only_join_columns_with_nan_dropped_for_cardinality_estimation(
            join_order=join_order,
            all_dataframes=all_dataframes,
            all_simhashes=all_simhashes
        )

        indices_validated_in_single_table_per_query_row = join_order_info["candidate_path_info"]["indices_validated_in_single_table_per_query_row"]

        # console.log(indices_validated_in_single_table_per_query_row)

        table_to_example_match_index_map, table_to_example_match_index_map_per_query_row = online_query_table_processor.get_table_row_indices_from_query_example(
            detected_columns=join_order_info['detected_columns'],
            indices_validated_in_single_table_per_query_row = indices_validated_in_single_table_per_query_row
        )

        table_to_query_column_map = defaultdict(list)
        for detected_column in detected_columns:
            table = detected_column['table']
            column = detected_column['column']
            table_to_query_column_map[table].append(column)

        # console.log(table_to_example_match_index_map['act_table_full'])

        tables_ready_for_join = online_query_table_processor.trim_tables_ready_for_join_by_row_indices(
            tables_ready_for_join=tables_ready_for_join,
            table_to_example_match_index_map=table_to_example_match_index_map
        )

        cardinality_info = join_dataframes.calculate_true_join_cardinality(
            tables_ready_for_join=tables_ready_for_join,
            join_order=join_order
        )

        if cardinality_info['cardinality'] == 0:
            console.log("joined table has zero rows")
            continue
        elif cardinality_info['cardinality'] > max_cardinality_for_row_eval:
            console.log(f"joined table has more than {max_cardinality_for_row_eval} rows. skipping row check")
            continue

        join_result = join_dataframes.perform_joins(
            tables_ready_for_join=tables_ready_for_join,
            join_order=join_order,
        )


        joined_table = join_result['joined_table']

        console.log("joined table shape: ", joined_table.shape, pretty=False)

        join_order_info["whole_example_row_find_count"] = online_query_table_processor.get_whole_query_row_find_count_by_equi_match(
            joined_table=joined_table,
            detected_columns=join_order_info['detected_columns'],
            all_dataframes=all_dataframes,
            all_simhashes=all_simhashes,
            is_numeric_of_columns=is_numeric_of_columns,
            simhash_hyperplanes=simhash_hyperplanes,
        )

        enriched_best_join_orders.append(join_order_info)
        if join_order_info["whole_example_row_find_count"] == 5:
            break

    count_of_query_rows_found_in_join_order = [join_order_info["whole_example_row_find_count"] for join_order_info in enriched_best_join_orders]
    if len(enriched_best_join_orders) == 0:
        console.log(f"No join paths found for query")
    else:
        found_all_examples += 1 if max(count_of_query_rows_found_in_join_order) == example_row_count else 0
    found_one_example += 1 if sum(count_of_query_rows_found_in_join_order) > 0 else 0
    console.log(f"Precision@{K_top_paths} for finding all examples: {found_all_examples/(query_number+1)}")
    console.log("_"*100)

