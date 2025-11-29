from semdisc.algorithm.query_builder import simple_induced_paths
from semdisc.algorithm import pairwise_simple_path_index, join_dataframes
from custom_lib import console, utils, db
from tqdm import tqdm
from semdisc.lib import constants

def generate_query_from_simple_path_and_join_order(
    path,
    join_order,

    all_dataframes,
    is_numeric_of_columns,
    column_name_dictionary,
    all_simhashes,
    semantic_type_dictionary,

    number_of_vertices,
    max_cardinality_for_row_eval,
):
    sample_query_columns = []

    sample_query_columns.append({
        "table": path[0],
        "column": utils.select_one_element_randomly_from_list(all_dataframes[path[0]].columns),
    })

    intermediate_query_column_count = utils.sample_list(original_list=list(range(number_of_vertices-1)), sample_spec=1)[0]

    all_columns_of_selected_tables = []

    for table in path:
        for column in column_name_dictionary[table]:
            all_columns_of_selected_tables.append({
                "table": table,
                "column": column
            })

    intermediate_query_columns = utils.ordered_sample(all_columns_of_selected_tables, intermediate_query_column_count)
        
    for table_column_info in intermediate_query_columns:
        sample_query_columns.append({
            "table": table_column_info['table'],
            "column": table_column_info['column'],
        })

    sample_query_columns.append({
        "table": path[-1],
        "column": utils.select_one_element_randomly_from_list(all_dataframes[path[-1]].columns),
    })

    column_mapping_for_join_result = utils.nested_dict()
    for query_column in sample_query_columns:
        is_column_numeric = is_numeric_of_columns[query_column['table']][query_column['column']]
        join_type = constants.NATURAL_JOIN if is_column_numeric else constants.SEMANTIC_HASH_JOIN
        column_mapping_for_join_result[query_column['table']][query_column['column']] = join_type

    tables_ready_for_join = join_dataframes.prepare_tables_with_only_join_columns_with_nan_dropped_for_cardinality_estimation(
        join_order=join_order,
        all_dataframes=all_dataframes,
        all_simhashes=all_simhashes
        )
    
    if not max_cardinality_for_row_eval is None:
        cardinality_info = join_dataframes.calculate_true_join_cardinality(
            tables_ready_for_join=tables_ready_for_join,
            join_order=join_order,
            self_join=False
        )

        if cardinality_info["cardinality"] > max_cardinality_for_row_eval:
            console.log("cardinality of path is greater than max cardinality")
            return None

    join_result = join_dataframes.perform_joins(
        tables_ready_for_join=tables_ready_for_join,
        join_order=join_order,
        sampling_size=1000,
        self_join=False
        )


    index_df = join_result["joined_table"][[f'Index_{query_column["table"]}' for query_column in sample_query_columns]]

    index_df = db.remove_duplicate_columms(index_df)

    joined_projected_table = join_dataframes.build_combined_table(
        index_df=index_df,
        all_dataframes=all_dataframes,
        column_mapping=column_mapping_for_join_result,
        all_simhashes=all_simhashes,
        convert_values_to_simhash=False
        )
    joined_projected_table = joined_projected_table.dropna()
    joined_projected_table = joined_projected_table.drop_duplicates()

    if len(joined_projected_table) == 0:
        console.log("cardinality of path is 0")
        return None

    sample_size = min(len(joined_projected_table), 5)
    joined_projected_table_sample = joined_projected_table.sample(sample_size)

    single_query = []
    for sample_query_column_index, table_column_info in enumerate(sample_query_columns):
        table = table_column_info["table"]
        column = table_column_info["column"]
        single_query.append({
            "semantic_type": semantic_type_dictionary[table][column],
            "examples": joined_projected_table_sample[f"{table}_{column}"].astype(str).tolist(),
            "table": table,
            "column": column,
        })
    return single_query

def generate_queries(
        join_graph_dict,
        
        all_dataframes,
        all_simhashes,
        semantic_type_dictionary,
        column_name_dictionary,
        is_numeric_of_columns,
        max_cardinality_for_row_eval,
        number_of_vertices
        
        
        ):
    all_queries = []

    total_query = 100

    streaming_function = None

    failed_path = 0
    max_failed_path = 5000

    while True:
        streaming_function = simple_induced_paths.random_induced_paths_stream(joinability_dict=join_graph_dict, k=number_of_vertices)
        # break
        if next(streaming_function, None) is None:
            number_of_vertices = number_of_vertices - 1
            if number_of_vertices == 1:
                break
        else:
            streaming_function = simple_induced_paths.random_induced_paths_stream(joinability_dict=join_graph_dict, k=number_of_vertices)
            break

    # if induced_path_flag == False:
    #     streaming_function = simple_dfs_path.random_simple_paths_stream(joinability_dict=join_graph_gt_dict, k=number_of_vertices)

    console.log(f"using streaming function: {streaming_function}")
    console.log(f"number_of_vertices: {number_of_vertices}")

    for i, p in tqdm(enumerate(streaming_function, 1)):
        console.log(f"Path {i}: {p}")

        join_order_gt = pairwise_simple_path_index.get_join_order_from_path_using_best_edges_only(path=p, join_graph_dictionary=join_graph_dict)

        single_query = generate_query_from_simple_path_and_join_order(
            path=p,
            join_order=join_order_gt,
            all_dataframes=all_dataframes,
            is_numeric_of_columns=is_numeric_of_columns,
            all_simhashes=all_simhashes,
            semantic_type_dictionary=semantic_type_dictionary,
            column_name_dictionary=column_name_dictionary,
            max_cardinality_for_row_eval=max_cardinality_for_row_eval,
            number_of_vertices=number_of_vertices
        )

        if single_query is None:
            failed_path += 1
            if failed_path > max_failed_path:
                console.log("too many failed query build attempt. exiting code.")
                break

            continue

        all_queries.append({
            "query": single_query,
            "path": p,
            "join_order": join_order_gt,
            "query_id": f"Query {len(all_queries)}"
        })

        if len(all_queries) >= total_query:
            break

    return all_queries