from custom_lib import llm_interface
from semdisc.algorithm import semantic_type_extractor
from custom_lib import utils
from custom_lib import console
from custom_lib import db
from custom_lib import text_encoder
from semdisc.lib import table_embeddings
from semdisc.algorithm import pairwise_simple_path_index
from custom_lib import parallel
from semdisc.algorithm import semantic_hash_index
from collections import defaultdict
from tqdm import tqdm
from semdisc.algorithm import join_dataframes
from semdisc.lib import constants
from semdisc.lib import file_path_manager as fpm

def trim_tables_ready_for_join_by_row_indices(tables_ready_for_join, table_to_example_match_index_map):
    new_tables_ready_for_join = {}

    for table, dataframe in tables_ready_for_join.items():
        new_dataframe = dataframe
        if table in table_to_example_match_index_map:
            index_column = f"Index_{table}"
            indexes_to_keep = [index for index in table_to_example_match_index_map[table]]
            new_dataframe = dataframe[dataframe[index_column].isin(indexes_to_keep)]

        new_tables_ready_for_join[table] = new_dataframe.copy().dropna().drop_duplicates()

    return new_tables_ready_for_join

def get_table_row_indices_from_query_example(detected_columns, indices_validated_in_single_table_per_query_row):
    table_to_example_match_index_map = defaultdict(set)
    table_to_example_match_index_map_per_query_row = defaultdict(dict)
    for detected_column in detected_columns:
        table = detected_column['table']
        if table in table_to_example_match_index_map:
            continue
        if table in indices_validated_in_single_table_per_query_row:
            table_to_example_match_index_map[table] = set().union(*list(indices_validated_in_single_table_per_query_row[table].values()))
            table_to_example_match_index_map_per_query_row[table] = indices_validated_in_single_table_per_query_row[table]
            continue
        column = detected_column['column']
        index_set = set()
        for example_row_index, single_example_match_indices in enumerate(detected_column['example_match_indices']):
            index_set.update(set(single_example_match_indices['indices']))
            table_to_example_match_index_map_per_query_row[table][f'query row {example_row_index}'] = single_example_match_indices['indices']

        table_to_example_match_index_map[table] = index_set
    
    # console.log(table_to_example_match_index_map, pretty=False)
    return table_to_example_match_index_map, table_to_example_match_index_map_per_query_row

def get_whole_query_row_find_count(
    query_table_example_row_count,
    table_to_example_match_index_map_per_query_row,
    joined_table

):
    whole_example_row_find_count = 0
    for example_row in [f"query row {i}" for i in range(query_table_example_row_count)]:
        indices = []
        column_list = []
        for table, match_indices_per_query_row in table_to_example_match_index_map_per_query_row.items():
            match_indices_of_single_row = match_indices_per_query_row[example_row]
            indices.append(match_indices_of_single_row)
            column_list.append(f"Index_{table}")

        indices_across_tables = utils.list_product(indices)

        single_query_row_found_in_joined_table = False
        for single_index_set_across_tables in tqdm(indices_across_tables):
            formatted_index = {}
            for i, column in enumerate(column_list):
                formatted_index[column] = single_index_set_across_tables[i]
            matched_rows = db.get_count_of_rows_with_matching_value_in_all_specified_columns(
                df=joined_table,
                conditions=formatted_index
            )
            if matched_rows > 0:
                single_query_row_found_in_joined_table = True
                break

        if single_query_row_found_in_joined_table:
            whole_example_row_find_count += 1
    return whole_example_row_find_count

def get_whole_query_row_find_count_by_equi_match(
    joined_table,
    detected_columns,
    all_dataframes,
    all_simhashes,
    is_numeric_of_columns,
    simhash_hyperplanes
):

    expanded_table_columns = utils.nested_dict()
    table_column_value_map = utils.nested_dict()
    query_table_example_row_count = None
    for detected_column in detected_columns:
        table = detected_column['table']
        column = detected_column['column']
        expanded_table_columns[table][column]= constants.NATURAL_JOIN if is_numeric_of_columns[table][column] else constants.SEMANTIC_HASH_JOIN
        query_values = [example_match_info['value'] for example_match_info in detected_column['example_match_indices']]
        table_column_value_map[table][column] = query_values
        query_table_example_row_count = len(query_values)
    rows_to_be_matched = []

    for query_row_index in range(query_table_example_row_count):
        query_row = {}
        for table, columns in table_column_value_map.items():
            for column, values in columns.items():
                value_for_search = None
                if is_numeric_of_columns[table][column]:
                    value_for_search = values[query_row_index]
                else:
                    embeddings = text_encoder.encode(string_array=[values[query_row_index]])
                    value_for_search = semantic_hash_index.extract_simhash_using_hyperplanes(embeddings=embeddings, hyperplanes=simhash_hyperplanes)[0]
                query_row[f"{table}_{column}"] = value_for_search
        rows_to_be_matched.append(query_row)
    full_joined_table = join_dataframes.build_combined_table(
        index_df=joined_table, all_dataframes=all_dataframes, column_mapping=expanded_table_columns, all_simhashes=all_simhashes, convert_values_to_simhash=True
    )

    query_row_count_in_joined_table = 0
    for query_row in rows_to_be_matched:
        matched_count_in_joined_df = db.get_count_of_rows_with_matching_value_in_all_specified_columns(
            df=full_joined_table, conditions=query_row
        )
        if matched_count_in_joined_df > 0:
            query_row_count_in_joined_table+= 1

    return query_row_count_in_joined_table
    

def semantic_description_to_semantic_type_extraction_using_gpt(user_query):
    """
    user_query=
    [
        {
            "semantic_type_description": "description1",
            "examples":["example11", "example12"]
        },
        {
            "semantic_type_description": "description2",
            "examples":["example21"]
        }
    ]

    """

    """
    output=
    [
        {
            "semantic_type_description": "description1",
            "examples":["example11", "example12"],
            "semantic_type": "semantictype1"
        },
        {
            "semantic_type_description": "description2",
            "examples":["example21"],
            "semantic_type": "semantictype2"
        }
    ]

    """


    template = """
    Below is a list of column ids and their descriptions from a table. Give a name for each of the column without maintaining any programming language convention. The output should be in the following JSON format:

    [{{"id": columnid1, "name": columnname1}}, {{"id": columnid2, "name": columnname2}},]

    input:
    {input}
    """

    index_to_description_map = {}
    prompt_input = ""
    for idx, user_column in enumerate(user_query):
        index_to_description_map[f"{idx}"] = user_column["semantic_type_description"]
        prompt_input += f'column id: {idx}\ncolumn description: {user_column["semantic_type_description"]}\n\n'

    # console.log(prompt_input)

    llm_result = llm_interface.call_gpt_cached(
        template=template,
        run_gpt=True,
        input=prompt_input,
        model_name=llm_interface.MODEL_4O,
        temperature=0
    )

    user_sem_type_translations = utils.extract_json_from_string(llm_result['prediction']['content'])

    for user_sem_type_translation in user_sem_type_translations:
        index = int(user_sem_type_translation["id"])
        user_query[index]["semantic_type"] = user_sem_type_translation["name"]

    return user_query

def get_most_relevant_columns_based_on_semantic_type(user_query):
    all_semantic_type_infos = semantic_type_extractor.get_all_semantic_types_array()
    all_semantic_types_only = [semantic_type_info['semantic_type'] for semantic_type_info in all_semantic_type_infos]
    all_semantic_type_index = text_encoder.get_index_flat_l2_for_list_of_strings(all_semantic_types_only)
    for user_column in user_query:
        semantic_type_by_gpt = user_column['semantic_type']
        best_column_indexes = text_encoder.get_top_k_similar_string_indexes(
            target_string=semantic_type_by_gpt,
            k=50,
            list_IndexFlatL2=all_semantic_type_index
        )
        best_columns = [all_semantic_type_infos[i] for i in best_column_indexes]
        user_column['best_columns_by_semantic_type'] = best_columns

    return user_query

def get_most_relevant_columns_based_on_search_index_and_example_value_match_count(
        user_query,
        K_semantic_type,
        search_index,
        all_semantic_type_infos,
        column_value_indices,
        column_simhash_indices,
        simhash_hyperplanes,
        is_column_numeric,
        tables_without_path
        ):
    """
    Arguments:
    user_query = [
        {
            "semantic_type": "",
            "examples": [
                ""
            ]
        }
    ]

    Returns:
    [
        {
            "semantic_type": "",
            "examples": [
                ""
            ],
            "top_columns_by_semantic_type":{
                "columns": [
                    {
                        "table":"",
                        "column":"",
                        "example_match_count": 0,
                    },
                ],
            },
        }
    ]
    """
    user_query_with_detected_columns = utils.deep_copy(user_query)

    enriched_columns = []

    for query_column in user_query_with_detected_columns:
        enriched_query_column = utils.deep_copy(query_column)
        semantic_type = enriched_query_column['semantic_type']
        examples = enriched_query_column['examples']

        example_embeddings = text_encoder.encode(string_array=examples)
        example_simhashes = semantic_hash_index.extract_simhash_using_hyperplanes(
            embeddings=example_embeddings,
            hyperplanes=simhash_hyperplanes
        )

        semantic_type_embedding = text_encoder.encode([semantic_type])[0]
        # enriched_query_column["top_columns_by_semantic_type"] = text_encoder.get_top_k_closest_strings_using_hnsw(
        #     hnsw_index=search_index,
        #     list_of_objects_of_hnsw=all_semantic_type_infos,
        #     target_embedding=semantic_type_embedding,
        #     K=K_semantic_type
        #     )
        enriched_query_column["top_columns_by_semantic_type"] = text_encoder.get_top_k_closest_strings_using_hnsw(
            hnsw_index=search_index,
            list_of_objects=all_semantic_type_infos,
            target_embedding=semantic_type_embedding,
            K=K_semantic_type
            )
        
        new_top_columns = []
        for top_column_by_semantic_type in enriched_query_column["top_columns_by_semantic_type"]:

            new_top_column = utils.deep_copy(top_column_by_semantic_type)
            index_matches = []
            total_match = 0
            table = new_top_column['table']
            column = new_top_column['column']
            if table in tables_without_path:
                # console.debug("FOUND TABLE WITH NO PATH_________________________________________")
                continue
            if is_column_numeric[table][column]:
                for single_example in examples:
                    indices = column_value_indices[table][column].get(single_example, [])
                    index_matches.append({
                        'value': single_example,
                        'indices': indices
                    })
                    total_match += 0 if len(indices) == 0 else 1
            else:
                for i, single_simhash in enumerate(example_simhashes):
                    indices = column_simhash_indices[table][column].get(single_simhash, [])
                    index_matches.append({
                        'value': examples[i],
                        'indices': indices
                    })
                    total_match += 0 if len(indices) == 0 else 1
            
            new_top_column['example_match_indices'] = index_matches
            new_top_column['example_match_count'] = total_match
            new_top_column['type'] = constants.NATURAL_JOIN if is_column_numeric[table][column] else constants.SEMANTIC_HASH_JOIN
            new_top_columns.append(new_top_column)
        enriched_query_column["top_columns_by_semantic_type"] = new_top_columns
        # utils.file_dump(console.reduce_data(data=utils.deep_copy(new_top_columns), depth=3), f"debug_folder/new_top_columns.json")

        filtered_query_columns = []
        for top_column in enriched_query_column["top_columns_by_semantic_type"]:
            if top_column['example_match_count'] == 0:
                continue

            found_example_with_no_matches = False
            for index_info in top_column['example_match_indices']:
                if len(index_info['indices']) == 0:
                    found_example_with_no_matches = True
                    break

            if found_example_with_no_matches:
                continue

            filtered_query_columns.append(top_column)

        # if len(filtered_query_columns) == 0:
        #     utils.crash_code("filtered_query_columns has zero columns for a particular query column")
        enriched_query_column["top_columns_by_semantic_type"] = filtered_query_columns

        enriched_query_column["top_columns_by_semantic_type"].sort(key=lambda x: x['example_match_count'], reverse=True)

        for top_column in enriched_query_column["top_columns_by_semantic_type"]:
            if top_column["example_match_count"] == 0:
                utils.crash_code("this should not be 0")

        enriched_columns.append(enriched_query_column)

    return enriched_columns

def compute_hnsw_index_from_semantic_types(all_semantic_type_infos):
    all_semantic_types_only = [semantic_type_info['semantic_type'] for semantic_type_info in all_semantic_type_infos]
    hnsw_index = text_encoder.get_index_hnsw_flat_for_list_of_strings(list_of_strings=all_semantic_types_only)
    return hnsw_index

def compute_index_flat_index_from_semantic_types(all_semantic_type_infos):
    all_semantic_types_only = [semantic_type_info['semantic_type'] for semantic_type_info in all_semantic_type_infos]
    index_flat_index = text_encoder.get_index_flat_for_list_of_strings(list_of_strings=all_semantic_types_only)
    return index_flat_index

def get_hnsw_index_dill_path():
    return f"{fpm.get_datalake_path()}/hnsw.dill"

def save_semantic_type_hnsw_index(hnsw_index):
    utils.file_dump(hnsw_index, get_hnsw_index_dill_path())

def load_semantic_type_hnsw_index():
    return utils.file_load(get_hnsw_index_dill_path())

def get_index_flat_index_dill_path():
    return f"{fpm.get_datalake_path()}/index_flat.dill"

def save_semantic_type_index_flat_index(index_flat_index):
    utils.file_dump(index_flat_index, get_index_flat_index_dill_path())

def load_semantic_type_index_flat_index():
    return utils.file_load(get_index_flat_index_dill_path())

def get_template_for_best_columns_by_gpt():
    return """
    Below is a column name of a table. There is also a list of descriptions with ids. 
    Find the top 10 description ids where the description is similar as the column name.
    Make sure the output is only a JSON array of ids like the below.

    [id1, id2, id3,...]

    {input}
    """

def get_best_columns_based_on_gpt(user_query):

    for user_column in user_query:
        semantic_type = user_column['semantic_type']
        best_columns_based_on_semantic_types = user_column['best_columns_by_semantic_type']
        input_string_for_gpt = f'target column name: {semantic_type}\n\n'

        for column_prompt_id, columns_with_semtype in enumerate(best_columns_based_on_semantic_types):
            input_string_for_gpt += f'id: {column_prompt_id}\ndescription: {columns_with_semtype["semantic_type"]}\n\n'

        # console.log(input_string_for_gpt)

        gpt_result = llm_interface.call_gpt_cached(
            template=get_template_for_best_columns_by_gpt(),
            run_gpt=True,
            input=input_string_for_gpt,
            model_name=llm_interface.MODEL_4O,
            temperature=0
        )

        best_column_ids_by_gpt = utils.extract_json_from_string(gpt_result['prediction']['content'])
        best_columns_by_gpt = [best_columns_based_on_semantic_types[int(i)] for i in best_column_ids_by_gpt]
        user_column['best_columns_by_gpt'] = best_columns_by_gpt

    return user_query

def get_best_columns_based_on_examples(user_query):
    all_top_n_values = db.get_top_n_values_of_all_tables(n=10)
    all_top_n_value_embeddings = table_embeddings.all_column_top_n_value_embeddings(all_top_n_values)
    for user_column in user_query:
        examples = user_column['examples']
        example_embeddings = text_encoder.encode(string_array=examples)
        top_columns_by_gpt = user_column['best_columns_by_gpt']
        columns_with_similarity_info = []
        for column_info in top_columns_by_gpt:
            column = column_info['column']
            table = column_info['table']

            top_embeddings = all_top_n_value_embeddings[table][column]
            
            top_k_indexes, overall_mean_cosine_similarity = text_encoder.top_k_cosine_similarity(
                embeddings_list1=example_embeddings,
                embeddings_list2=top_embeddings,
                k=10
                )

            copied_column_info = column_info.copy()
            copied_column_info['similarity'] = overall_mean_cosine_similarity
            columns_with_similarity_info.append(copied_column_info)
        
        columns_with_similarity_info.sort(key=lambda x:x['similarity'], reverse=True)

        user_column['columns_sorted_by_example'] = columns_with_similarity_info

    return user_query

def get_all_possible_paths_from_user_query(user_query, similarity_threshold):
    tables_for_all_column = []
    path_combination_max_count = 1000
    for user_column in user_query:
        best_columns = [column_info for column_info in user_column['columns_sorted_by_example'] if column_info['similarity'] > similarity_threshold]
        unique_tables = utils.remove_dulicates_in_string_list([best_column['table'] for best_column in best_columns])
        tables_for_all_column.append(unique_tables)

    # console.log(tables_for_all_column,pretty=True)
    all_possible_needed_join_paths = utils.list_product(tables_for_all_column)[:path_combination_max_count]
    all_possible_needed_join_paths = [utils.remove_dulicates_in_string_list(candidate_list) for candidate_list in all_possible_needed_join_paths]
    # console.log(all_possible_needed_join_paths, pretty=True)
    return all_possible_needed_join_paths

def get_top_join_paths(all_possible_candidate_table_list):
    all_table_indices = pairwise_simple_path_index.get_table_index_map()
    table_to_path_index = pairwise_simple_path_index.load_table_to_path_index()
    all_join_orders = pairwise_simple_path_index.get_all_join_orders()
    all_paths = pairwise_simple_path_index.load_all_pairwise_simple_paths()

    final_join_orders = []
    single_tables = []
    for candidate_table_list in all_possible_candidate_table_list:
        # console.log(candidate_table_list, pretty=True)
        if len(candidate_table_list) == 1:
            single_tables.append(candidate_table_list[0])
            continue
        if len(candidate_table_list) == 0:
            utils.crash_code("a candidate list has 0 entries")
        best_join_path_result = pairwise_simple_path_index.get_join_order_from_initial_tables_using_table_path_inverted_index(
            initial_tables=candidate_table_list,
            all_paths=all_paths,
            all_table_indices=all_table_indices,
            table_to_path_index=table_to_path_index,
            all_join_orders=all_join_orders
        )

        if best_join_path_result is None:
            continue

        join_order, join_path, cardinality = best_join_path_result
        final_join_orders.append({
            'join_order': join_order,
            'join_path': join_path,
            'cardinality': cardinality
        })
    
    return final_join_orders, single_tables

def compute_column_value_indices_for_one_column(argument, common_argument, common_argument_for_batch):
    table = argument['table']
    column = argument['column']
    values = argument['values'] # pandas series

    # Create a dictionary to store value to indices mapping
    value_indices = defaultdict(list)
    
    # Iterate through the column values with their indices
    for index, value in enumerate(values):
        value_indices[value].append(index)
    return {
        'table': table,
        'column': column,
        'indices': value_indices
    }

def compute_column_simhash_indices_for_one_column(argument, common_argument, common_argument_for_batch):
    table = argument['table']
    column = argument['column']
    simhashes = argument['simhashes']  # Assuming this contains the column values
    
    # Create a dictionary to store value to indices mapping
    simhash_indices = defaultdict(list)
    
    # Iterate through the column values with their indices
    for index, simhash in enumerate(simhashes):
        simhash_indices[simhash].append(index)
    
    return {
        'table': table,
        'column': column,
        'indices': simhash_indices
    }

def compute_column_value_and_simhash_indices_for_all_column(all_dataframes, all_simhashes):
    parallel_data = [{'table': table, 'column': column, 'values': df[column]} for table, df in all_dataframes.items() for column in df.columns]
    results = parallel.execute(
        func=compute_column_value_indices_for_one_column,
        argument_list=parallel_data
    )
    column_value_indices = utils.nested_dict()
    for result in results:
        column_value_indices[result['table']][result['column']] = result['indices']

    parallel_data_simhash = [{'table': table, 'column': column, 'simhashes': df_simhash[column]} for table, df_simhash in all_simhashes.items() for column in df_simhash.columns]

    results_simhash = parallel.execute(
        func=compute_column_simhash_indices_for_one_column,
        argument_list=parallel_data_simhash
    )
    simhash_value_indices = utils.nested_dict()
    for result in results_simhash:
        simhash_value_indices[result['table']][result['column']] = result['indices']


    return column_value_indices, simhash_value_indices


def get_candidate_paths(user_query_with_detected_columns):
    all_candidate_tables = [single_user_query_with_detected_columns['top_columns_by_semantic_type'] for single_user_query_with_detected_columns in user_query_with_detected_columns]
    # all_needed_join_paths = utils.list_product(all_candidate_tables)
    all_needed_join_paths = utils.sort_product_by_index_sum(all_candidate_tables)
    # console.log(console.reduce_data(all_needed_join_paths, depth=1))
    # exit(0)
    all_needed_join_paths = [{"candidate_join_path": candidate_join_path} for candidate_join_path in all_needed_join_paths]
    return all_needed_join_paths


if __name__ == "__main__":
    pass




