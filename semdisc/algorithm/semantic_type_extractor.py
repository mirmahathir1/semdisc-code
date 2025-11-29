from custom_lib import console
from custom_lib import llm_interface
from custom_lib import text_encoder
from custom_lib import utils
from custom_lib import parallel
from custom_lib import db
from semdisc.lib import file_path_manager as fpm

def get_semantic_type_directory():
    return f"{fpm.get_datalake_path()}/semantic_types"

def call_gpt_for_semantic_types(all_dataframes, semantic_type_directory):
    template = """
    Below is a list of column ids and a sample of their values seperated by commas from a table. Give detailed elaborate name for each column in the following format:
    [{{"id": columnid1, "name":columnname1}}, {{"id": columnid2, "name":columnname2}}, ...]

    input:

    {input}

    """

    sample_count = 20
    sample_length = 150

    for table, dataframe in all_dataframes.items():
        all_column_values = []
        for column in dataframe.columns:
            unique_values = db.top_n_occurying_values_in_column(dataframe=dataframe, column_name=column, n=sample_count, to_string=True)
            unique_values = ", ".join(unique_values)
            unique_values = unique_values[:sample_length].replace('\n', ' ')
            value_string = "id: "+column+"\nsample values: "+unique_values
            all_column_values.append(value_string)

        all_column_data  = "\n\n".join(all_column_values)

        all_column_data = f"table name: {table}\n\n" + all_column_data

        result = llm_interface.call_gpt_cached(
            template=template,
            run_gpt=True,
            input=all_column_data,
            temperature=0,
            model_name=llm_interface.MODEL_4O
        )

        raw_output = result['prediction']['content']

        semantic_types = utils.extract_json_from_string(raw_output)

        utils.file_dump(semantic_types, f"{semantic_type_directory}/{table}.json")

        natural_text_output = llm_interface.get_natural_text_prompt_and_prediction(result)

        utils.text_dump(natural_text_output, f"{semantic_type_directory}/{table}.txt")

# last balance 9.28 usd
# after extracing semantic types for drugcentral 9.10 usd

def compile_all_semantic_types(path=None):
    if path is None:
        path = get_semantic_type_directory()

    all_table_semantic_types = []
    for table_json in utils.listdir(path, '.json'):
        table_name = utils.get_filename_without_extension(table_json)
        semantic_type_infos = utils.file_load(f"{path}/{table_json}")
        for semantic_type_info in semantic_type_infos:
            column = str(semantic_type_info['column'])
            all_table_semantic_types.append({
                'table': table_name,
                'column': column,
                'semantic_type': semantic_type_info['semantic_type']
            })
    
    return all_table_semantic_types

def get_semantic_types_single_file_path():
    return f"{fpm.get_datalake_path()}/semantic_type_single_file.json"

def save_all_semantic_types_single_file(semantic_type_infos):
    utils.file_dump(semantic_type_infos, get_semantic_types_single_file_path())

def load_all_semantic_types_single_file():
    return utils.file_load(get_semantic_types_single_file_path())

def get_semantic_type_similarity_for_single_pair(argument, common_argument, common_argument_for_batch):
    (column_info_1, column_info_2) = argument
    table1 = column_info_1['table']
    table2 = column_info_2['table']
    column1 = column_info_1['column']
    column2 = column_info_2['column']
    semantic_type_embedding_dictionary = common_argument['semantic_type_embedding_dictionary']
    similarity = utils.get_cosine_similarity(semantic_type_embedding_dictionary[table1][column1], semantic_type_embedding_dictionary[table2][column2])

    return {
        "table1": table1,
        "table2": table2,
        "column1": column1,
        "column2": column2,
        "similarity": similarity
    }

def compute_semantic_type_similarities(all_semantic_types):
    console.log("getting semantic type similarities")
    only_semantic_types = [column_info["semantic_type"] for column_info in all_semantic_types]
    semantic_type_embeddings = text_encoder.encode(string_array= only_semantic_types)
    semantic_type_embedding_dictionary = utils.nested_dict()
    for idx, semantic_type in enumerate(all_semantic_types):
        semantic_type_embedding_dictionary[semantic_type['table']][semantic_type['column']] = semantic_type_embeddings[idx]

    semantic_type_similarities = utils.nested_dict()

    column_pairs = utils.get_combinations(all_semantic_types, 2)

    results = parallel.execute(
        func=get_semantic_type_similarity_for_single_pair,
        argument_list=column_pairs,
        common_arguments={'semantic_type_embedding_dictionary': semantic_type_embedding_dictionary},
        function_for_batch=utils.empty_function
    )

    semantic_type_similarities = utils.nested_dict()
    
    for result in results:
        table1 = result['table1']
        table2 = result['table2']
        column1 = result['column1']
        column2 = result['column2']
        similarity = result['similarity']
        semantic_type_similarities[table1][table2][column1][column2] = similarity
        semantic_type_similarities[table2][table1][column2][column1] = similarity

    semantic_type_similarities = utils.convert_to_regular_dict(semantic_type_similarities)
    return semantic_type_similarities

def get_semantic_type_similarities_path():
    return f"{fpm.get_datalake_path()}/semantic_type_similarities.pickle"

def save_semantic_type_similarities(semantic_type_similarities):
    utils.file_dump(semantic_type_similarities, get_semantic_type_similarities_path())

def load_semantic_type_similarities():
    return utils.file_load(get_semantic_type_similarities_path())

def compute_semantic_type_dictionary(all_dataframes):
    all_semantic_types = load_all_semantic_types_single_file()
    semantic_type_dictionary = utils.nested_dict()
    for semantic_type_info in all_semantic_types:
        table = semantic_type_info['table']
        column = semantic_type_info['column']
        semantic_type = semantic_type_info['semantic_type']
        if column in all_dataframes[table].columns:
            semantic_type_dictionary[table][column] = semantic_type
        else:
            utils.crash_code("unknown column that does not exist for semantic types")
    return semantic_type_dictionary

def get_and_save_semantic_types(all_dataframes, semantic_type_directory=None):
    if semantic_type_directory is None:
        semantic_type_directory = get_semantic_type_directory()
    utils.create_directory(semantic_type_directory)
    call_gpt_for_semantic_types(all_dataframes=all_dataframes, semantic_type_directory=semantic_type_directory)
    backup_semantic_types(semantic_type_directory=semantic_type_directory)
    compute_refined_semantic_types_for_missing_responses(semantic_type_directory=semantic_type_directory, all_dataframes=all_dataframes)
    save_all_semantic_types_single_file(compile_all_semantic_types())

def backup_semantic_types(semantic_type_directory):
    backup_directory = f"{get_semantic_type_backup_directory_timestamped()}/{utils.get_formatted_time()}"

    utils.create_directory(get_semantic_type_backup_directory_timestamped())

    utils.copy_file_or_directory(source=semantic_type_directory, destination=backup_directory)

def get_semantic_type_backup_directory_timestamped():
    return f"{fpm.get_datalake_path()}/semantic_types_original_responses"

def compute_refined_semantic_types_for_missing_responses(semantic_type_directory, all_dataframes):
    for table, dataframe in all_dataframes.items():
        all_semantic_types = utils.file_load(f"{semantic_type_directory}/{table}.json")
        refined_semantic_types = []
        all_column_ids_by_gpt = [gpt_result['id'] for gpt_result in all_semantic_types]
        all_semantic_types_by_gpt = [gpt_result['name'] for gpt_result in all_semantic_types]
        for column in dataframe.columns:
            best_matched_column_id_by_gpt = utils.best_match_difflib(strings=all_column_ids_by_gpt, query=column)
            best_match_index = all_column_ids_by_gpt.index(best_matched_column_id_by_gpt)
            semantic_type_of_column = all_semantic_types_by_gpt[best_match_index]
            refined_semantic_types.append({'column': column, 'semantic_type': semantic_type_of_column, 'id_by_gpt': best_matched_column_id_by_gpt})
        utils.file_dump(refined_semantic_types, f"{semantic_type_directory}/{table}.json")