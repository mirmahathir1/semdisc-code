from tqdm import tqdm
from custom_lib import utils, parallel, console, text_encoder
from custom_lib import db
from semdisc.lib import file_path_manager as fpm
import random
import numpy as np

def get_column_header_similarity_for_one_pair(argument, common_argument, common_argument_for_batch):
    header_similarities = utils.nested_dict()
    (column_info_1, column_info_2) = argument
    table1 = column_info_1['table']
    table2 = column_info_2['table']
    column1 = column_info_1['column']
    column2 = column_info_2['column']
    header_embeddings = common_argument['header_embeddings']
    similarity = utils.get_cosine_similarity(embedding1=header_embeddings[table1][column1], embedding2=header_embeddings[table2][column2])

    header_similarities[table1][table2][column1][column2] = similarity
    header_similarities[table2][table1][column2][column1] = similarity
    
    return header_similarities

def all_column_header_similarities():
    console.log("getting column header similarities")
    all_columns = db.get_all_column_headers()
    only_headers = [column_info['column'] for column_info in all_columns]
    embeddings = text_encoder.encode(only_headers)
    header_embeddings = utils.nested_dict()
    for idx, column_info in tqdm(enumerate(all_columns)):
        header_embeddings[column_info['table']][column_info['column']] = embeddings[idx]
    
    column_pairs = utils.get_combinations(all_columns,2)

    header_similarities = utils.nested_dict()
    results = parallel.execute(
        func=get_column_header_similarity_for_one_pair,
        argument_list=column_pairs,
        common_arguments={'header_embeddings': header_embeddings},
        function_for_batch=utils.empty_function
    )

    console.log("merging results")
    for result in tqdm(results):
        for table1 in result:
            for table2 in result[table1]:
                for column1 in result[table1][table2]:
                    for column2 in result[table1][table2][column1]:
                        header_similarities[table1][table2][column1][column2]= result[table1][table2][column1][column2]
    
    return header_similarities

def get_all_table_embeddings_pickle_path():
    return f"{fpm.get_datalake_path()}/all_table_embeddings.pickle"

def load_all_table_embeddings():
    return utils.file_load(get_all_table_embeddings_pickle_path())

def save_all_table_embeddings(all_embeddings):
    utils.file_dump(all_embeddings, get_all_table_embeddings_pickle_path())

def compute_embeddings_for_single_table_column(argument, common_argument, common_argument_for_batch):
    table_column = argument
    values = table_column['values']
    table = table_column['table']
    column = table_column['column']
    normalize = common_argument['normalize']
    model = common_argument_for_batch

    encoded_values = text_encoder.encode(string_array=values, normalize=normalize, model=model)
    return {
        'table': table,
        'column': column,
        'embeddings': encoded_values
    }

def compute_all_table_embeddings(all_dataframes, normalize):
    console.log(f"---rebuild embeddings of all tables: started")

    all_table_column_info = []
    for table, df in all_dataframes.items():
        for column in df.columns:
            all_table_column_info.append({
                'table': table,
                'column': column,
                'values': list(map(str, df[column].tolist()))
            })
    
    random.shuffle(all_table_column_info)

    console.log("parallel extraction started")
    results = parallel.execute(
        func=compute_embeddings_for_single_table_column,
        argument_list=all_table_column_info,
        common_arguments={'normalize': normalize},
        function_for_batch=text_encoder.create_sentence_encoder
    )

    all_embeddings = utils.nested_dict()

    for result in results:
        table = result['table']
        column = result['column']
        embeddings = result['embeddings']
        all_embeddings[table][column] = embeddings

    console.log(f"rebuild embeddings of all tables: complete")
    return all_embeddings

if __name__ == '__main__':
    result_gen.startup()