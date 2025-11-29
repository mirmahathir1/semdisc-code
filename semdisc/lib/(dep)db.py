from custom_lib import utils, console
import semdisc.lib.file_path_manager as fpm
import os
import pandas as pd
import json
from collections import defaultdict
import numpy as np
import itertools
import glob
from custom_lib import parallel

def get_count_of_rows_with_matching_value_in_all_specified_columns(df, conditions):
    # conditions = {
    #     'A': 1,       # Column A should equal 1
    #     'B': 'x',     # Column B should equal 'x'
    #     'C': 10       # Column C should equal 10
    # }

    # Create a boolean mask for all conditions
    
    return (df[list(conditions.keys())] == pd.Series(conditions)).all(axis=1).sum()

def remove_duplicate_columms(df):
    df = df.loc[:, ~df.columns.duplicated(keep='last')]
    return df

def remove_duplicate_rows_based_on_selected_columns(df, column_list):
    return df.drop_duplicates(subset=column_list, keep='first', inplace=False)

def df_to_frontend_format(df):
    """ dataframe convert to list of rows [ {col1:value1, col2:value2 ...}, ... ]"""
    res = []
    N,M = len(df), len(df.columns)
    for i in range(0,N):
        curr_row = {}
        for j in range(0,M):
            curr_row[str(df.columns[j])] = str(df.iloc[i][df.columns[j]])
        res += [curr_row]
    return res

def top_n_occurying_values_in_column(dataframe, column_name, n, to_string=False):
    values = dataframe[column_name].dropna().value_counts().nlargest(n).index.tolist()
    if to_string == True:
        values = list(map(str, values))
    return values

def get_all_uploaded_csv_names(override_path = None):
    path = fpm.get_uploaded_datasets_path()
    if override_path is not None:
        path = override_path

    all_raw_names =  utils.listdir(path, '.csv')
    all_table_names =  [utils.get_filename_without_extension(file_name) for file_name in all_raw_names]

    # printbroken(all_raw_names)
    # printbroken(len(all_raw_names))
    # printbroken(all_table_names)
    # printbroken(len(all_table_names))

    return all_table_names

def get_dataframe_from_csv_by_table_name(argument, common_argument, common_argument_for_batch):
    table = argument
    override_path = common_argument['override_path']
    path = fpm.get_uploaded_datasets_path()
    if override_path is not None:
        path = override_path
    return {
        'table': table,
        'dataframe': get_dataframe_from_path(f"{path}/{table}.csv")
    }

def load_all_dataframes(override_path = None):
    all_dataframes = {}
    all_tables = get_all_uploaded_csv_names(override_path)

    results = parallel.execute(
        func=get_dataframe_from_csv_by_table_name,
        argument_list=all_tables,
        common_arguments={'override_path': override_path}
    )

    all_dataframes = {}
    for result in results:
        all_dataframes[result['table']] = result['dataframe']

    return all_dataframes

def get_dataframe_from_path(path):
    return pd.read_csv(path, on_bad_lines='skip', engine='python', dtype=str).applymap(lambda x: np.nan if isinstance(x, str) and x.lower() == 'nan' else x)

def save_dataframe_to_path(path, dataframe):
    dataframe.to_csv(path, index=False)

def get_column_names_from_csv(table, override_path = None):
    path = fpm.get_uploaded_datasets_path()
    if override_path is not None:
        path = override_path

    return pd.read_csv(f"{path}/{table}.csv", nrows=0).columns.tolist()

def get_numeric_portion_of_column(dataframe, column_name):
    concatenated_string = "".join(map(str, top_n_occurying_values_in_column(dataframe=dataframe, column_name=column_name, n=10)))
    num_numeric_chars = sum(c.isdigit() for c in concatenated_string)
    if len(concatenated_string) == 0:
        return 0
    return (num_numeric_chars / len(concatenated_string))

def frontend_table_to_df(frontend_grid, is_initial_result):
    """ convert table from frontend (json {data:[ 0:[row 0 values], ... ], columns:[]} ) to pandas dataframe """
    grid = json.loads(frontend_grid) if is_initial_result else frontend_grid
    column_len, row_len = len(grid['columns']), len(grid['data'])

    res_dict = defaultdict(list)
    for i in range(0, row_len):
        for j in range(0, column_len):
            value = grid['data'][str(i)][j]
            res_dict[i] += [value]
    res_df = pd.DataFrame.from_dict(res_dict, orient="index")
    res_df.columns = grid['columns']

    # console.log(res_df)
    return res_df

def get_all_pairs_of_tables(all_dataframes):
    return list(itertools.combinations(list(all_dataframes.keys()), 2))

def delete_all_csvs():
    console.log("deleting all csv files: ")

    for file in glob.glob(f"{fpm.get_uploaded_datasets_path()}/*"):
        os.remove(file)

def get_all_column_headers(all_dataframes):
    all_columns = []
    for table, dataframe in all_dataframes.items():
        all_columns += [{"table": table, "column": column} for column in dataframe.columns]
    
    return all_columns

def get_top_n_values_of_all_tables(n):
    all_n_values = {}
    for table in get_all_uploaded_csv_names():
        all_n_values[table]={}
        dataframe = get_dataframe_from_csv_by_table_name(table)
        for column in dataframe.columns:
            all_n_values[table][column] = top_n_occurying_values_in_column(dataframe=dataframe, column_name=column, n=n, to_string=True)
    
    return all_n_values

def get_column_name_dictionary(all_dataframes):
    column_name_dictionary = {}
    for table, dataframe in all_dataframes.items():
        column_name_dictionary[table] = list(dataframe.columns)
    return column_name_dictionary