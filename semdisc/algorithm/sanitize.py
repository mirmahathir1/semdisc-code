from custom_lib import utils
from custom_lib import console
from custom_lib import parallel

from custom_lib import db
from semdisc.lib import file_path_manager as fpm
from collections import defaultdict
import os
from tqdm import tqdm
import shutil

"""
TODO: a column was found in fws that has no column name and no values. yet it still was there
"""
def get_new_names_for_csv(directory_path):
    file_name_limit = 50

    file_names = [utils.get_filename_without_extension(f) for f in utils.listdir(directory_path, '.csv') if os.path.isfile(os.path.join(directory_path, f))]

    start_clusters = defaultdict(list)
    end_clusters = defaultdict(list)

    for name in file_names:
        start_prefix = name[:file_name_limit]
        end_suffix = name[-file_name_limit:]

        start_clusters[start_prefix].append(name)
        end_clusters[end_suffix].append(name)

    renamed_files = {}
    new_names = []

    counts = {}

    duplicate_names = defaultdict(int)

    for idx, name in enumerate(file_names):
        start_prefix = name[:file_name_limit]
        end_suffix = name[-file_name_limit:]

        start_cluster = start_clusters[start_prefix]
        end_cluster = end_clusters[end_suffix]

        if len(start_cluster) == 1 and len(end_cluster) == 1:
            renamed_files[name] = start_prefix
        elif len(start_cluster) > len(end_cluster):
            renamed_files[name] = end_suffix
        elif len(start_cluster) < len(end_cluster):
            renamed_files[name] = start_prefix
        else:
            duplicate_names[start_prefix] = duplicate_names[start_prefix] + 1
            renamed_files[name] = f"{start_prefix}_{duplicate_names[start_prefix]}"

        if renamed_files[name] in counts:
            counts[renamed_files[name]] += 1
            renamed_files[name] = f"{renamed_files[name]}{counts[renamed_files[name]]}"
        else:
            counts[renamed_files[name]] = 0
        
        new_names.append(renamed_files[name])

    if not len(new_names) == len(set(new_names)):
        utils.crash_code("duplicate file names found.")

    csv_added_names = {}

    for old_name, new_name in renamed_files.items():
        csv_added_names[f"{directory_path}/{old_name}.csv"] = f"{directory_path}/{new_name}.csv"
    
    return csv_added_names

max_csv_length = 1000

def sanitize_single_table(argument, common_argument, common_argument_for_batch):
    table = argument
    sampling_enabled = common_argument['sampling_enabled']
    path = common_argument['path']

    csv_path = os.path.join(path, f"{table}.csv")
    # utils.remove_non_utf8_lines(os.path.join(fpm.get_uploaded_datasets_path(), f"{table}.csv"))
    utils.remove_non_utf8_lines(csv_path)
    table_dataframe_dict = db.get_dataframe_from_csv_by_table_name(argument=table, common_argument={'override_path': path}, common_argument_for_batch=None)
    utils.delete(csv_path)
    df = table_dataframe_dict['dataframe']
    df.columns = [utils.os_saveable_filename(col) for col in df.columns]
    dataframe_length = len(df)
    if dataframe_length == 0:
        return
    if sampling_enabled:
        if dataframe_length > max_csv_length:
            df = df.sample(max_csv_length)
    df = df.dropna(how='all', axis=1)
    df.to_csv(csv_path, index=False)

def santize_data_function(path = None):
    console.log(f"---sanitize_data started")
    if path is None:
        path = fpm.get_uploaded_datasets_path()

    all_tables = db.get_all_uploaded_csv_names(path=path)

    console.log(f"all tables: {all_tables}")

    parallel.execute(
        func=sanitize_single_table,
        argument_list=all_tables,
        common_arguments={
            'sampling_enabled': True,
            'path': path
        },
        function_for_batch=utils.empty_function
        )
    
    console.log("renaming csv files")
    rename_all_csv_files(path)

    console.log(f"sanitize_data complete")

def rename_single_file(old_name, new_name):
    try:
        os.rename(old_name, new_name)
        # printbroken(f"File renamed from '{old_name}' to '{new_name}'")
    except FileNotFoundError:
        utils.crash_code(f"File '{old_name}' not found.")
    except FileExistsError:
        utils.crash_code(f"File '{new_name}' already exists.")
    except Exception as e:
        utils.crash_code(f"Error renaming file: {e}")

def rename_all_csv_files(path):
    for old_name, new_name in get_new_names_for_csv(path).items():
        rename_single_file(old_name=old_name, new_name=new_name)

def delete_all_processed_data():
    exceptions = set([
        'original',
        'original.zip'
    ])

    path = fpm.get_datalake_path()

    content_list = set(utils.listdir(path))
    
    deletable_set = content_list - exceptions

    for deletable_item in deletable_set:
        utils.delete(f"{path}/{deletable_item}")

def copy_datalakes_from_original():
    console.log("---copy_datalakes_from_original")
    utils.reset_directory(fpm.get_uploaded_datasets_path())
    for file_path in tqdm(os.listdir(fpm.get_original_csv_path())):
        shutil.copy(f"{fpm.get_original_csv_path()}/{file_path}", f"{fpm.get_uploaded_datasets_path()}/{file_path}")

if __name__ == '__main__':
    fpm.startup()
    delete_all_processed_data()
    
