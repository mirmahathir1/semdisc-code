from custom_lib import utils
from semdisc.lib import file_path_manager as fpm
from semdisc import result_gen

result_gen.startup()
def get_query_table_json_path():
    return fpm.get_datalake_path() + "/query_tables.json"

all_queries = utils.file_load(get_query_table_json_path())
all_queries = utils.sample_list(all_queries, 100)
utils.file_dump(all_queries, get_query_table_json_path())
