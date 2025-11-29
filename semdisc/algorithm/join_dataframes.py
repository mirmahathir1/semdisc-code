from custom_lib import utils
from custom_lib import console
import pandas as pd
from custom_lib import db
from semdisc.lib import constants

def prepend_column_names_with_table_name_and_add_index(tables_ready_for_join):
    """
    Prepend table name to column names.
    """
    for table in tables_ready_for_join:
        tables_ready_for_join[table].columns = [f"{table}_{col}" for col in tables_ready_for_join[table].columns]
        tables_ready_for_join[table][f'Index_{table}'] = tables_ready_for_join[table].index
    return tables_ready_for_join

def calculate_true_join_cardinality(tables_ready_for_join, join_order, self_join=False):

    max_memory_usage = 0

    start_time = utils.get_current_time()

    wasted_time = 0

    first_table_name = join_order[0]["parent"]
    first_column_name = join_order[0]["parent_column"]
    value_counts = tables_ready_for_join[first_table_name].groupby([f"{first_table_name}_{first_column_name}"]).size().to_dict()

    wasted_start_time = utils.get_current_time()
    max_memory_usage = max(max_memory_usage, utils.get_size_in_bytes_using_pickle(value_counts))
    wasted_time += (utils.get_current_time() - wasted_start_time)

    already_joined_tables = set([first_table_name])

    for join_index in range(len(join_order)-1):
        current_join_order = join_order[join_index]
        next_join_order = join_order[join_index + 1]
    
        child_table = current_join_order["child"]
        child_table_current_col = current_join_order["child_column"]
        child_table_next_col = next_join_order["parent_column"]

        if self_join == False and child_table in already_joined_tables:
            continue
        else:
            already_joined_tables.add(child_table)
        
        grouped_by_values = tables_ready_for_join[child_table].groupby([f"{child_table}_{child_table_current_col}", f"{child_table}_{child_table_next_col}"]).size().to_dict()
        
        # if child_table_current_col == child_table_next_col:
        #     utils.crash_code("manual exit")
        
        new_value_counts = {}
        for (current_col_value, next_col_value), count in grouped_by_values.items():
            if current_col_value in value_counts:
                new_value_counts[next_col_value] = new_value_counts.get(next_col_value, 0) + value_counts[current_col_value] * count

        wasted_start_time = utils.get_current_time()
        max_memory_usage = max(max_memory_usage, utils.get_size_in_bytes_using_pickle(value_counts) + utils.get_size_in_bytes_using_pickle(new_value_counts))
        wasted_time += (utils.get_current_time() - wasted_start_time)

        value_counts = new_value_counts

    last_table_name = join_order[-1]["child"]
    last_column_name = join_order[-1]["child_column"]
    last_value_counts = tables_ready_for_join[last_table_name].groupby([f"{last_table_name}_{last_column_name}"]).size().to_dict()

    cardinality = 0
    for last_col_value, count in last_value_counts.items():
        if last_col_value in value_counts:
            cardinality += value_counts[last_col_value] * count

    end_time = utils.get_current_time()
    time_taken = (end_time - start_time)-wasted_time

    return {"cardinality": cardinality, "time": time_taken, 'max_memory_usage': max_memory_usage}

def perform_joins(tables_ready_for_join, join_order, sampling_size = None, self_join = False):
    """
    Perform a series of joins on tables as specified in join_order.
    
    Args:
        tables_dict: Dictionary where keys are table names and values are DataFrames
        join_order: List of dictionaries specifying join operations
        
    Returns:
        The final joined DataFrame
    """
    # Make a copy of the tables dictionary to avoid modifying the original
    start_time = utils.get_current_time()
    final_result = tables_ready_for_join[join_order[0]["parent"]]

    if self_join == False:
        new_join_order = []
        already_seen_tables = [join_order[0]['parent']]
        for join in join_order:
            child_table = join['child']
            if child_table in already_seen_tables:
                continue
            else:
                new_join_order.append(join)
                already_seen_tables.append(child_table)

    join_order = new_join_order

    for join in join_order:
        parent_table = join["parent"]
        child_table = join["child"]
        parent_col = join["parent_column"]
        child_col = join["child_column"]
    
    for join in join_order:
        parent_table = join["parent"]
        child_table = join["child"]
        parent_col = join["parent_column"]
        child_col = join["child_column"]
        
        # Perform the join
        final_result = pd.merge(
            final_result,
            tables_ready_for_join[child_table],
            left_on=f"{parent_table}_{parent_col}",
            right_on=f"{child_table}_{child_col}",
            how='inner'
        )
        if sampling_size is not None and len(final_result) > sampling_size:
            final_result = final_result.sample(sampling_size)
        
    end_time = utils.get_current_time()
    time_taken = end_time - start_time

    return {"joined_table": final_result, "time": time_taken}

def build_combined_table(index_df, all_dataframes, all_simhashes, column_mapping, convert_values_to_simhash):
    """
    Build a combined table by joining data from multiple tables based on indexes.
    
    Parameters:
    - index_df: DataFrame containing indexes for each table
    - all_dataframes: Dictionary with table names as keys and DataFrames as values
    - column_mapping: Dictionary mapping table names to lists of columns to include
    
    Returns:
    - A combined DataFrame with columns from all specified tables
    """
    # Initialize the result DataFrame with the index_df
    combined_df = index_df.copy()

    # Iterate through each table in the column mapping
    for table_name, columns in column_mapping.items():
        # Get the corresponding index column name in index_df
        index_col = f"Index_{table_name}"
        
        # For each column to include from this table
        for column_name, type in columns.items():
            # Create the new column name
            new_col_name = f"{table_name}_{column_name}"

            # Map the values from the source table using the indexes
            selected_series = None

            if convert_values_to_simhash:
                if type == constants.NATURAL_JOIN:
                    selected_series = all_dataframes[table_name][column_name]
                elif type == constants.SEMANTIC_HASH_JOIN:
                    selected_series = all_simhashes[table_name][column_name]
                else:
                    utils.crash_code("column type invalid")
            else:
                selected_series = all_dataframes[table_name][column_name]

            combined_df[new_col_name] = combined_df[index_col].map(selected_series)
    
    return combined_df

def remove_duplicate_rows_ignoring_index(tables_ready_for_join):
    new_tables_ready_for_join = {}
    for table, dataframe in tables_ready_for_join.items():
        non_index_columns = [column for column in dataframe.columns if "Index_" not in column]
        new_tables_ready_for_join[table] = db.remove_duplicate_rows_based_on_selected_columns(
            df=dataframe,
            column_list=non_index_columns
        )

    return new_tables_ready_for_join


def prepare_tables_with_only_join_columns_with_nan_dropped_for_cardinality_estimation(join_order, all_dataframes, all_simhashes):
    """
    Prepare tables with only join columns based on the join order.
    """
    tables_ready_for_join = {}
    for join in join_order:
        parent_table = join["parent"]
        child_table = join["child"]
        parent_column = join["parent_column"]
        child_column = join["child_column"]
        join_type = join["join_type"]

        if parent_table not in tables_ready_for_join:
            tables_ready_for_join[parent_table] = pd.DataFrame()

        if child_table not in tables_ready_for_join:
            tables_ready_for_join[child_table] = pd.DataFrame()

        if join_type == "semantichash":
            tables_ready_for_join[parent_table][parent_column] = all_simhashes[parent_table][parent_column]
            tables_ready_for_join[child_table][child_column] = all_simhashes[child_table][child_column]
        else:
            try:
                tables_ready_for_join[parent_table][parent_column] = all_dataframes[parent_table][parent_column]
            except Exception as e:
                console.log("parent table:", parent_table, "parent column:", parent_column)
                utils.crash_code("column not found")
            tables_ready_for_join[child_table][child_column] = all_dataframes[child_table][child_column]
    # Drop rows from the dataframes that contain even a NaN value in any column
    for table in tables_ready_for_join:
        tables_ready_for_join[table] = tables_ready_for_join[table].dropna()

    tables_ready_for_join = prepend_column_names_with_table_name_and_add_index(tables_ready_for_join=tables_ready_for_join)

    return tables_ready_for_join