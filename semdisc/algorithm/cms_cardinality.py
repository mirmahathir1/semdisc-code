from custom_lib import count_min_sketch, utils
def calculate_cms_join_cardinality(tables_ready_for_join, join_order, self_join=False):

    max_memory_usage = 0

    start_time = utils.get_current_time()

    wasted_time = 0

    first_table_name = join_order[0]["parent"]
    first_column_name = join_order[0]["parent_column"]
    value_counts = tables_ready_for_join[first_table_name].groupby([f"{first_table_name}_{first_column_name}"]).size().to_dict()
    cms = count_min_sketch.build_count_min_sketch([])
    for value, count in value_counts.items():
        cms.add(value, count)
    del value_counts

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
        
        # new_value_counts = {}
        new_cms = count_min_sketch.build_count_min_sketch([])
        for (current_col_value, next_col_value), count in grouped_by_values.items():
            count_min_sketch.add_to_count_min_sketch(
                cms=new_cms,
                value=count_min_sketch.get_frequency(cms=cms, value=current_col_value),
                frequency=count
            )
                # new_value_counts[next_col_value] = new_value_counts.get(next_col_value, 0) + value_counts[current_col_value] * count

        # value_counts = new_value_counts
        cms = new_cms

    last_table_name = join_order[-1]["child"]
    last_column_name = join_order[-1]["child_column"]
    last_value_counts = tables_ready_for_join[last_table_name].groupby([f"{last_table_name}_{last_column_name}"]).size().to_dict()

    cardinality = 0
    for last_col_value, count in last_value_counts.items():
        if last_col_value in value_counts:
            # cardinality += value_counts[last_col_value] * count
            cardinality += count_min_sketch.get_frequency(cms=cms, value=last_col_value) * count

    del last_value_counts

    return {"cardinality": cardinality, "time": 0, 'max_memory_usage': 0}