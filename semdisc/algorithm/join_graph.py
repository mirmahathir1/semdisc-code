from custom_lib import utils
from custom_lib import parallel
from custom_lib import console
from custom_lib import db
from semdisc.lib import file_path_manager as fpm
from semdisc.lib import constants
from semdisc.lib import main_memory
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import random
def mySort(e):
  return e[2]

def generate_edges_for_single_pair_of_tables(argument, common_argument, common_argument_for_batch):
    table1 = argument['table1']
    table2 = argument['table2']
    minhash_jaccard = argument["minhash_jaccard"]
    minhash_simhash_jaccard = argument["minhash_simhash_jaccard"]
    semantic_type_similarities = argument["semantic_type_similarities"]
    column_name_dictionary = argument["column_name_dictionary"]
    diversity = argument["diversity"]
    is_numeric_of_columns = argument['is_numeric_of_columns']

    natural_join_enabled = common_argument["natural_join_enabled"]
    semantic_join_enabled = common_argument["semantic_join_enabled"]
    diversity_enabled = common_argument["diversity_enabled"]

    table1_columns = column_name_dictionary[table1]
    table2_columns = column_name_dictionary[table2]

    join_edge_threshold = common_argument['join_edge_threshold']
    diversity_multiplier_threshold = common_argument['diversity_multiplier_threshold']
    semantic_type_similarity_threshold = common_argument['semantic_type_similarity_threshold']

    all_possible_pairs = [(i,j) for i in table1_columns for j in table2_columns]

    edges_table1_to_table2 = []
    edges_table2_to_table1 = []

    for (key1, key2) in all_possible_pairs:
        equi_join_jaccard = minhash_jaccard[key1][key2]
        semantic_join_jaccard = minhash_simhash_jaccard[key1][key2]
        semantic_type_similarity = semantic_type_similarities[key1][key2]
        semantic_diversity = diversity[table1][key1]*diversity[table2][key2]
        is_column1_numeric = is_numeric_of_columns[table1][key1]
        is_column2_numeric = is_numeric_of_columns[table2][key2]

        edge_type = constants.NO_JOIN
        dist = 0
        quality = 0

        is_equi_join_eligible = equi_join_jaccard >= join_edge_threshold and (is_column1_numeric and is_column2_numeric)
        is_semantic_join_eligible = semantic_join_jaccard >= join_edge_threshold and (not is_column1_numeric and not is_column2_numeric)

        is_eligible_by_diversity = not diversity_enabled or semantic_diversity >= diversity_multiplier_threshold
        is_eligible_by_semantic_type_similarity = semantic_type_similarity >= semantic_type_similarity_threshold

        natural_join_quality =  equi_join_jaccard
        semantic_join_quality = semantic_join_jaccard

        # printbroken(is_eligible_by_semantic_type_similarity, is_eligible_by_diversity, is_equi_join_eligible, is_semantic_join_eligible)

        # if diversity_enabled:
        #     natural_join_quality = natural_join_quality * semantic_diversity
        #     semantic_join_quality = semantic_join_quality * semantic_diversity

        if is_eligible_by_semantic_type_similarity and is_eligible_by_diversity:
            if is_equi_join_eligible:
                dist = equi_join_jaccard
                quality = natural_join_quality
                edge_type = constants.NATURAL_JOIN

            elif is_semantic_join_eligible:
                dist = semantic_join_jaccard
                quality = semantic_join_quality
                edge_type = constants.SEMANTIC_HASH_JOIN

        if quality > 0:
            metadata = {'jaccard': dist, 'diversity': semantic_diversity, 'sem_type_sim': semantic_type_similarity}
            edges_table1_to_table2.append([key1, key2, quality, edge_type, metadata])
            edges_table2_to_table1.append([key2, key1, quality, edge_type, metadata])

    edges_table1_to_table2.sort(key=mySort, reverse=True)
    edges_table2_to_table1.sort(key=mySort, reverse=True)

    return {
        'table1': table1,
        'table2': table2,
        'table1_to_table2_edges': edges_table1_to_table2,
        'table2_to_table1_edges': edges_table2_to_table1
    }

def compute_join_graph(
        all_dataframes,
        minhash_jaccard,
        minhash_simhash_jaccard,
        semantic_type_similarities,
        column_name_dictionary,
        all_diversity,
        is_numeric_of_columns,
        natural_join_enabled,
        semantic_join_enabled,
        diversity_enabled,
        join_edge_threshold,
        diversity_multiplier_threshold,
        semantic_type_similarity_threshold
        ):
    console.log("---build_join_graph started")

    console.log(f"diversity_enabled: {diversity_enabled},\
                join_edge_threshold: {join_edge_threshold}, \
                    diversity_multiplier_threshold: {diversity_multiplier_threshold}, \
                        semantic_type_similarity_threshold: {semantic_type_similarity_threshold}")

    all_table_pairs = db.get_all_pairs_of_tables(all_dataframes=all_dataframes)

    parallel_data = []

    for (table1, table2) in all_table_pairs:
        parallel_data.append({
            'table1': table1,
            'table2': table2,
            'minhash_jaccard': minhash_jaccard[table1][table2],
            'minhash_simhash_jaccard': minhash_simhash_jaccard[table1][table2],
            'semantic_type_similarities': semantic_type_similarities[table1][table2],
            'column_name_dictionary': {table1: column_name_dictionary[table1], table2: column_name_dictionary[table2]},
            'diversity': {table1: all_diversity[table1], table2: all_diversity[table2]},
            'is_numeric_of_columns': {table1: is_numeric_of_columns[table1], table2: is_numeric_of_columns[table2]}
        })

    results = parallel.execute(
        func=generate_edges_for_single_pair_of_tables,
        argument_list=parallel_data,
        common_arguments={
            "natural_join_enabled": natural_join_enabled,
            "semantic_join_enabled": semantic_join_enabled,
            "diversity_enabled": diversity_enabled,
            'join_edge_threshold': join_edge_threshold,
            'diversity_multiplier_threshold': diversity_multiplier_threshold,
            'semantic_type_similarity_threshold': semantic_type_similarity_threshold
        },
    )

    join_graph = utils.nested_dict()
    for result in results:
        table1 = result['table1']
        table2 = result['table2']
        table1_to_table2_edges = result['table1_to_table2_edges']
        table2_to_table1_edges = result['table2_to_table1_edges']
        if len(table1_to_table2_edges) == 0:
            continue
        join_graph[table1][table2] = table1_to_table2_edges
        join_graph[table2][table1] = table2_to_table1_edges

    console.log("build join graph complete")
    return join_graph

def get_join_graph_json_path():
    return f"{fpm.get_datalake_path()}/join_graph.json"

def save_join_graph(join_graph_dictionary):
    utils.file_dump(join_graph_dictionary, get_join_graph_json_path())

def load_join_graph():
    return utils.file_load(get_join_graph_json_path())

def compile_all_neighbors_of_table(table):
    neighbors = {}
    for neighbor in get_neighbour_tables(table):
        neighbors[neighbor] = get_edges(table, neighbor)
    return neighbors

def get_graph_nodes(join_graph_dictionary):
    return list(join_graph_dictionary.keys())

def get_neighbour_tables(source_table):
    return list(main_memory.get_join_graph_from_memory()[source_table].keys())

def get_edges(join_graph_dictionary, source_table, destination_table):
    if not destination_table in join_graph_dictionary[source_table]:
        return []
    return join_graph_dictionary[source_table][destination_table]

def get_edges_from_join_graph(join_graph, source_table, destination_table):
    if not source_table in join_graph:
        return []
    if not destination_table in join_graph[source_table]:
        return []
    return join_graph[source_table][destination_table]

def get_all_node_pairs(join_graph_dictionary):
    return list(itertools.combinations(get_graph_nodes(join_graph_dictionary), 2))


def print_graph(join_graph_dict, png_path):

    G = nx.DiGraph()

    # Track added edges to avoid duplicates
    added_edges = set()

    for src, targets in join_graph_dict.items():
        for dst, edges in targets.items():
            if edges:
                edge = edges[0]  # Only consider the first element
                edge_label = f"{edge[0]}, {edge[1]}, {edge[2]}"
                edge_type = edge[3]  # semantichash or natural
                
                # Ensure unique edges
                if (src, dst) not in added_edges and (dst, src) not in added_edges:
                    G.add_edge(src, dst, label=edge_label, color="orange" if edge_type == constants.SEMANTIC_HASH_JOIN else "blue")
                    added_edges.add((src, dst))

    # Generate random positions for nodes
    pos = {node: (random.uniform(0, 1), random.uniform(0, 1)) for node in G.nodes()}
    edges = G.edges(data=True)
    edge_colors = [edge[2]['color'] for edge in edges]
    edge_labels = {(edge[0], edge[1]): edge[2]['label'] for edge in edges}

    plt.figure(figsize=(20, 16))
    nx.draw(G, pos, with_labels=True, edge_color=edge_colors, node_size=100, node_color="lightgray", font_size=4, font_weight="bold")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, bbox=dict(alpha=0))

    plt.savefig(png_path, format="png", dpi=300)
