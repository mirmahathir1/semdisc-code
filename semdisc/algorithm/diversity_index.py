from custom_lib import db
from semdisc.lib import file_path_manager as fpm

from custom_lib import utils
from custom_lib import parallel
import numpy as np
import math
from collections import Counter
from custom_lib import console

def shannon_diversity_index_evenness(value_counts_dict):
    # Convert the dictionary of value counts to an array of counts
    counts = np.array(list(value_counts_dict.values()))
    
    total = counts.sum()

    # Proportions (p_i)
    proportions = counts / total

    # Shannon Diversity Index (H')
    shannon_index = -np.sum(proportions * np.log(proportions))

    # Evenness (E)
    species_count = len(counts)  # Number of unique species (S)
    evenness = shannon_index / np.log(species_count) if not species_count == 1 else 0
    return evenness

def simpsons_diversity_index_function(key_counts):
    N = sum(key_counts.values())

    if N <= 1:
        simpsons_diversity_index = 0  # No diversity if there are no elements or only 1 element
    else:
        # Simpson's diversity index
        simpson_index = sum((n * (n - 1)) for n in key_counts.values()) / (N * (N - 1))

        # Simpson's reciprocal index
        simpsons_diversity_index = 1 - simpson_index

    return simpsons_diversity_index

def entropy(counts):
    total = sum(counts.values())

    # Calculate probabilities
    probabilities = [count / total for count in counts.values()]

    # Calculate entropy
    entropy = -sum(p * math.log2(p) for p in probabilities)
    return entropy

def get_simhash_counts(simhashes):
    counts = Counter(simhashes)
    counts_dict = dict(counts)
    return counts_dict

def compute_semantic_diversity_for_one_column(argument, common_argument, common_argument_for_batch):
    table = argument['table']
    column = argument['column']
    simhashes = argument['simhashes']
    semantic_diversity = simpsons_diversity_index_function(get_simhash_counts(simhashes=simhashes))
    return {
        'table': table,
        'column': column,
        'semantic_diversity': semantic_diversity
    }

def compute_diversity_index(all_dataframes, all_simhashes):
    column_headers = db.get_all_column_headers(all_dataframes=all_dataframes)
    parallel_data = []
    for column_header in column_headers:
        parallel_data.append({
            'table': column_header['table'],
            'column': column_header['column'],
            'simhashes': all_simhashes[column_header['table']][column_header['column']]
        })

    results = parallel.execute(
        func=compute_semantic_diversity_for_one_column,
        argument_list=parallel_data
    )
    all_diversity = utils.nested_dict()
    for result in results:
        table = result['table']
        column = result['column']
        semantic_diversity = result['semantic_diversity']
        all_diversity[table][column] = semantic_diversity
    return all_diversity

def get_all_diversity_json_path():
    return f"{fpm.get_datalake_path()}/all_diversity.json"

def save_all_diversity(all_diversity):
    utils.file_dump(all_diversity, get_all_diversity_json_path())

def load_all_diversity():
    return utils.file_load(get_all_diversity_json_path())

if __name__ == "__main__":
    examples = [
        {'2':2, '3': 2},
        {'2':1, '3': 1, '4': 1, '5':1},
        {'2':10, '3': 10, '4': 10, '5':10},
        {'2':10, '3': 10, '4': 10, '5':10, '6': 10},
        {'2':1, '3': 1, '4': 1, '5':10}
    ]
    for example in examples:
        console.log(f"example: {example}")
        console.log("shannon")
        console.log(shannon_diversity_index_evenness(example))
        console.log("simpson")
        console.log(simpsons_diversity_index_function(example))

    fpm.startup()
    console.log()

    
