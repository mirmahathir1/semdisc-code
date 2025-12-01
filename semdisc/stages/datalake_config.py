_config = {
    "drugcentral":{
        'simhash_size': 18,
        'join_edge_threshold': 0.5,
        'semantic_type_similarity_threshold': 0.5,
        'number_of_hashes_for_minhash': 128,
        'diversity_multiplier_threshold': 0.1,
        'simple_path_max_length': 5,
        'normalize_embeddings': True,
        'diversity_enabled': False
    }
}

def get_config(datalake_name):
    if datalake_name in _config:
        return _config[datalake_name]
    return _config[datalake_name.split('##')[0]]
