_config = {
    "complaints":{
        'simhash_size': 18,
        'join_edge_threshold': 0.5,
        'semantic_type_similarity_threshold': 0.5,
        'number_of_hashes_for_minhash': 128,
        'diversity_multiplier_threshold': 0.1,
        'simple_path_max_length': 5,
        'normalize_embeddings': True,
        'diversity_enabled': False
    },
    "drugcentral":{
        'simhash_size': 18,
        'join_edge_threshold': 0.5,
        'semantic_type_similarity_threshold': 0.5,
        'number_of_hashes_for_minhash': 128,
        'diversity_multiplier_threshold': 0.1,
        'simple_path_max_length': 5,
        'normalize_embeddings': True,
        'diversity_enabled': False
    },
    "mitdwh":{
        'simhash_size': 18,
        'join_edge_threshold': 0.5,
        'semantic_type_similarity_threshold': 0.5,
        'number_of_hashes_for_minhash': 128,
        'diversity_multiplier_threshold': 0.1,
        'simple_path_max_length': 5,
        'normalize_embeddings': True,
        'diversity_enabled': False
    },
    "fws":{
        'simhash_size': 18,
        'join_edge_threshold': 0.7,
        'semantic_type_similarity_threshold': 0.9,
        'number_of_hashes_for_minhash': 128,
        'diversity_multiplier_threshold': 0.1,
        'simple_path_max_length': 5,
        'normalize_embeddings': True,
        'diversity_enabled': False
    },
    "cdc":{
        'simhash_size': 18,
        'join_edge_threshold': 0.999,
        'semantic_type_similarity_threshold': 0.999,
        'number_of_hashes_for_minhash': 128,
        'diversity_multiplier_threshold': 0.975,
        'simple_path_max_length': 5,
        'normalize_embeddings': True,
        'diversity_enabled': True
    },
    "spider":{
        'simhash_size': 18,
        'join_edge_threshold': 0.7,
        'semantic_type_similarity_threshold': 0.5,
        'number_of_hashes_for_minhash': 128,
        'diversity_multiplier_threshold': 0.1,
        'simple_path_max_length': 5,
        'normalize_embeddings': True,
        'diversity_enabled': False
    },
    "opendata":{
        'simhash_size': 18,
        'join_edge_threshold': 0.99,
        'semantic_type_similarity_threshold': 0.99,
        'number_of_hashes_for_minhash': 128,
        'diversity_multiplier_threshold': 0.5,
        'simple_path_max_length': 5,
        'normalize_embeddings': True,
        'diversity_enabled': True
    },
    "opendata##opendata_CAN":{
        'simhash_size': 18,
        'join_edge_threshold': 0.8,
        'semantic_type_similarity_threshold': 0.9,
        'number_of_hashes_for_minhash': 128,
        'diversity_multiplier_threshold': 0.5,
        'simple_path_max_length': 5,
        'normalize_embeddings': True,
        'diversity_enabled': True
    },
    "opendata##opendata_SG":{
        'simhash_size': 18,
        'join_edge_threshold': 0.7,
        'semantic_type_similarity_threshold': 0.5,
        'number_of_hashes_for_minhash': 128,
        'diversity_multiplier_threshold': 0.5,
        'simple_path_max_length': 5,
        'normalize_embeddings': True,
        'diversity_enabled': True
    },
}

def get_config(datalake_name):
    if datalake_name in _config:
        return _config[datalake_name]
    return _config[datalake_name.split('##')[0]]
