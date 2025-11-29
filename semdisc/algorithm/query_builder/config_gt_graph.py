_config = {
    'complaints':{
        "cosine_similarity_threshold": 0.9,
        "semantic_similarity_threshold": 0.50,
        "joinability_threshold": 0.8,
        "diversity_threshold": 0,
        "induced_path_flag": False
    },
    'drugcentral':{
        "cosine_similarity_threshold": 0.9,
        "semantic_similarity_threshold": 0.50,
        "joinability_threshold": 0.8,
        "diversity_threshold": 0,
        "induced_path_flag": False
    },
    'mitdwh': {
        "cosine_similarity_threshold": 0.9,
        "semantic_similarity_threshold": 0.5,
        "joinability_threshold": 0.8,
        "diversity_threshold": 0,
        "induced_path_flag": True
    },
    'fws': {
        "cosine_similarity_threshold": 0.9,
        "semantic_similarity_threshold": 0.9,
        "joinability_threshold": 0.9,
        "diversity_threshold": 0,
        "induced_path_flag": True
    },
    'cdc': {
        "cosine_similarity_threshold": 0.999,
        "semantic_similarity_threshold": 0.999,
        "joinability_threshold": 0.9,
        "diversity_threshold": 0.975,
        "induced_path_flag": False 
    },
    'opendata': {
        "cosine_similarity_threshold": 0.999,
        "semantic_similarity_threshold": 0.999,
        "joinability_threshold": 0.9,
        "diversity_threshold": 0.1,
        "induced_path_flag": False 
    },
    'spider': {
        "cosine_similarity_threshold": 0.9,
        "semantic_similarity_threshold": 0.5,
        "joinability_threshold": 0.8,
        "diversity_threshold": 0,
        "induced_path_flag": True
    }
}

def get_config(datalake_name):
    return _config[datalake_name.split('##')[0]]

_config_graph_approximation = {
    'complaints':{
        "cosine_similarity_threshold": 0.9,
        "semantic_similarity_threshold": 0.50,
        "joinability_threshold": 0.8,
        "diversity_threshold": 0,
        "induced_path_flag": False
    },
    'drugcentral':{
        "cosine_similarity_threshold": 0.9,
        "semantic_similarity_threshold": 0.50,
        "joinability_threshold": 0.8,
        "diversity_threshold": 0,
        "induced_path_flag": False
    },
    'mitdwh': {
        "cosine_similarity_threshold": 0.9,
        "semantic_similarity_threshold": 0.5,
        "joinability_threshold": 0.8,
        "diversity_threshold": 0,
        "induced_path_flag": True
    },
    'fws': {
        "cosine_similarity_threshold": 0.9,
        "semantic_similarity_threshold": 0.50,
        "joinability_threshold": 0.8,
        "diversity_threshold": 0,
        "induced_path_flag": True
    },
    'cdc': {
        "cosine_similarity_threshold": 0.9,
        "semantic_similarity_threshold": 0.50,
        "joinability_threshold": 0.8,
        "diversity_threshold": 0,
        "induced_path_flag": True
    },
    'opendata': {
        "cosine_similarity_threshold": 0.9,
        "semantic_similarity_threshold": 0.50,
        "joinability_threshold": 0.8,
        "diversity_threshold": 0,
        "induced_path_flag": True
    },
    'spider': {
        "cosine_similarity_threshold": 0.9,
        "semantic_similarity_threshold": 0.50,
        "joinability_threshold": 0.8,
        "diversity_threshold": 0,
        "induced_path_flag": True
    },
}

def get_config_graph_approximation(datalake_name):
    return _config_graph_approximation[datalake_name.split('##')[0]]
