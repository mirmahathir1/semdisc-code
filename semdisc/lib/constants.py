config = {
    # "NATURAL_JOIN_EDGE_WEIGHT_THRESHOLD" : 0.3,
    # "SEMANTIC_HASH_EDGE_WEIGHT_THRESHOLD": 0.4,

    # "faiss_range_search_radius": 0.8, # increase this to get lower cardinality on semantic edges, decrease this to get more rows in semantic joins
    # "semantic_type_similarity_threshold_for_natural_join": 0.8,
    # "semantic_type_similarity_threshold_for_semantic_join": 0.8,

    # "simple_path_max_length": 7,
    # "simhash_bit_count": 6,

    # "diversity_multiplier_threshold": 0,
    # "NATURAL_JOIN_EDGE_WEIGHT_THRESHOLD_for_diversity": 0.6,
    # "SEMANTIC_HASH_EDGE_WEIGHT_THRESHOLD_for_diversity": 0.6
}

LLM_JOIN = "llm"
EMBEDDING_JOIN = "embedding"
NATURAL_JOIN = "natural"
SEMANTIC_HASH_JOIN = "semantichash"
NO_JOIN = "none"

SEMANTIC_JOIN_TYPE_EMBEDDING = "SEMANTIC_JOIN_TYPE_EMBEDDING"
SEMANTIC_JOIN_TYPE_SIMHASH = "SEMANTIC_JOIN_TYPE_SIMHASH"

MAX_VALUE_FOR_EDGE = 999999



