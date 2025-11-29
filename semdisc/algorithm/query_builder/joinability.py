from semdisc.lib import constants
from custom_lib import utils
import numpy as np
import faiss

def normalized_euclidean_similarity(embedding1, embedding2):
    """
    Normalize two embeddings and compute similarity as:
    1 - (Euclidean distance / max possible distance)

    For unit-norm vectors, max Euclidean distance is 2 (when they are opposite).

    Args:
        embedding1: First embedding (list or numpy array)
        embedding2: Second embedding (list or numpy array)

    Returns:
        float: Similarity score in [0, 1]
    """
    # Convert to numpy arrays
    a = np.array(embedding1, dtype=np.float64)
    b = np.array(embedding2, dtype=np.float64)

    # Normalize both vectors to unit norm
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    # Compute Euclidean distance
    distance = np.linalg.norm(a - b)

    # Max distance between any two unit vectors is 2
    max_distance = 2.0

    # Compute similarity
    similarity = 1 - (distance / max_distance)

    return float(similarity)


def cosine_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two unnormalized embeddings.
    
    Args:
        embedding1: First embedding (numpy array or list)
        embedding2: Second embedding (numpy array or list)
        
    Returns:
        float: Cosine similarity between -1 and 1
    """
    # Convert to numpy arrays if they aren't already
    a = np.array(embedding1)
    b = np.array(embedding2)
    
    # Compute dot product
    dot_product = np.dot(a, b)
    
    # Compute L2 norms
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # Compute cosine similarity
    similarity = dot_product / (norm_a * norm_b)
    
    return float(similarity)

def get_semantic_matches_by_embeddings_and_euclidean_distance(
        column1_values,
        column2_values,
        column1_embeddings,
        column2_embeddings,
        euclidean_distance
        ):
    """
    Find semantic matches between two columns based on their embeddings.
    
    Args:
        column1_values: pandas Series of values from the first column
        column2_values: pandas Series of values from the second column
        column1_embeddings: numpy array of normalized embeddings for column1 values
        column2_embeddings: numpy array of normalized embeddings for column2 values
        euclidean_distance: maximum distance threshold for considering a match
        
    Returns:
        tuple: (matches_1_to_2, matches_2_to_1) where each is a list of matched quadruples:
              - value from first column
              - value from second column
              - index in first column
              - index in second column
              where distance is less than the threshold and neither value is NaN
    """
    matches_1_to_2 = []
    matches_2_to_1 = []
    
    # Compute all pairwise dot products (cosine similarities since embeddings are normalized)
    dot_products = np.dot(column1_embeddings, column2_embeddings.T)
    
    # Convert to Euclidean distances squared
    # For normalized vectors, ||u-v||² = 2 - 2(u·v)
    euclidean_distances_sq = 2 - 2 * dot_products
    
    # Find indices where distance is below threshold
    rows, cols = np.where(euclidean_distances_sq <= (euclidean_distance ** 2))
    
    # Create quadruples of matching values with their indices, excluding NaN values
    for row, col in zip(rows, cols):
        val1 = column1_values.iloc[row]
        val2 = column2_values.iloc[col]
        
        # Skip if either value is NaN
        if pd.isna(val1) or pd.isna(val2):
            continue
            
        matches_1_to_2.append([
            val1,  # value from first column
            val2,  # value from second column
            int(row),  # index in first column
            int(col)   # index in second column
        ])
        matches_2_to_1.append([
            val2,  # value from second column
            val1,  # value from first column
            int(col),  # index in second column
            int(row)   # index in first column
        ])
    
    return matches_1_to_2, matches_2_to_1

import pandas as pd

def compute_exact_jaccard(series1, series2):
    """
    Compute the Jaccard similarity between two pandas Series, ignoring NaN values.
    
    The Jaccard similarity is defined as the size of the intersection divided by the size of the union
    of the two sets of non-NaN values.
    
    Parameters:
    series1 (pd.Series): First input series
    series2 (pd.Series): Second input series (must be same length as series1)
    
    Returns:
    float: Jaccard similarity score between 0 and 1
    """
    
    # Create sets of non-NaN values for each series
    set1 = set(series1.dropna().unique())
    set2 = set(series2.dropna().unique())
    
    # Calculate intersection and union
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    # Handle case where both sets are empty
    if len(union) == 0:
        return 1.0  # By convention, two empty sets are considered identical
    
    return {
        "jaccard": len(intersection) / len(union)
    }


def compute_simhash_exact_jaccard_by_equi_join_or_semantic_join(
        column1_values,
        column2_values,
        column1_simhashes,
        column2_simhashes,
        column1_is_numeric,
        column2_is_numeric,
):
    if column1_is_numeric and column2_is_numeric:
        jaccard_info = compute_exact_jaccard(column1_values, column2_values)
        jaccard_info['type'] = constants.NATURAL_JOIN
        return jaccard_info
    elif not column1_is_numeric and not column2_is_numeric:
        jaccard_info = compute_exact_jaccard(
            column1_simhashes, column2_simhashes
        )
        jaccard_info['type'] = constants.SEMANTIC_HASH_JOIN
        return jaccard_info
    utils.crash_code("Two different types of columns are not supported yet")


def compute_joinability_of_unique_values_by_equi_join_or_semantic_join(
        column1_values,
        column2_values,
        column1_embeddings,
        column2_embeddings,
        cosine_similarity_threshold,
        column1_is_numeric,
        column2_is_numeric,
):
    if column1_is_numeric and column2_is_numeric:
        joinability_info = compute_equi_joinability_with_unique_values(column1_values, column2_values)
        joinability_info['type'] = constants.NATURAL_JOIN
        return joinability_info
    elif not column1_is_numeric and not column2_is_numeric:
        joinability_info = compute_joinability_by_embeddings_and_cosine_similarity_with_unique_values(
            column1_values, column2_values,
            column1_embeddings, column2_embeddings,
            cosine_similarity_threshold
        )
        joinability_info['type'] = constants.SEMANTIC_HASH_JOIN
        return joinability_info

    utils.crash_code("Two different types of columns are not supported yet")

def compute_equi_joinability_with_unique_values(column1_values: pd.Series,
                             column2_values: pd.Series) -> dict:
    """
    Calculate *equi-joinability* between two columns **using only the distinct
    (unique) non-NULL values** from each column.

    Joinability from column A → column B is the fraction of *unique* values in A
    that appear at least once in B.  The overall score is the mean of the two
    directed scores.

    Parameters
    ----------
    column1_values, column2_values : pandas.Series
        Columns to compare.  NaNs are ignored when building the sets.

    Returns
    -------
    dict
        {
            "joinability_1_to_2": float,
            "joinability_2_to_1": float,
            "overall_joinability": float
        }
    """
    # ── Distinct non-NULL sets ───────────────────────────────────────────
    col1_set = set(column1_values.dropna().unique())
    col2_set = set(column2_values.dropna().unique())

    # If either side has no distinct values, joinability is zero
    if not col1_set or not col2_set:
        return {
            "joinability_1_to_2": 0.0,
            "joinability_2_to_1": 0.0,
            "overall_joinability": 0.0,
        }

    # Intersection is the same for both directions
    intersection_size = len(col1_set & col2_set)

    # ── Directed scores ──────────────────────────────────────────────────
    joinability_1_to_2 = intersection_size / len(col1_set)
    joinability_2_to_1 = intersection_size / len(col2_set)

    # ── Overall score ────────────────────────────────────────────────────
    overall_joinability = (joinability_1_to_2 + joinability_2_to_1) / 2

    return {
        "joinability_1_to_2": float(joinability_1_to_2),
        "joinability_2_to_1": float(joinability_2_to_1),
        "overall_joinability": float(overall_joinability),
    }

def _rows_with_match(src_emb: np.ndarray,
                     trg_emb: np.ndarray,
                     sim_thresh: float) -> set[int]:
    """
    For every embedding in `src_emb`, find whether it has at least
    one neighbour in `trg_emb` with cosine similarity ≥ `sim_thresh`.

    Returns the set of row indices in `src_emb` that do have a match.
    """
    dim = src_emb.shape[1]

    # Build a flat inner-product index on the target side
    index = faiss.IndexFlatIP(dim)
    index.add(trg_emb)

    # Range search ⇒ all pairs whose inner product ≥ threshold
    # (because vectors are unit-norm, inner product == cosine similarity)
    lims, _D, _I = index.range_search(src_emb, sim_thresh)

    # If lims[i] < lims[i+1], query-vector i has ≥1 hit
    return {i for i in range(src_emb.shape[0]) if lims[i] < lims[i + 1]}

def compute_joinability_by_embeddings_and_cosine_similarity_with_unique_values_faiss(
    column1_values,
    column2_values,
    column1_embeddings,
    column2_embeddings,
    cosine_similarity_threshold: float,
):
    """
    Identical I/O contract to the original function, but accelerated with FAISS.
    """

    # ── 1. Prepare data ──────────────────────────────────────────────────────
    vals1 = pd.Series(column1_values).dropna()
    vals2 = pd.Series(column2_values).dropna()
    emb1 = np.asarray(column1_embeddings, dtype=np.float32)[vals1.index]
    emb2 = np.asarray(column2_embeddings, dtype=np.float32)[vals2.index]

    # Normalise so that inner product == cosine similarity
    faiss.normalize_L2(emb1)
    faiss.normalize_L2(emb2)

    # Unique value sets (needed for the denominators)
    col1_set = set(vals1.unique())
    col2_set = set(vals2.unique())

    # ── 2. Find which rows have at least one semantic match ─────────────────
    rows1_with_match = _rows_with_match(emb1, emb2, cosine_similarity_threshold)
    rows2_with_match = _rows_with_match(emb2, emb1, cosine_similarity_threshold)

    col1_matched_values = set(vals1.iloc[list(rows1_with_match)])
    col2_matched_values = set(vals2.iloc[list(rows2_with_match)])

    # ── 3. Directed & overall joinability ───────────────────────────────────
    joinability_1_to_2 = (
        len(col1_matched_values) / len(col1_set) if col1_set else 0.0
    )
    joinability_2_to_1 = (
        len(col2_matched_values) / len(col2_set) if col2_set else 0.0
    )
    overall_joinability = 0.5 * (joinability_1_to_2 + joinability_2_to_1)

    return {
        "joinability_1_to_2": joinability_1_to_2,
        "joinability_2_to_1": joinability_2_to_1,
        "overall_joinability": overall_joinability,
    }

def compute_joinability_by_embeddings_and_cosine_similarity_with_unique_values(
        column1_values,
        column2_values,
        column1_embeddings,
        column2_embeddings,
        cosine_similarity_threshold: float,
):
    """
    Calculate semantic *joinability* between two columns.

    Joinability from column A → column B is the fraction of rows in A that
    can be matched to **at least one** row in B within the cosine‑similarity
    threshold.  Overall joinability is the mean of the two directed scores.

    Returns
    -------
    dict
        {
            "joinability_1_to_2": float,
            "joinability_2_to_1": float,
            "overall_joinability": float
        }
    """

    col1_set = column1_values.dropna().unique()
    col2_set = column2_values.dropna().unique()

    col1_set_embeddings = column1_embeddings[col1_set.index]
    col2_set_embeddings = column2_embeddings[col2_set.index]

    # --- find all pairwise matches -----------------------------------------
    m12, m21 = get_semantic_matches_by_embeddings_and_cosine_similarity(
        col1_set, col2_set,
        col1_set_embeddings, col2_set_embeddings,
        cosine_similarity_threshold
    )

    # --- map each match list to the set of source‑side indices -------------
    idx1_with_match = [idx1 for _, _, idx1, _ in m12]
    idx2_with_match = [idx2 for _, _, idx2, _ in m21]

    col1_matched_value_set = set(column1_values[idx1_with_match])
    col2_matched_value_set = set(column2_values[idx2_with_match])

    # printbroken(col1_matched_value_set)
    # printbroken(col2_matched_value_set)

    joinability_1_to_2 = len(col1_matched_value_set) / len(col1_set) if not len(col1_set) == 0 else 0.0
    joinability_2_to_1 = len(col2_matched_value_set) / len(col2_set) if not len(col2_set) == 0 else 0.0

    # --- overall (symmetric) score ----------------------------------------
    overall_joinability = 0.5 * (joinability_1_to_2 + joinability_2_to_1)

    return {
        "joinability_1_to_2": joinability_1_to_2,
        "joinability_2_to_1": joinability_2_to_1,
        "overall_joinability": overall_joinability,
    }


import numpy as np
import pandas as pd

def get_semantic_matches_by_embeddings_and_cosine_similarity(
        column1_values,
        column2_values,
        column1_embeddings,
        column2_embeddings,
        cosine_similarity_threshold,
):
    """
    Find semantic matches between two columns using **cosine distance**.

    Args:
        column1_values (pd.Series): values from the first column
        column2_values (pd.Series): values from the second column
        column1_embeddings (np.ndarray): embeddings for column 1 (shape: [n₁, d])
        column2_embeddings (np.ndarray): embeddings for column 2 (shape: [n₂, d])
        cosine_similarity_threshold (float): minimum cosine-similarity allowed for a match

    Returns
        tuple[list, list]:
            - matches_1_to_2: [val₁, val₂, idx₁, idx₂] for all pairs within the threshold  
            - matches_2_to_1: symmetric list ([val₂, val₁, idx₂, idx₁])
    """
    matches_1_to_2, matches_2_to_1 = [], []

    # Normalize the embeddings
    norm1 = np.linalg.norm(column1_embeddings, axis=1, keepdims=True)
    norm2 = np.linalg.norm(column2_embeddings, axis=1, keepdims=True)
    
    # Avoid division by zero for zero vectors
    norm1 = np.where(norm1 == 0, 1, norm1)
    norm2 = np.where(norm2 == 0, 1, norm2)
    
    norm_embeddings1 = column1_embeddings / norm1
    norm_embeddings2 = column2_embeddings / norm2

    # Compute cosine similarity matrix
    cos_sim = norm_embeddings1 @ norm_embeddings2.T

    # Find indices where distance ≤ threshold
    rows, cols = np.where(cos_sim >= cosine_similarity_threshold)

    for r, c in zip(rows, cols):
        v1, v2 = column1_values.iloc[r], column2_values.iloc[c]
        if pd.isna(v1) or pd.isna(v2):
            continue

        matches_1_to_2.append([v1, v2, int(r), int(c)])
        matches_2_to_1.append([v2, v1, int(c), int(r)])

    return matches_1_to_2, matches_2_to_1


def get_the_closest_semantically_similar_values_by_euclidean_distance(
        column1_values,
        column2_values,
        column1_embeddings,
        column2_embeddings,
        K,
        allow_exacts=True
        ):
    """
    Find the top K closest semantic matches between two columns based on their embeddings,
    ensuring no duplicate (value1, value2) pairs are returned, and optionally filtering exact matches.

    Args:
        column1_values: pandas Series of values from the first column
        column2_values: pandas Series of values from the second column
        column1_embeddings: numpy array of normalized embeddings for column1 values
        column2_embeddings: numpy array of normalized embeddings for column2 values
        K: number of closest unique pairs to return
        allow_exacts: If True, allow pairs where value1 == value2. If False, exclude exact matches.

    Returns:
        list: List of matched pairs with their distances, each element is [value1, value2, distance]
              sorted by ascending distance (closest first), with no duplicate pairs.
    """
    # Compute all pairwise dot products (cosine similarities since embeddings are normalized)
    dot_products = np.dot(column1_embeddings, column2_embeddings.T)

    # Convert to Euclidean distances (for normalized vectors, ||u-v|| = sqrt(2 - 2(u·v)))
    euclidean_distances = np.sqrt(2 - 2 * dot_products)

    # Flatten and sort all possible pairs by distance
    row_indices, col_indices = np.unravel_index(
        np.argsort(euclidean_distances.ravel()),
        euclidean_distances.shape
    )

    # Iterate through sorted pairs and collect unique matches until we reach K
    top_pairs = []
    seen_pairs = set()  # Track (value1, value2) pairs to avoid duplicates

    for row, col in zip(row_indices, col_indices):
        value1 = column1_values.iloc[row]
        value2 = column2_values.iloc[col]
        pair_key = (value1, value2)  # Use a tuple to check for duplicates

        # Skip if pair is a duplicate
        if pair_key in seen_pairs:
            continue

        # Skip if exact matches are not allowed and values are equal
        if not allow_exacts and value1 == value2:
            continue

        # Add valid pair to results
        distance = euclidean_distances[row, col]
        top_pairs.append([value1, value2, distance])
        seen_pairs.add(pair_key)

        if len(top_pairs) >= K:
            break  # Stop once we have K unique pairs

    return top_pairs

def get_the_closest_semantically_similar_values_by_cosine_similarity(
        column1_values,
        column2_values,
        column1_embeddings,
        column2_embeddings,
        K,
        allow_exacts=True
        ):
    """
    Find the top K closest semantic matches between two columns based on their embeddings,
    ensuring no duplicate (value1, value2) pairs are returned, and optionally filtering exact matches.

    Args:
        column1_values: pandas Series of values from the first column
        column2_values: pandas Series of values from the second column
        column1_embeddings: numpy array of normalized embeddings for column1 values
        column2_embeddings: numpy array of normalized embeddings for column2 values
        K: number of closest unique pairs to return
        allow_exacts: If True, allow pairs where value1 == value2. If False, exclude exact matches.

    Returns:
        list: List of matched pairs with their distances, each element is [value1, value2, distance]
              sorted by ascending distance (closest first), with no duplicate pairs.
    """
    # Compute all pairwise cosine similarities (dot products since embeddings are normalized)
    cosine_similarities = np.dot(column1_embeddings, column2_embeddings.T)

    row_indices, col_indices = np.unravel_index(
        np.argsort(-cosine_similarities.ravel()),  # Negate values to get descending order
        cosine_similarities.shape
    )

    # Iterate through sorted pairs and collect unique matches until we reach K
    top_pairs = []
    seen_pairs = set()  # Track (value1, value2) pairs to avoid duplicates

    for row, col in zip(row_indices, col_indices):
        value1 = column1_values.iloc[row]
        value2 = column2_values.iloc[col]
        pair_key = (value1, value2)  # Use a tuple to check for duplicates

        # Skip if pair is a duplicate
        if pair_key in seen_pairs:
            continue

        # Skip if exact matches are not allowed and values are equal
        if not allow_exacts and value1 == value2:
            continue

        # Add valid pair to results
        distance = cosine_similarities[row, col]
        top_pairs.append([value1, value2, distance])
        seen_pairs.add(pair_key)

        if len(top_pairs) >= K:
            break  # Stop once we have K unique pairs

    return top_pairs

def filter_semantic_matches_by_discarding_equi_joins_and_duplicate_rows(matches):
    """
    Filters semantic matches by:
    1. Removing duplicate value pairs (keeping only the first occurrence)
    2. Removing exact matches (where column1 value == column2 value)

    Args:
        matches: List of matches from get_semantic_matches_by_embeddings_and_euclidean_distance, where each match is:
                [column1_value, column2_value, column1_index, column2_index]

    Returns:
        list: Filtered list of matches without duplicates or exact matches
    """
    seen_pairs = set()
    filtered_matches = []

    for match in matches:
        # printbroken(match)
        # printbroken("match")
        val1, val2, idx1, idx2 = match
        pair = (val1, val2)  # Use a tuple for hashability in the set

        # Skip if this is an exact match (values are identical)
        if val1 == val2:
            continue

        # Skip if we've already seen this (val1, val2) pair before
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            filtered_matches.append(match)

    return filtered_matches