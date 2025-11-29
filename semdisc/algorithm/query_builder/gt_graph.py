from custom_lib import utils
from custom_lib import parallel
from custom_lib import console
from custom_lib import db
from semdisc.lib import file_path_manager as fpm
from semdisc.lib import constants
from semdisc.lib import main_memory
from semdisc.algorithm import semantic_type_extractor
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from semdisc.lib import table_embeddings
from semdisc.algorithm import semantic_hash_index
from custom_lib import db
from semdisc.algorithm.query_builder import joinability
from collections import defaultdict
from typing import Dict, List, Any
from semdisc.algorithm import join_graph

import numpy as np
import faiss

import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
import os

faiss.omp_set_num_threads(int(os.environ.get("FAISS_THREADS", "1")))

# def extract_for_single_table(argument, common_argument, common_argument_for_batch):
#     tbl, df = argument


import numpy as np
import faiss

# ----------------------------------------------------------------------
# 1. Worker that handles ONE table
# ----------------------------------------------------------------------
def _prepare_single_table(argument, _common_args, _scratch):
    """
    argument == (tbl_name, dataframe, embedding_dataframe)

    Returns
    -------
    tuple
        (tbl_name, dict[column_name → (index | None,
                                       emb_arr | None,
                                       value_set)])
    """
    tbl, df, emb_df = argument
    tbl_dict: dict[str, tuple] = {}

    for col in df.columns:
        non_nan_mask = df[col].notna()
        if not non_nan_mask.any():
            continue          # column is all-NaN → skip

        # ---- unique non-NaN values ------------------------------------
        unique_rows = df[non_nan_mask].drop_duplicates(subset=[col]).index
        value_set   = set(df.loc[unique_rows, col])

        # ---- corresponding embeddings (may be empty) -----------------
        emb_list = emb_df[col][unique_rows]
        if len(emb_list):                      # text / categorical cols
            emb_arr = np.ascontiguousarray(np.stack(emb_list)
                                           .astype("float32"))
            index   = faiss.IndexFlatIP(emb_arr.shape[1])
            index.add(emb_arr)
        else:                                 # numeric-only cols
            index, emb_arr = None, None

        tbl_dict[col] = (index, emb_arr, value_set)

    return tbl, tbl_dict            # caller will drop empty tbl_dicts


# ----------------------------------------------------------------------
# 2. Public API – parallel version
# ----------------------------------------------------------------------
def build_prepared_embeddings(
    all_dataframes: dict[str, pd.DataFrame],
    all_embeddings: dict[str, pd.DataFrame],
) -> dict[str, dict[str, tuple[faiss.IndexFlatIP, np.ndarray, set]]]:
    """
    prepared_embeddings[table][column] → (faiss.IndexFlatIP | None,
                                          np.ndarray        | None,
                                          set)               # unique non-NaN values
    """
    # --- pack everything each worker needs into its argument -----------
    argument_list = [
        (tbl, df, all_embeddings[tbl])
        for tbl, df in all_dataframes.items()
    ]

    # --- fire off the parallel workers --------------------------------
    # common_arguments is empty because each worker already has
    # everything it needs inside its own argument tuple.
    results = parallel.execute(
        _prepare_single_table,
        argument_list,
        common_arguments={},        # read-only; nothing needed
    )

    # --- gather results into the expected nested dict ------------------
    prepared: dict[str, dict[str, tuple]] = {
        tbl: tbl_dict for tbl, tbl_dict in results if tbl_dict
    }
    return prepared


def calculate_exact_joinability(set1: set, set2: set) -> tuple[float, float]:
    """
    Directed exact-match joinability based on unique, non-NaN values.
    Returns (joinability_1_to_2, joinability_2_to_1).
    """
    if not set1 or not set2:
        return 0.0, 0.0

    inter_size = len(set1 & set2)
    return inter_size / len(set1), inter_size / len(set2)

def calculate_joinability(index_col: faiss.Index,
                   emb_arr_col: np.ndarray,
                   threshold: float) -> int:
    """
    Count how many vectors in *column 2* have at least one match in *column 1*
    with cosine similarity > `threshold`.

    Parameters
    ----------
    index_col1 : faiss.Index
        FAISS index built on **unit-normalized** embeddings from column 1
        (e.g. `faiss.IndexFlatIP`).
    emb_arr_col2 : np.ndarray, shape (n2, d), dtype float32
        **Unit-normalized** embeddings from column 2.
    threshold : float
        Cosine-similarity cut-off (0 < threshold ≤ 1).

    Returns
    -------
    int
        Number of column-2 vectors that have ≥ 1 neighbour in column 1
        with similarity above the threshold.
    """
    # Empty inputs → no joinable elements
    if index_col.ntotal == 0 or emb_arr_col.size == 0:
        return 0

    # Make sure queries are contiguous float32
    # queries = np.ascontiguousarray(emb_arr_col2, dtype="float32")

    # 1-NN search is enough: if the best match is ≤ threshold, no other is higher
    # sims, _ = index_col1.search(queries, 1)   # shape (n2, 1)
    sims, _ = index_col.search(emb_arr_col, 1)   # shape (n2, 1)
    # Count queries whose best similarity exceeds the threshold
    return int((sims.ravel() > threshold).sum())/len(emb_arr_col)

def build_join_graph_dict(
    joinability_dict: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]],
    threshold: float,
) -> Dict[str, Dict[str, List[List[Any]]]]:
    """
    Convert a four-level ``joinability_dict``

        joinability_dict[table1][table2][col1][col2] = {
            "joinability": float,
            "type": <constants.NATURAL_JOIN | constants.SEMANTIC_HASH_JOIN>,
            ...
        }

    into a three-level ``join_graph_dict``

        join_graph_dict[table1][table2] = [
            [col1, col2, joinability, type],   # sorted ↓ by joinability
            ...
        ]

    Parameters
    ----------
    joinability_dict : dict
        Output from the previous pipeline (symmetric in table/column order).
    threshold : float
        Minimum joinability (0 ≤ threshold ≤ 1).  Edges below this value are
        discarded.

    Returns
    -------
    dict
        Nested dictionary of *sorted* edge lists.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("`threshold` must be between 0 and 1, inclusive.")

    join_graph_dict: Dict[str, Dict[str, List[List[Any]]]] = defaultdict(dict)

    for t1, t1_map in tqdm(joinability_dict.items()):
        for t2, t2_map in t1_map.items():
            # collect all candidate edges between (t1, t2)
            edges: List[List[Any]] = []
            for c1, c1_map in t2_map.items():
                for c2, info in c1_map.items():
                    j = info.get("joinability", 0.0)
                    if j >= threshold:
                        edges.append([c1, c2, j, info.get("type")])

            # keep only if there is at least one edge ≥ threshold
            if edges:
                # highest-joinability first
                edges.sort(key=lambda e: e[2], reverse=True)
                join_graph_dict[t1][t2] = edges

    return join_graph_dict

# --------------------------------------------------------------------
# 1. worker – processes ONE unordered table pair
# --------------------------------------------------------------------
def _joinability_for_pair(argument, common_args, _scratch):
    """
    argument
      = (t1, t2,
         df1, df2,
         prep1, prep2,
         num1_map, num2_map,
         sem_pair)                        # sem_pair[c1][c2]

    returns
      list[(tblA, tblB, colA, colB, info_dict)]
    """
    (t1, t2,
     df1, df2,
     prep1, prep2,
     num1_map, num2_map,
     sem_pair, 
     diversity1, diversity2) = argument

    cos_sim  = common_args["cosine_similarity"]
    sem_thr  = common_args["semantic_similarity_threshold"]
    diversity_threshold = common_args['diversity_threshold']

    out = []

    for c1 in df1.columns:
        idx1, emb1, set1 = prep1[c1]
        num1             = num1_map[c1]

        for c2 in df2.columns:
            idx2, emb2, set2 = prep2[c2]
            num2             = num2_map[c2]

            if diversity1[c1]*diversity2[c2] < diversity_threshold:
                info = {}

            # ---- semantic-type gate ---------------------------------
            elif sem_pair[c1][c2] < sem_thr:
                info = {}

            # ---- ① numeric ⇄ numeric --------------------------------
            elif num1 and num2:
                j12, j21 = calculate_exact_joinability(set1, set2)
                info = {"type": constants.NATURAL_JOIN,
                        "joinability": 0.5 * (j12 + j21)}

            # ---- ② mixed numeric / text ------------------------------
            elif num1 ^ num2:
                info = {}

            # ---- ③ text ⇄ text --------------------------------------
            else:
                j12 = calculate_joinability(idx2, emb1, cos_sim)
                j21 = calculate_joinability(idx1, emb2, cos_sim)
                info = {"type": constants.SEMANTIC_HASH_JOIN,
                        "joinability": 0.5 * (j12 + j21)}

            # symmetric write-back
            out.append((t1, t2, c1, c2, info))
            out.append((t2, t1, c2, c1, info))
    return out


# --------------------------------------------------------------------
# 2. public helper – constructs argument list & launches workers
# --------------------------------------------------------------------
def build_joinability_dict(
    all_dataframes,
    prepared_embeddings,
    is_numeric_of_columns,
    semantic_type_similarities,
    all_diversity,

    cosine_similarity,
    semantic_similarity_threshold,
    diversity_threshold
    ):
    """
    joinability_dict[t1][t2][c1][c2] = info
    """
    # --- one argument tuple per unordered table pair -----------------
    argument_list = []
    for t1, t2 in db.get_all_pairs_of_tables(all_dataframes):
        argument_list.append(
            (t1, t2,
             all_dataframes[t1], all_dataframes[t2],
             prepared_embeddings[t1], prepared_embeddings[t2],
             is_numeric_of_columns[t1], is_numeric_of_columns[t2],
             semantic_type_similarities[t1][t2],
             all_diversity[t1], all_diversity[t2]
             )   # pair-specific slice
        )

    # --- tiny, truly common payload ----------------------------------
    common_arguments = {
        "cosine_similarity":            cosine_similarity,
        "semantic_similarity_threshold": semantic_similarity_threshold,
        "diversity_threshold": diversity_threshold
    }

    # --- run the workers ---------------------------------------------
    partial_results = parallel.execute(
        _joinability_for_pair,
        argument_list,
        common_arguments=common_arguments,
        parallel_enabled=True
    )

    # --- merge flat tuples into the nested dict ----------------------
    joinability_dict = utils.nested_dict()
    for pair_list in partial_results:
        for tA, tB, colA, colB, info in pair_list:
            joinability_dict[tA][tB][colA][colB] = info

    return joinability_dict


def get_join_graph_gt_json_path():
    return f"{fpm.get_datalake_path()}/join_graph_gt.json"

def save_join_graph_gt(join_graph_gt):
    utils.file_dump(join_graph_gt, get_join_graph_gt_json_path())

def load_join_graph_gt():
    return utils.file_load(get_join_graph_gt_json_path())
