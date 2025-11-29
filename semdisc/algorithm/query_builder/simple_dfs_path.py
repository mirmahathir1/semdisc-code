import random
from collections import defaultdict
from semdisc.algorithm.query_builder import simple_induced_paths
from custom_lib import utils

# ───────────────────── Random DFS for simple paths (non-induced) ────────────────
def _random_dfs_simple(adj, path, visited, seen, k, rng):
    """
    Return the first simple path of length *k* (k vertices) that hasn't been
    yielded before, or None if none remains reachable from this prefix.
    """
    if len(path) == k:
        key = simple_induced_paths._canonical(path)
        return None if key in seen else list(path)

    tail = path[-1]
    nxts = [n for n in adj[tail] if n not in visited]   # only “simple-path” rule
    rng.shuffle(nxts)                                   # random exploration order

    for n in nxts:
        visited.add(n)
        res = _random_dfs_simple(adj, path + [n], visited, seen, k, rng)
        visited.remove(n)
        if res is not None:                             # found a new path
            return res
    return None


# ───────────────────────── Streaming generator (public) ─────────────────────────
def random_simple_paths_stream(joinability_dict, k, *, seed=None):
    """
    Yield every simple path of exactly *k* vertices (k ≥ 2) once, in random order.

    Parameters
    ----------
    joinability_dict : dict   {table: [joinable_table, …]}
    k                : int    number of vertices in each path (≥ 2)
    seed             : int | None   reproducible randomness if provided
    """
    if k < 2:
        raise ValueError("k must be ≥ 2 (need at least one edge)")

    adj   = simple_induced_paths.build_graph(joinability_dict)
    verts = list(adj.keys())
    rng   = utils.get_random_variable()

    seen = set()        # canonicalised paths already produced

    while True:
        rng.shuffle(verts)            # fresh random start order each round
        for start in verts:
            path = _random_dfs_simple(adj, [start], {start}, seen, k, rng)
            if path is not None:
                seen.add(simple_induced_paths._canonical(path))
                yield path            # restart outer loop for new randomness
                break
        else:
            # every vertex scanned without discovering a new path → done
            return
