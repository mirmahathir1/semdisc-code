from collections import defaultdict
from custom_lib import utils

# ────────────────────────── Build the undirected graph ──────────────────────────
def build_graph(joinability_dict):
    """
    Flatten the nested “joinability” JSON into an undirected adjacency list.
    """
    adj = defaultdict(set)
    for src, links in joinability_dict.items():
        for dst in links:
            adj[src].add(dst)
            adj[dst].add(src)
    return adj


# ───────────────────── Path enumeration with reverse-duplicate filter ───────────
def _canonical(path):
    """
    Return a hashable, orientation-independent encoding of the path
    so that [A,B,C] and [C,B,A] map to the same key.
    """
    return tuple(path) if path[0] < path[-1] else tuple(reversed(path))


def _extend(adj, path, visited, results, k):
    """
    Depth-first search that extends an induced path until it contains *k* vertices.
    Only paths of exactly length *k* are recorded (reverse duplicates removed).
    """
    if len(path) == k:                          # reached target size
        results.add(_canonical(path))
        return

    tail = path[-1]
    for nxt in adj[tail]:
        if nxt in visited:
            continue
        # induced-path restriction: nxt must NOT be adjacent to any vertex in path[:-1]
        if any(nxt in adj[u] for u in path[:-1]):
            continue

        visited.add(nxt)
        _extend(adj, path + [nxt], visited, results, k)
        visited.remove(nxt)                     # back-track


def induced_paths_of_length(joinability_dict, k):
    """
    Enumerate **all** simple induced paths of exactly *k* vertices (k ≥ 2).
    Reversed paths are treated as duplicates and returned once.

    Parameters
    ----------
    joinability_dict : dict
        Nested JSON mapping {table: [joinable_table, …]}.
    k : int
        Desired number of vertices in each path (≥ 2).

    Returns
    -------
    list[list[str]]
        Each element is a list of table names representing one induced path.
    """
    if k < 2:
        raise ValueError("k must be ≥ 2 (need at least one edge).")

    adj = build_graph(joinability_dict)
    results = set()

    for start in adj:
        _extend(adj, [start], {start}, results, k)

    # convert back to list-of-lists before returning
    return [list(p) for p in results]

def _random_dfs(adj, path, visited, seen, k, rng):
    """
    Randomised DFS: return the *first* length-k induced path
    that hasn’t been yielded before, or None.
    """
    if len(path) == k:
        key = _canonical(path)
        return None if key in seen else list(path)

    tail = path[-1]
    # neighbours that keep the path induced
    nxts = [
        n for n in adj[tail]
        if n not in visited and all(n not in adj[u] for u in path[:-1])
    ]
    rng.shuffle(nxts)                          # random exploration order

    for n in nxts:
        visited.add(n)
        res = _random_dfs(adj, path + [n], visited, seen, k, rng)
        visited.remove(n)
        if res is not None:
            return res
    return None

# --------------------------------------------------- public streaming generator
def random_induced_paths_stream(joinability_dict, k, *, seed=None):
    """
    Yield every simple induced path of exactly *k* vertices once, in random order,
    without ever storing the full collection.

    Parameters
    ----------
    joinability_dict : dict   {table: [joinable_table, …]}
    k                : int    number of vertices in each path (≥2)
    seed             : int|None  reproducible randomness if provided
    """
    if k < 2:
        raise ValueError("k must be ≥ 2")

    adj     = build_graph(joinability_dict)
    verts   = list(adj.keys())
    rng     = utils.get_random_variable()
    seen    = set()             # canonical paths already yielded

    while True:
        rng.shuffle(verts)      # fresh random start order each round
        for start in verts:
            path = _random_dfs(adj, [start], {start}, seen, k, rng)
            if path is not None:
                seen.add(_canonical(path))
                yield path
                break           # restart outer loop for new randomness
        else:
            # scanned every vertex without finding a new path → done
            return