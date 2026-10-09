"""Exact accuracy of the best guesser that sees the exercise but not the tables.

Test textbooks tabulate independent uniformly random permutations of K symbols. For a single
look-up the answer is uniform. In a chain, a function may occur more than once, and then the
answer equals the input x more often than 1/K (for instance f(f(x)) = x with probability 2/K).
This script enumerates all triples of permutations and all chains.
"""
import itertools, json, sys
import numpy as np

K, N_DEF = 5, 3
P = np.array(list(itertools.permutations(range(K))), dtype=np.int8)        # (120, K)
n = len(P)
A, B, C = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
IDX = [A.ravel(), B.ravel(), C.ravel()]                                    # every triple of tables


def canonical(chain):
    m = {}
    return tuple(m.setdefault(j, len(m)) for j in chain)


def p_fixed(chain):
    """P(chain(x) = x) over all triples of permutations (x = 0 without loss of generality)."""
    v = np.zeros(n ** 3, dtype=np.int8)
    for j in chain:
        v = P[IDX[j], v]
    return float((v == 0).mean())


out = {}
for d in range(1, 7):
    cache, acc = {}, 0.0
    for chain in itertools.product(range(N_DEF), repeat=d):
        c = canonical(chain)
        if c not in cache:
            cache[c] = p_fixed(c)
        px = cache[c]
        acc += max(px, (1 - px) / (K - 1))                                 # guess x, or anything else
    out[d] = acc / N_DEF ** d
    print(f"chain length {d}: {100 * out[d]:.2f}%  ({len(cache)} patterns)")
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "guess_levels.json", "w"), indent=1)
