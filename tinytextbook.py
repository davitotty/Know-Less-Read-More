"""
TinyTextbook: a minimal synthetic benchmark for instructional in-context learning.

Every episode is a small "textbook" (facts about functions that exist only in the
context) followed by exercises. The functions are resampled for every episode, so
the only way to answer an exercise is to read the material.

Usage:
    python tinytextbook.py --exp e1 --seed 0 --out results/
    python tinytextbook.py --exp e2 --seed 0 --out results/
    python tinytextbook.py --exp e3 --seed 0 --out results/
    python tinytextbook.py --exp e4 --seed 0 --out results/
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Vocabulary and episode geometry
# ----------------------------------------------------------------------------
K = 5            # number of value symbols
N_NAMES = 8      # pool of function-name tokens (re-assigned in every textbook)
PAD, BOS, Q, EQ = 0, 1, 2, 3
SYM0 = 4
NAME0 = SYM0 + K
VOCAB = NAME0 + N_NAMES
N_DEF = 3        # fully tabulated functions in a full-size textbook
N_EXS = 2        # example facts shown for the function that must be induced
MAX_DEPTH = 6
MAX_QUERIES = 12
MATERIAL_LEN = 1 + 3 * (N_DEF * K + N_EXS)
MAX_LEN = MATERIAL_LEN + MAX_QUERIES * (MAX_DEPTH + 4)
IND = -1         # index of the induced function inside a chain
# "aligned": every textbook uses the same page layout (a table with fixed slots), so a
#            value is found by its address.  "shuffled": facts are free-floating triples in
#            random order under random names, so a value must be found by its content.
LAYOUT = os.environ.get("TT_LAYOUT", "aligned")


class Textbook:
    """n_def fully tabulated permutations + one function shown only through N_EXS examples.

    The induced function follows the hidden rule x -> x + shift (mod K).
    """

    def __init__(self, rng, n_def=N_DEF, perm=False):
        self.n_def = n_def
        # Training textbooks tabulate arbitrary maps; test textbooks tabulate permutations, for
        # which the answer is uniform over the K symbols whatever the model has (not) read.
        self.perms = [rng.permutation(K) if perm else rng.integers(K, size=K) for _ in range(n_def)]
        self.shift = int(rng.integers(1, K))
        self.ex_keys = [int(k) for k in rng.permutation(K)[:N_EXS]]
        examples = [(k, (k + self.shift) % K) for k in self.ex_keys]
        toks = [BOS]
        if LAYOUT == "aligned":
            # Row j lists f_j(0), ..., f_j(K-1); the function in row j is always called name j.
            assert n_def == N_DEF
            self.names, self.ind_name = list(range(n_def)), n_def
            for j in range(n_def):
                toks += [SYM0 + int(v) for v in self.perms[j]]
            for k, v in examples:
                toks += [SYM0 + k, SYM0 + v]
        else:
            # A fact is the triple `f k v` (f(k) = v); names are random and facts are shuffled.
            names = rng.permutation(N_NAMES)[: n_def + 1]
            self.names, self.ind_name = [int(n) for n in names[:n_def]], int(names[n_def])
            facts = [(self.names[j], k, int(self.perms[j][k])) for j in range(n_def) for k in range(K)]
            facts += [(self.ind_name, k, v) for k, v in examples]
            for i in rng.permutation(len(facts)):
                n, k, v = facts[i]
                toks += [NAME0 + n, SYM0 + k, SYM0 + v]
        self.tokens = toks

    def query(self, rng, kind, depth=None):
        """Return (tokens, answer).  Prefix notation, innermost last:  Q h g f x =  ->  h(g(f(x)))."""
        while True:
            if kind == "follow":
                chain = [int(rng.integers(self.n_def))]
            elif kind == "compose":
                chain = [int(rng.integers(self.n_def)) for _ in range(depth)]
            elif kind == "induce":
                chain = [IND]
            elif kind == "mixed":                              # tabulated + induced in one chain
                chain = [int(rng.integers(self.n_def)) for _ in range(depth)]
                chain[int(rng.integers(depth))] = IND
            else:
                raise ValueError(kind)
            x = int(rng.integers(K))
            v, ok = x, True
            for j in chain:                                    # chain is in order of application
                if j == IND:
                    if v in self.ex_keys:                      # must require induction, not lookup
                        ok = False
                        break
                    v = (v + self.shift) % K
                else:
                    v = int(self.perms[j][v])
            if ok:
                break
        name = lambda j: NAME0 + (self.ind_name if j == IND else self.names[j])
        return [Q] + [name(j) for j in reversed(chain)] + [SYM0 + x, EQ], SYM0 + v


def sample_kind(rng, mix):
    """mix: list of (prob, kind, depth-choices)."""
    r, acc = rng.random(), 0.0
    for p, kind, depths in mix:
        acc += p
        if r < acc:
            break
    return kind, (int(rng.choice(depths)) if depths else None)


def make_batch(rng, bs, mix, n_queries, pool=None, var_def=False, perm=False):
    """Returns x (bs, T) and y (bs, T); y is -100 everywhere except at '=' positions."""
    X = np.zeros((bs, MAX_LEN), dtype=np.int64)
    Y = np.full((bs, MAX_LEN), -100, dtype=np.int64)
    L = 0
    for b in range(bs):
        if pool is not None:
            tb = pool[int(rng.integers(len(pool)))]
        else:
            n_def = int(rng.integers(1, N_DEF + 1)) if var_def and LAYOUT != "aligned" else N_DEF
            tb = Textbook(rng, n_def, perm)
        toks = list(tb.tokens)
        for _ in range(n_queries):
            kind, depth = sample_kind(rng, mix)
            q, a = tb.query(rng, kind, depth)
            toks += q
            Y[b, len(toks) - 1] = a
            toks.append(a)
        X[b, : len(toks)] = toks
        L = max(L, len(toks))
    return (torch.from_numpy(np.ascontiguousarray(X[:, :L])),
            torch.from_numpy(np.ascontiguousarray(Y[:, :L])))


# ----------------------------------------------------------------------------
# Model: a small causal transformer; `loops` re-applies the same block stack.
# ----------------------------------------------------------------------------
def rope(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        q, k = rope(q, cos, sin), rope(k, cos, sin)            # rotary position encoding
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, d=64, heads=4, layers=4, loops=1):
        super().__init__()
        self.loops = loops
        self.tok, self.pos = nn.Embedding(VOCAB, d), nn.Embedding(MAX_LEN, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, VOCAB)
        half = d // heads // 2
        ang = torch.arange(MAX_LEN)[:, None] * (1.0 / 1000 ** (torch.arange(half) / half))[None]
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T))        # absolute addresses ...
        cos, sin = self.cos[:T], self.sin[:T]                # ... plus rotary relative offsets
        for _ in range(self.loops):                            # recurrent depth: shared weights
            for blk in self.blocks:
                x = blk(x, cos, sin)
        return self.head(self.ln(x))


@torch.no_grad()
def evaluate(model, rng, mix, n=1000, pool=None, bs=250, perm=True):
    """Accuracy on a single exercise that directly follows a full-size textbook."""
    model.eval()
    correct = total = 0
    for _ in range(n // bs):
        x, y = make_batch(rng, bs, mix, 1, pool, perm=perm)     # fresh textbooks: permutations by default
        pred = model(x).argmax(-1)
        m = y != -100
        correct += (pred[m] == y[m]).sum().item()
        total += m.sum().item()
    model.train()
    return correct / total


def train(cfg, tests, log):
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    pool = None
    if cfg.get("n_textbooks"):
        prng = np.random.default_rng(10_000 + cfg["seed"])
        pool = [Textbook(prng) for _ in range(cfg["n_textbooks"])]
    model = TinyLM(cfg["d"], cfg["heads"], cfg["layers"], cfg["loops"])
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01, betas=(0.9, 0.98))
    steps, warm = cfg["steps"], 200
    sched = lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))
    curve, t0 = [], time.time()
    for step in range(steps + 1):
        if step % cfg["eval_every"] == 0 or step == steps:
            final = step == steps
            erng = np.random.default_rng(777 + step)
            res = {name: evaluate(model, erng, spec[0], 4000 if final else 500,
                                  pool if spec[1] else None, perm=spec[2] if len(spec) > 2 else True)
                   for name, spec in tests.items()}
            curve.append({"step": step, **res})
            log(f"  step {step:5d} | {time.time()-t0:6.0f}s | " +
                " ".join(f"{k}={v:.3f}" for k, v in res.items()))
            if final:
                break
        for g in opt.param_groups:
            g["lr"] = cfg["lr"] * sched(step)
        x, y = make_batch(rng, cfg["bs"], cfg["mix"], cfg["nq"], pool, cfg["var_def"],
                          perm=cfg.get("train_perm", False))
        loss = F.cross_entropy(model(x).view(-1, VOCAB), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return {"cfg": {k: v for k, v in cfg.items() if k != "mix"}, "n_params": n_params,
            "final": curve[-1], "curve": curve, "seconds": time.time() - t0}


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------
T_FOLLOW = [(1.0, "follow", None)]
T_INDUCE = [(1.0, "induce", None)]


def T_COMPOSE(d):
    return [(1.0, "compose", [d])]


def T_MIXED(d):
    return [(1.0, "mixed", [d])]


def experiments(exp):
    # Default model: a 2-layer block applied twice (4 layer passes, weights shared across loops).
    base = dict(d=64, heads=4, layers=2, loops=2, lr=2e-3, bs=32, nq=8, var_def=True,
                steps=5000, eval_every=500)
    runs = []
    if exp == "e0":      # diagnostic: lookup only
        tests = {"follow": (T_FOLLOW, False)}
        runs.append(("follow_only", dict(base, mix=T_FOLLOW), tests))
    elif exp == "e1":    # read vs. memorise: number of distinct textbooks seen in training
        mix = [(0.5, "follow", None), (0.5, "compose", [2])]
        for M in [1, 4, 16, 64, 0]:
            tests = {"seen_follow": (T_FOLLOW, True), "seen_compose2": (T_COMPOSE(2), True),
                     "new_follow": (T_FOLLOW, False), "new_compose2": (T_COMPOSE(2), False)}
            runs.append((f"M={M if M else 'inf'}", dict(base, mix=mix, n_textbooks=M), tests))
    elif exp == "e2":    # which skills must be in the weights: leave-one-skill-out
        f, c, i = ("follow", None), ("compose", [2]), ("induce", None)
        conds = {"full": [f, c, i], "no_induce": [f, c], "no_compose": [f, i], "no_follow": [c, i]}
        tests = {"follow": (T_FOLLOW, False), "compose2": (T_COMPOSE(2), False),
                 "induce": (T_INDUCE, False), "mixed2": (T_MIXED(2), False),
                 "compose3": (T_COMPOSE(3), False)}
        for name, skills in conds.items():
            mix = [(1.0 / len(skills), k, d) for k, d in skills]
            runs.append((name, dict(base, mix=mix), tests))
    elif exp == "e3":    # recurrent depth: same weights, more loops
        mix = [(1.0, "compose", [1, 2, 3, 4])]
        tests = {f"depth{d}": (T_COMPOSE(d), False) for d in range(1, MAX_DEPTH + 1)}
        for name, layers, loops in [("tied_2x1", 2, 1), ("tied_2x2", 2, 2), ("tied_2x3", 2, 3),
                                    ("untied_4", 4, 1), ("untied_6", 6, 1)]:
            runs.append((name, dict(base, mix=mix, layers=layers, loops=loops, steps=8000), tests))
    elif exp == "e4":    # symmetry: are the training tables permutations or arbitrary maps?
        tests = {"follow_perm": (T_FOLLOW, False, True), "follow_map": (T_FOLLOW, False, False)}
        for name, train_perm in [("train_maps", False), ("train_perms", True)]:
            runs.append((name, dict(base, mix=T_FOLLOW, train_perm=train_perm), tests))
    return runs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", default=None)
    ap.add_argument("--threads", type=int, default=1)
    for key, typ in [("steps", int), ("d", int), ("lr", float), ("bs", int), ("nq", int),
                     ("var_def", int), ("layers", int), ("loops", int), ("eval_every", int)]:
        ap.add_argument(f"--{key}", type=typ, default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    for name, cfg, tests in experiments(a.exp):
        if a.only and name not in a.only.split(","):
            continue
        cfg["seed"] = a.seed
        for key in ("steps", "d", "lr", "bs", "nq", "var_def", "layers", "loops", "eval_every"):
            if getattr(a, key) is not None:
                cfg[key] = getattr(a, key)
        path = os.path.join(a.out, f"{a.exp}_{name}_s{a.seed}.json")
        if os.path.exists(path):
            continue
        print(f"[{a.exp}] {name} seed={a.seed}", flush=True)
        res = train(cfg, tests, lambda s: print(s, flush=True))
        res["exp"], res["name"] = a.exp, name
        json.dump(res, open(path, "w"), indent=1)
