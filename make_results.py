"""Turn the JSON logs in results/ into LaTeX macros, tables and figures for the paper.

Nothing in the paper's gen/ directory is typed by hand: run this script after the runs finish.

    python make_results.py results ../paper/gen
"""
import glob, json, os, re, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = sys.argv[2] if len(sys.argv) > 2 else "../paper/gen"
os.makedirs(OUT, exist_ok=True)

runs = defaultdict(lambda: defaultdict(list))        # exp -> condition -> [run, ...]
for p in sorted(glob.glob(os.path.join(RES, "*.json"))):
    r = json.load(open(p))
    runs[r["exp"]][r["name"]].append(r)
for e in runs.values():
    for rs in e.values():
        rs.sort(key=lambda r: r["cfg"]["seed"])

macros = {}
GUESS = {int(k): v for k, v in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "guess_levels.json"))).items()}
WORDS = dict(zip("0123456789", "Zero One Two Three Four Five Six Seven Eight Nine".split()))
pct = lambda v: f"{100 * v:.1f}"
num = lambda n: f"{n:,}".replace(",", "{,}")


def mname(*parts):
    """LaTeX macro names may contain letters only."""
    out = ""
    for part in parts:
        for chunk in re.findall(r"[A-Za-z]+|\d", part):
            out += WORDS.get(chunk, chunk[0].upper() + chunk[1:])
    return out


def listing(items):
    """'a', 'a and b', 'a, b and c'."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def vals(rs, key):
    return np.array([r["final"][key] for r in rs])


SOLVED = 0.95            # a seed "solves" a test if it reaches this accuracy


def cell(rs, key):
    v = vals(rs, key)
    if len(v) == 1:
        return pct(v[0])
    return f"{pct(v.mean())} {{\\scriptsize\\,{int((v >= SOLVED).sum())}/{len(v)}}}"


def table(path, header, rows, colspec):
    with open(path, "w") as f:
        f.write(f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n")
        for row in rows:
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def register(exp_tag, exp):
    for name, rs in runs[exp].items():
        for k in rs[0]["final"]:
            if k != "step":
                base = mname(exp_tag, name.replace("M=", "M ").replace("inf", "Inf"), k)
                v = vals(rs, k)
                macros[base] = pct(v.mean())
                macros[base + "Min"], macros[base + "Max"] = pct(v.min()), pct(v.max())
                macros[base + "Solved"] = str(int((v >= SOLVED).sum()))
                macros[base + "List"] = listing([pct(x) for x in v])          # one value per seed
        macros[mname(exp_tag, name.replace("M=", "M ").replace("inf", "Inf"), "Seeds")] = str(len(rs))
    macros[mname(exp_tag, "MinSeeds")] = str(min(len(rs) for rs in runs[exp].values()))
    macros[mname(exp_tag, "MaxSeeds")] = str(max(len(rs) for rs in runs[exp].values()))


for d, v in GUESS.items():                      # exact table-blind guessing levels (guess_levels.py)
    macros["Guess" + WORDS[str(d)]] = f"{100 * v:.1f}"

plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "axes.linewidth": 0.6})
C = ["#1b4f72", "#b9770e", "#1e8449", "#922b21", "#5b2c6f", "#515a5a"]
CHANCE = dict(color="gray", lw=0.7, ls=":")


def curve(rs, key):
    steps = [pt["step"] for pt in rs[0]["curve"]]
    return steps, np.mean([[pt[key] * 100 for pt in r["curve"]] for r in rs], axis=0)


# ---------------------------------------------------------------------- E1 + E2 figure
E1_ORDER = ["M=1", "M=4", "M=16", "M=64", "M=inf"]
E2_ORDER = ["full", "no_follow", "no_compose", "no_induce"]
E2_NAMES = {"full": "Full curriculum", "no_follow": "without \\skill{follow}",
            "no_compose": "without \\skill{compose}", "no_induce": "without \\skill{induce}"}
mlab = lambda n: "$\\infty$" if n == "M=inf" else n[2:]

if "e1" in runs:
    order = [n for n in E1_ORDER if n in runs["e1"]]
    keys = ["seen_follow", "seen_compose2", "new_follow", "new_compose2"]
    table(f"{OUT}/tab_e1.tex",
          "Distinct textbooks & \\multicolumn{2}{c}{G0: seen textbooks} & \\multicolumn{2}{c}{G1: new textbooks} \\\\\n"
          "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
          "in training ($M$) & \\skill{follow} & \\skill{compose}-2 & \\skill{follow} & \\skill{compose}-2",
          [[mlab(n)] + [cell(runs["e1"][n], k) for k in keys] for n in order], "@{}lcccc@{}")
    register("EOne", "e1")

if "e2" in runs:
    order = [n for n in E2_ORDER if n in runs["e2"]]
    keys = ["follow", "compose2", "induce", "mixed2", "compose3"]
    table(f"{OUT}/tab_e2.tex",
          "& \\multicolumn{3}{c}{G1: exercise types in the curriculum} & \\multicolumn{2}{c}{G2: never trained} \\\\\n"
          "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\n"
          "Training set & \\skill{follow} & \\skill{compose}-2 & \\skill{induce} & \\skill{mixed}-2 & \\skill{compose}-3",
          [[E2_NAMES[n]] + [cell(runs["e2"][n], k) for k in keys] for n in order], "@{}lccccc@{}")
    register("ETwo", "e2")

if "e1" in runs and "e2" in runs:
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 3.0))
    order = [n for n in E1_ORDER if n in runs["e1"]]
    xs = np.arange(len(order))
    for k, c, l, ls, mk in [("seen_follow", C[0], "seen textbooks, follow", "-", "o"),
                            ("seen_compose2", C[0], "seen textbooks, compose-2", "--", "s"),
                            ("new_follow", C[1], "new textbooks, follow", "-", "o"),
                            ("new_compose2", C[1], "new textbooks, compose-2", "--", "s")]:
        a.plot(xs, [vals(runs["e1"][n], k).mean() * 100 for n in order], ls, marker=mk, color=c, label=l, ms=3.5, lw=1.1)
    a.axhline(20, **CHANCE)
    a.set_xticks(xs); a.set_xticklabels([mlab(n) for n in order])
    a.set_xlabel("distinct textbooks in training ($M$)"); a.set_ylabel("accuracy (%)"); a.set_ylim(0, 106)
    a.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2); a.set_title("(a) memorising vs. reading", fontsize=8.5)
    for k, c, l in [("follow", C[0], "follow"), ("compose2", C[1], "compose-2"), ("induce", C[2], "induce"),
                    ("mixed2", C[3], "mixed-2 (never trained)")]:
        for i, r in enumerate(runs["e2"]["full"]):
            st, cu = curve([r], k)
            b.plot(st, cu, color=c, lw=1.1, alpha=1.0 if i == 0 else 0.45, label=l if i == 0 else None)
    b.axhline(20, **CHANCE)
    b.set_xlabel("training step"); b.set_ylabel("accuracy, new textbooks (%)"); b.set_ylim(0, 106)
    b.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2); b.set_title("(b) full curriculum, order of acquisition", fontsize=8.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_e1_e2.pdf"); plt.close(fig)

# ---------------------------------------------------------------------- E2 pooled counts
if "e2" in runs:
    from math import comb

    def pooled(conds, key):
        v = np.concatenate([vals(runs["e2"][c], key) for c in conds if c in runs["e2"]])
        return int((v >= SOLVED).sum()), len(v)

    for tag, conds, key in [("FollowTrained", ["full", "no_compose", "no_induce"], "follow"),
                            ("ComposeWithFollow", ["full", "no_induce"], "compose2"),
                            ("ComposeWithoutFollow", ["no_follow"], "compose2"),
                            ("InduceTrained", ["full", "no_compose", "no_follow"], "induce"),
                            ("MixedAll", E2_ORDER, "mixed2"), ("ComposeThreeAll", E2_ORDER, "compose3")]:
        k, n = pooled(conds, key)
        macros[f"ETwoPool{tag}Solved"], macros[f"ETwoPool{tag}Runs"] = str(k), str(n)
    macros["ETwoMixedMax"] = pct(max(r["final"]["mixed2"] for rs in runs["e2"].values() for r in rs))
    macros["ETwoRunsAll"] = str(sum(len(rs) for rs in runs["e2"].values()))
    macros["ETwoFullAllSolved"] = str(sum(all(r["final"][k] >= SOLVED for k in ("follow", "compose2", "induce"))
                                          for r in runs["e2"]["full"]))
    # one-sided Fisher exact test: is compose-2 solved less often without follow in training?
    (a1, n1), (a2, n2) = pooled(["no_follow"], "compose2"), pooled(["full", "no_induce"], "compose2")
    tot = a1 + a2
    p = sum(comb(n1, i) * comb(n2, tot - i) for i in range(0, a1 + 1) if 0 <= tot - i <= n2) / comb(n1 + n2, tot)
    macros["ETwoFisherP"] = f"{p:.3f}"

# ---------------------------------------------------------------------- E3
E3_ORDER = ["tied_2x1", "tied_2x2", "tied_2x3", "untied_4", "untied_6"]
E3_NAMES = {"tied_2x1": "2 layers $\\times$ 1 loop", "tied_2x2": "2 layers $\\times$ 2 loops",
            "tied_2x3": "2 layers $\\times$ 3 loops", "untied_4": "4 layers, unshared", "untied_6": "6 layers, unshared"}
E3_PLAIN = {"tied_2x1": "2 layers × 1 loop", "tied_2x2": "2 layers × 2 loops", "tied_2x3": "2 layers × 3 loops",
            "untied_4": "4 layers, unshared", "untied_6": "6 layers, unshared"}
E3_PASSES = {"tied_2x1": 2, "tied_2x2": 4, "tied_2x3": 6, "untied_4": 4, "untied_6": 6}
E3_STYLE = {"tied_2x1": (C[5], "-"), "tied_2x2": (C[0], "-"), "tied_2x3": (C[2], "-"),
            "untied_4": (C[0], "--"), "untied_6": (C[2], "--")}
if "e3" in runs:
    order = [n for n in E3_ORDER if n in runs["e3"]]
    keys = [f"depth{d}" for d in range(1, 7)]
    rows = []
    for n in order:
        rs = runs["e3"][n]
        rows.append([E3_NAMES[n], f"{rs[0]['n_params'] / 1000:.0f}k", str(E3_PASSES[n]), str(len(rs))] + [cell(rs, k) for k in keys])
        macros[mname("EThree", n, "Params")] = num(rs[0]["n_params"])
    table(f"{OUT}/tab_e3.tex",
          "& & Layer & & \\multicolumn{4}{c}{G1: chain lengths in training} & \\multicolumn{2}{c}{G2: longer} \\\\\n"
          "\\cmidrule(lr){5-8}\\cmidrule(lr){9-10}\n"
          "Model & Params & passes & Seeds & 1 & 2 & 3 & 4 & 5 & 6", rows, "@{}lccccccccc@{}")
    register("EThree", "e3")
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 3.15))
    for n in order:
        c, ls = E3_STYLE[n]
        a.plot(range(1, 7), [vals(runs["e3"][n], k).mean() * 100 for k in keys], ls, marker="o", ms=3.5, lw=1.1,
               color=c, label=E3_PLAIN[n])
    a.plot(range(1, 7), [100 * GUESS[d] for d in range(1, 7)], color="gray", lw=0.9, ls=":", label="guessing level")
    a.axvline(4.5, **CHANCE)
    a.text(4.6, 96, "not in\ntraining", fontsize=6.3, color="gray", va="top")
    a.set_xlabel("chain length of the exercise"); a.set_ylabel("accuracy, new textbooks (%)"); a.set_ylim(0, 106)
    a.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    a.set_title("(a) final accuracy by chain length", fontsize=8.5)
    for n in [m for m in ("tied_2x2", "untied_4") if m in runs["e3"]]:
        c, ls = E3_STYLE[n]
        for k, col in [("depth1", C[5]), ("depth2", C[1]), ("depth3", C[3])]:
            for i, r in enumerate(runs["e3"][n]):
                st, cu = curve([r], k)
                b.plot(st, cu, ls, color=col, lw=1.1, alpha=1.0 if i == 0 else 0.45,
                       label=f"length {k[-1]}, {'looped' if n.startswith('tied') else 'unshared'}" if i == 0 else None)
    b.axhline(20, **CHANCE)
    b.set_xlabel("training step"); b.set_ylabel("accuracy, new textbooks (%)"); b.set_ylim(0, 106)
    b.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    b.set_title("(b) four layer passes: looped vs. unshared", fontsize=8.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_e3.pdf"); plt.close(fig)

# ---------------------------------------------------------------------- E4
E4_ORDER = ["train_maps", "train_perms"]
if "e4" in runs:
    rows = []
    for n, lab in [("train_maps", "Arbitrary maps"), ("train_perms", "Permutations")]:
        if n not in runs["e4"]:
            continue
        rs, first = runs["e4"][n], []
        for r in rs:
            st = [pt["step"] for pt in r["curve"] if pt["follow_perm"] >= SOLVED]
            first.append(num(st[0]) if st else "never")
        rows.append([lab, str(len(rs)), cell(rs, "follow_perm"), cell(rs, "follow_map"), ", ".join(first)])
        macros[mname("EFour", n, "First")] = ", ".join(first)
    table(f"{OUT}/tab_e4.tex",
          "Tables in & & \\multicolumn{2}{c}{\\skill{follow} on new textbooks with} & First checkpoint \\\\\n"
          "\\cmidrule(lr){3-4}\n"
          "training & Seeds & permutations & arbitrary maps & at $\\ge 95\\%$ (per seed)", rows, "@{}lcccc@{}")
    register("EFour", "e4")

# ---------------------------------------------------------------------- per-run appendix tables
PRETTY = {"seen_follow": "G0 fol.", "seen_compose2": "G0 comp-2", "new_follow": "G1 fol.",
          "new_compose2": "G1 comp-2", "follow": "follow", "compose2": "comp-2", "compose3": "comp-3",
          "induce": "induce", "mixed2": "mixed-2", "follow_perm": "perm.", "follow_map": "maps", **{f"depth{d}": f"len {d}" for d in range(1, 7)}}
ORDERS = {"e1": E1_ORDER, "e2": E2_ORDER, "e3": E3_ORDER, "e4": E4_ORDER}
for exp in ("e1", "e2", "e3", "e4"):
    if exp not in runs:
        continue
    keys = [k for k in next(iter(runs[exp].values()))[0]["final"] if k != "step"]
    rows = []
    for name in [n for n in ORDERS[exp] if n in runs[exp]]:
        for r in runs[exp][name]:
            rows.append([name.replace("_", "\\_").replace("M=inf", "M=$\\infty$"), str(r["cfg"]["seed"]),
                         num(r["n_params"]), num(r["cfg"]["steps"]), f"{r['seconds'] / 60:.1f}"]
                        + [pct(r["final"][k]) for k in keys])
    table(f"{OUT}/tab_runs_{exp}.tex",
          "Condition & Seed & Params & Steps & Min. & " + " & ".join(PRETTY[k] for k in keys),
          rows, "@{}lrrrr" + "r" * len(keys) + "@{}")

# ---------------------------------------------------------------------- misc
allr = [r for e in ("e1", "e2", "e3", "e4") for rs in runs.get(e, {}).values() for r in rs]
main = [r for e in ("e1", "e2") for rs in runs.get(e, {}).values() for r in rs]
if main:
    macros["NparamsMain"] = num(main[0]["n_params"])
    macros["NparamsRounded"] = f"{round(main[0]['n_params'] / 1000)} thousand"
macros["TotalRuns"] = str(len(allr))
macros["MinRunMinutes"] = f"{min(r['seconds'] for r in allr) / 60:.0f}"
macros["MaxRunMinutes"] = f"{max(r['seconds'] for r in allr) / 60:.0f}"
macros["TotalCpuHours"] = f"{sum(r['seconds'] for r in allr) / 3600:.1f}"

with open(f"{OUT}/numbers.tex", "w") as f:
    for k, v in sorted(macros.items()):
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote {len(macros)} macros, tables and figures to {OUT}")
if "-v" in sys.argv:
    for k, v in sorted(macros.items()):
        print(f"  {k} = {v}")
