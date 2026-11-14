import json, random, sys, time
import numpy as np
from real_audit_core import commit, run_log, publish, check_bookkeeping, check_cells, all_cells

C_EVENT = 2

def prefix_of(tr, t, i):
    return tuple(tr["cps"][t - 1]["pre"][i])

def cost(tr, cells):
    return len({prefix_of(tr, t, i) for t, i in cells})

def relevant_cells(tr):
    H, out = tr["config"]["H"], []
    for t in range(1, H + 1):
        remaining = H - t + 1
        for i, c in enumerate(tr["cps"][t - 1]["cnt"]):
            if c < C_EVENT and c + remaining >= C_EVENT:
                out.append((t, i))
    return out

def equal_cost_cells(tr, budget, vrng):
    cells = all_cells(tr); vrng.shuffle(cells)
    seen, chosen = set(), []
    for t, i in cells:
        p = prefix_of(tr, t, i)
        if p not in seen and len(seen) >= budget: continue
        seen.add(p); chosen.append((t, i))
    return chosen

def equal_cost_groups(tr, budget, vrng):
    groups = {}
    for t, i in all_cells(tr):
        groups.setdefault(prefix_of(tr, t, i), []).append((t, i))
    keys = list(groups); vrng.shuffle(keys)
    return [cell for k in keys[:budget] for cell in groups[k]]

def main(next_topk, prompt_ids, trig, truth, env, A=30, top_k=50, per_bin=2530):
    cfg0 = {"prompt_ids": prompt_ids, "trigger_ids": trig, "H": 3, "per_bin": per_bin,
            "n0": per_bin, "top_k": top_k, "env": env}
    vrng = random.Random(4242)
    attacks = [("suppress", 0), ("inflate", 1), ("inflate", 4), ("inflate", 16)]
    names = ["feasibility-pruned (all relevant cells)", "equal-cost uniform cells", "equal-cost uniform prefix groups"]
    det = {(s, a): 0 for s in names for a in attacks}
    nat = {a: 0 for a in attacks}
    honest_ok = {s: 0 for s in names}
    costs, totals, relcells, allc = [], [], [], []
    t0 = time.time()
    for a in range(A):
        cfg = dict(cfg0, label=f"strat-{a}"); c = commit(cfg)
        trs = {}
        h = publish(cfg, run_log(cfg, next_topk)[0])
        trs["honest"] = (h, 1)
        for atk in attacks:
            cps, forged = run_log(cfg, next_topk, mode=atk[0], x=atk[1])
            trs[atk] = (publish(cfg, cps), forged)
        rel = relevant_cells(h); b = cost(h, rel)
        costs.append(b); relcells.append(len(rel)); allc.append(len(all_cells(h)))
        totals.append(cost(h, all_cells(h)))
        def verdicts(tr):
            cells_f = relevant_cells(tr); budget = cost(tr, cells_f)
            return {names[0]: cells_f,
                    names[1]: equal_cost_cells(tr, budget, vrng),
                    names[2]: equal_cost_groups(tr, budget, vrng)}
        sel = verdicts(h)
        for s in names:
            honest_ok[s] += check_bookkeeping(h, c)[0] and check_cells(h, next_topk, sel[s])
        for atk in attacks:
            tr, forged = trs[atk]
            if atk[0] == "suppress" and not forged: continue
            nat[atk] += 1
            sel = verdicts(tr)
            for s in names:
                det[(s, atk)] += not check_cells(tr, next_topk, sel[s])
    print(f"transcripts: {A}; per transcript about {int(np.mean(allc))} cells, {int(np.mean(totals))} distinct prefixes in total")
    print(f"event-relevant cells: about {int(np.mean(relcells))} of them, needing about {np.mean(costs):.0f} distinct-prefix model passes")
    print(f"honest accepted: " + ", ".join(f"{honest_ok[s]}/{A}" for s in names))
    print()
    head = "strategy (same verifier cost, in model passes)".ljust(44) + " | suppress | infl x=1 | infl x=4 | infl x=16"
    print(head)
    for s in names:
        row = s.ljust(44)
        for atk in attacks:
            n = nat[atk]; row += f" | {det[(s, atk)]/n:8.0%}" + (" " if atk[0] == "suppress" else "  ")
        print(row)
    print("(suppression is scored on the " + str(nat[attacks[0]]) + f" transcripts that had hits)  [{time.time()-t0:.0f}s]")

if __name__ == "__main__":
    from lm import load_lm, MODEL
    import torch
    A = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    per_bin = int(sys.argv[2]) if len(sys.argv) > 2 else 2530
    info = json.load(open("exact_truth.json"))
    tok, next_topk = load_lm(info["top_k"])
    main(next_topk, tok.encode(info["prompt"]), info["trigger_ids"], info["exact_p"],
         f"torch {torch.__version__}; {MODEL}", A, info["top_k"], per_bin)

