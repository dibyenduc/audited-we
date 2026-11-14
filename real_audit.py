import json, math, random, sys, time
import numpy as np
from toy_we import step_rng
from real_audit_core import (commit, run_log, publish, estimate, check_bookkeeping,
                             check_cells, all_cells, sample_token)

def mc_estimate(cfg, next_topk, n_mc, label):
    root = commit(dict(cfg, label="mc-" + label))
    u = step_rng(root, 0, "mc").random((n_mc, cfg["H"]))
    S, hits = set(cfg["trigger_ids"]), 0
    for j in range(n_mc):
        pre, c = list(cfg["prompt_ids"]), 0
        for t in range(cfg["H"]):
            toks, q = next_topk(pre)
            tk = sample_token(toks, q, u[j, t]); pre.append(tk); c += tk in S
        hits += c >= 2
    return hits / n_mc

def probe(next_topk, prompt_ids, trig):
    S = set(trig); P = list(prompt_ids)
    t1, q1 = next_topk(P)
    p1 = float(sum(pr for t, pr in zip(t1, q1) if int(t) in S))
    pA, num = 0.0, 0.0
    for a, qa in zip(t1, q1):
        if int(a) in S: continue
        t2, q2 = next_topk(P + [int(a)])
        for b, qb in zip(t2, q2):
            if int(b) in S:
                pA += float(qa) * float(qb)
                t3, q3 = next_topk(P + [int(a), int(b)])
                num += float(qa) * float(qb) * float(sum(x for c, x in zip(t3, q3) if int(c) in S))
    pB = num / pA if pA > 0 else 0.0
    return p1, pA, pB

def main(next_topk, prompt_ids, trig, truth, env, R=200, A=50, top_k=50, n0=128, per_bin=16):
    cfg0 = {"prompt_ids": prompt_ids, "trigger_ids": trig, "H": 3, "per_bin": per_bin,
            "n0": n0, "top_k": top_k, "env": env}
    t0 = time.time()
    p1, pA, pB = probe(next_topk, prompt_ids, trig)
    print(f"exact probability: {truth:.4e}")
    print(f"probe: P(token 1 is a trigger) = {p1:.3e}")
    print(f"       P(token 2 is a trigger, token 1 not) = {pA:.3e}; P(token 3 is a trigger | that) = {pB:.3e}")
    print(f"       with per_bin={per_bin}: expected walkers entering bin 1 at step 2 = {per_bin*pA:.2f}; "
          f"expected hit walkers at step 3 if bin 1 is occupied = {per_bin*pB:.2f}")
    print(f"       suggestion: n0 = per_bin = about {math.ceil(5/max(min(pA, pB), 1e-9))}")
    if R == 0: return
    est, budgets = [], []
    for r in range(R):
        cfg = dict(cfg0, label=f"honest-{r}")
        cps, _ = run_log(cfg, next_topk)
        est.append(estimate(cps[-1]))
        budgets.append(sum(len(cp["pre"]) for cp in cps[:-1]))
    mean = np.mean(est); se = np.std(est, ddof=1) / math.sqrt(R)
    n_mc = int(np.mean(budgets)) // cfg0["H"]
    mc = [mc_estimate(cfg0, next_topk, n_mc, str(r)) for r in range(R)]
    z = (mean - truth) / se if se > 0 else float("nan")
    print(f"WE  over {R} runs: mean {mean:.3e} +/- {se:.1e}  (z vs exact = {z:+.2f}), zero-hit runs {sum(e == 0 for e in est)}/{R}")
    print(f"MC  equal budget ({n_mc} sequences/run): zero-hit runs {sum(e == 0 for e in mc)}/{R}, mean {np.mean(mc):.3e}")
    print(f"[{time.time()-t0:.0f}s elapsed]")
    if A == 0: return

    vrng = random.Random(2026)
    nominal = 3 * per_bin
    ms = [0, nominal // 100, nominal // 20, nominal // 10, nominal // 4, nominal // 2]
    res = {"honest": 0, "boost": 0, "boost_n": 0}
    inflate = {x: {"n": 0, "bias": [], "book": 0, "nz": 0, **{m: 0 for m in ms}} for x in (1, 4, 16, 64)}
    supp = {"n": 0, "book": 0, **{m: 0 for m in ms}}
    N = []
    for a in range(A):
        cfg = dict(cfg0, label=f"audit-{a}"); c = commit(cfg)
        h = publish(cfg, run_log(cfg, next_topk)[0]); N.append(len(all_cells(h)))
        res["honest"] += check_bookkeeping(h, c)[0] and check_cells(h, next_topk, vrng.sample(all_cells(h), 25))
        btr = publish(cfg, run_log(cfg, next_topk, mode="boost")[0])
        if btr["claimed"] > 0:
            res["boost_n"] += 1
            res["boost"] += not check_bookkeeping(btr, c)[0]
        for x in inflate:
            cps, forged = run_log(cfg, next_topk, mode="inflate", x=x)
            tr = publish(cfg, cps); d = inflate[x]
            d["n"] += 1; d["bias"].append(tr["claimed"] / truth); d["nz"] += tr["claimed"] > 0
            d["book"] += not check_bookkeeping(tr, c)[0]
            cells = all_cells(tr)
            for m in ms: d[m] += not check_cells(tr, next_topk, vrng.sample(cells, min(m, len(cells))))
        cps, forged = run_log(cfg, next_topk, mode="suppress")
        if forged:
            tr = publish(cfg, cps); supp["n"] += 1
            supp["book"] += not check_bookkeeping(tr, c)[0]
            cells = all_cells(tr)
            for m in ms: supp[m] += not check_cells(tr, next_topk, vrng.sample(cells, min(m, len(cells))))
    Nbar = int(np.mean(N))
    print(f"\nAUDIT over {A} transcripts; model-move cells per transcript: about {Nbar}")
    print(f"honest accepted: {res['honest']}/{A}")
    print(f"weight-boost tamper caught by bookkeeping alone: {res['boost']}/{res['boost_n']} (transcripts with at least one hit)")
    print("targeted INFLATION (1 forged move at step 2 to seed a chain, then x forged moves at step 3)")
    print("   x | est>0 runs | mean est/exact | bookkeeping | " + " | ".join(f"m={m:<5d}" for m in ms))
    for x, d in inflate.items():
        print(f" {x:3d} | {d['nz']:5d}/{d['n']:<4d} | {np.mean(d['bias']):14.1f} | {d['book']:5d}/{d['n']:<4d} | " + " | ".join(f"{d[m]/d['n']:7.0%}" for m in ms))
    if supp["n"]:
        print(f"SUPPRESSION (hide every hit), on the {supp['n']} transcripts that had hits")
        print("  bookkeeping caught: " + f"{supp['book']}/{supp['n']}  | " + " | ".join(f"m={m}: {supp[m]/supp['n']:.0%}" for m in ms))
    print(f"[{time.time()-t0:.0f}s total]")

if __name__ == "__main__":
    import torch
    from lm import load_lm, MODEL
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    A = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    n0 = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    per_bin = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    info = json.load(open("exact_truth.json"))
    tok, next_topk = load_lm(info["top_k"])
    main(next_topk, tok.encode(info["prompt"]), info["trigger_ids"], info["exact_p"],
         f"torch {torch.__version__}; {MODEL}", R, A, info["top_k"], n0, per_bin)

