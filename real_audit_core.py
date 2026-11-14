import hashlib, json, math, random, sys, time
import numpy as np
from toy_we import step_rng

def H(b): return hashlib.sha256(b).digest()
def commit(cfg): return H(json.dumps(cfg, sort_keys=True).encode())
def hx(a): return [float(x).hex() for x in a]
def unhx(a): return np.array([float.fromhex(x) for x in a])
FILLER = 13

def sample_token(toks, q, u):
    c = np.cumsum(q); c = c / c[-1]
    return int(toks[min(int(np.searchsorted(c, u, side="right")), len(c) - 1)])

def resample(cfg, root, t, pre, cnt, w, tk):
    S, m = set(cfg["trigger_ids"]), cfg["per_bin"]
    cm = np.array([min(2, c + (1 if x in S else 0)) for c, x in zip(cnt, tk)])
    mpre = [p + [x] for p, x in zip(pre, tk)]
    new_pre, new_cnt, new_w, par = [], [], [], []
    for b in np.unique(cm):
        idx = np.where(cm == b)[0]
        total = w[idx].sum()
        kids = step_rng(root, t, f"resample-{b}").choice(idx, size=m, p=w[idx] / total)
        for k in kids:
            new_pre.append(mpre[int(k)]); new_cnt.append(int(b)); par.append(int(k))
        new_w += [total / m] * m
    return par, new_pre, new_cnt, np.array(new_w)

def run_log(cfg, next_topk, mode=None, x=0):
    root, S = commit(cfg), set(cfg["trigger_ids"])
    n = cfg["n0"]
    pre, cnt, w = [list(cfg["prompt_ids"]) for _ in range(n)], [0] * n, np.full(n, 1.0 / n)
    cps = [{"t": 0, "tk": [], "par": [], "pre": pre, "cnt": cnt, "w": hx(w)}]
    forged = 0
    for t in range(1, cfg["H"] + 1):
        u = step_rng(root, t, "move").random(len(pre))
        tk = []
        for i in range(len(pre)):
            toks, q = next_topk(pre[i]); tk.append(sample_token(toks, q, u[i]))
        if mode == "inflate" and t in (2, 3):
            k = 1 if t == 2 else x
            order = sorted([i for i in range(len(tk)) if tk[i] not in S],
                           key=lambda i: (-cnt[i], -w[i]))[:k]
            for i in order: tk[i] = cfg["trigger_ids"][0]
            forged += len(order)
        if mode == "suppress" and t >= 2:
            for i in range(len(tk)):
                if tk[i] in S and cnt[i] + 1 >= 2: tk[i] = FILLER; forged += 1
        par, npre, ncnt, nw = resample(cfg, root, t, pre, cnt, w, tk)
        if mode == "boost" and t == cfg["H"]:
            nw = nw.copy(); nw[np.array(ncnt) >= 2] *= 10.0; nw /= nw.sum()
        cps.append({"t": t, "tk": tk, "par": par, "pre": npre, "cnt": ncnt, "w": hx(nw)})
        pre, cnt, w = npre, ncnt, nw
    return cps, forged

def leaf(cp): return H(json.dumps(cp, sort_keys=True).encode())
def merkle_root(leaves):
    cur = list(leaves)
    while len(cur) > 1:
        if len(cur) % 2: cur.append(cur[-1])
        cur = [H(cur[i] + cur[i + 1]) for i in range(0, len(cur), 2)]
    return cur[0]

def estimate(cp):
    return float(unhx(cp["w"])[np.array(cp["cnt"]) >= 2].sum())

def publish(cfg, cps):
    return {"config": cfg, "cps": cps, "root": merkle_root([leaf(c) for c in cps]), "claimed": estimate(cps[-1])}

def check_bookkeeping(tr, expected_commitment):
    cfg, cps = tr["config"], tr["cps"]
    if commit(cfg) != expected_commitment: return False, "config != published commitment"
    if merkle_root([leaf(c) for c in cps]) != tr["root"]: return False, "merkle root mismatch"
    for t in range(1, cfg["H"] + 1):
        prev, cur = cps[t - 1], cps[t]
        if len(cur["tk"]) != len(prev["pre"]): return False, f"step {t}: wrong number of moves"
        par, npre, ncnt, nw = resample(cfg, expected_commitment, t, prev["pre"], prev["cnt"],
                                       unhx(prev["w"]), cur["tk"])
        if par != cur["par"] or npre != cur["pre"] or ncnt != cur["cnt"] or hx(nw) != cur["w"]:
            return False, f"step {t}: bookkeeping mismatch"
    if estimate(cps[-1]) != tr["claimed"]: return False, "claimed estimate wrong"
    return True, "ok"

def all_cells(tr):
    return [(t, i) for t in range(1, tr["config"]["H"] + 1) for i in range(len(tr["cps"][t - 1]["pre"]))]

def check_cells(tr, next_topk, cells):
    root = commit(tr["config"]); cps = tr["cps"]
    for t, i in cells:
        u = step_rng(root, t, "move").random(len(cps[t - 1]["pre"]))[i]
        toks, q = next_topk(cps[t - 1]["pre"][i])
        if sample_token(toks, q, u) != cps[t]["tk"][i]:
            return False
    return True

