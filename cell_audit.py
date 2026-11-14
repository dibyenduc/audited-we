import hashlib, json, math, random, sys
import numpy as np
from toy_we import T, P_UP, THRESH, step_rng, exact_prob

def H(b): return hashlib.sha256(b).digest()
def make_config(n_walkers=64, per_bin=4, thresh=THRESH, label="run"):
    return {"T": T, "p_up": P_UP, "thresh": thresh, "n_walkers": n_walkers, "per_bin": per_bin, "label": label}
def commit(cfg): return H(json.dumps(cfg, sort_keys=True).encode())
def hx(a): return [float(x).hex() for x in a]
def unhx(a): return np.array([float.fromhex(x) for x in a])

def we_step(pos, w, root, t, pb, fake=0):
    draws = step_rng(root, t, "move").random(len(pos))
    mv = pos + np.where(draws < P_UP, 1, -1)
    if fake:
        down = np.where(draws >= P_UP)[0]
        pick = down[np.argsort(-pos[down], kind="stable")[:fake]]
        mv[pick] += 2
    new_pos, new_w, par = [], [], []
    for b in np.unique(mv):
        idx = np.where(mv == b)[0]
        total = w[idx].sum()
        kids = step_rng(root, t, f"resample-{b}").choice(idx, size=pb, p=w[idx] / total)
        new_pos += [int(b)] * pb; new_w += [total / pb] * pb; par += [int(k) for k in kids]
    return mv, np.array(par), np.array(new_pos, dtype=np.int64), np.array(new_w)

def run_log(cfg, fake_at=None, fake=0, boost_at=None):
    root, pb, n = commit(cfg), cfg["per_bin"], cfg["n_walkers"]
    pos, w = np.zeros(n, dtype=np.int64), np.full(n, 1.0 / n)
    cps = [{"t": 0, "mv": [], "par": [], "pos": [int(x) for x in pos], "w": hx(w)}]
    for t in range(1, cfg["T"] + 1):
        mv, par, npos, nw = we_step(pos, w, root, t, pb, fake if t == fake_at else 0)
        if t == boost_at:
            top = npos >= np.sort(npos)[-max(1, len(npos) // 8)]
            nw = nw.copy(); nw[top] *= 10.0; nw /= nw.sum()
        cps.append({"t": t, "mv": [int(x) for x in mv], "par": [int(x) for x in par],
                    "pos": [int(x) for x in npos], "w": hx(nw)})
        pos, w = npos, nw
    return cps

def leaf(cp): return H(json.dumps(cp, sort_keys=True).encode())
def merkle_root(leaves):
    cur = list(leaves)
    while len(cur) > 1:
        if len(cur) % 2: cur.append(cur[-1])
        cur = [H(cur[i] + cur[i + 1]) for i in range(0, len(cur), 2)]
    return cur[0]

def estimate(cp, thresh):
    return float(unhx(cp["w"])[np.array(cp["pos"]) >= thresh].sum())

def publish(cfg, cps, claimed=None):
    est = estimate(cps[-1], cfg["thresh"]) if claimed is None else claimed
    return {"config": cfg, "cps": cps, "root": merkle_root([leaf(c) for c in cps]), "claimed": est}

def check_bookkeeping(tr, expected_commitment):
    cfg, cps = tr["config"], tr["cps"]
    if commit(cfg) != expected_commitment: return False, "config != published commitment"
    if merkle_root([leaf(c) for c in cps]) != tr["root"]: return False, "merkle root mismatch"
    root, pb = expected_commitment, cfg["per_bin"]
    for t in range(1, cfg["T"] + 1):
        prev, cur = cps[t - 1], cps[t]
        pw, mv = unhx(prev["w"]), np.array(cur["mv"])
        if len(mv) != len(pw): return False, f"step {t}: wrong number of moves"
        exp_pos, exp_w, exp_par = [], [], []
        for b in np.unique(mv):
            idx = np.where(mv == b)[0]
            total = pw[idx].sum()
            kids = step_rng(root, t, f"resample-{b}").choice(idx, size=pb, p=pw[idx] / total)
            exp_pos += [int(b)] * pb; exp_w += [total / pb] * pb; exp_par += [int(k) for k in kids]
        if cur["pos"] != exp_pos or cur["par"] != exp_par or unhx(cur["w"]).tolist() != exp_w:
            return False, f"step {t}: weight/parent bookkeeping mismatch"
    if estimate(cps[-1], cfg["thresh"]) != tr["claimed"]: return False, "claimed estimate wrong"
    return True, "ok"

def check_cells(tr, m, vrng):
    cfg, cps = tr["config"], tr["cps"]
    root = commit(cfg)
    cells = [(t, i) for t in range(1, cfg["T"] + 1) for i in range(len(cps[t - 1]["pos"]))]
    for t, i in vrng.sample(cells, min(m, len(cells))):
        d = step_rng(root, t, "move").random(len(cps[t - 1]["pos"]))[i]
        if cps[t]["mv"][i] != cps[t - 1]["pos"][i] + (1 if d < P_UP else -1):
            return False, f"model replay mismatch at step {t}, walker {i}"
    return True, "ok"

if __name__ == "__main__":
    truth = exact_prob()
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    ms = [0, 25, 50, 100, 200, 400, 800]
    xs = [1, 2, 4, 8, 16]
    vrng = random.Random(777)
    honest = weight_caught = 0
    bias = {x: [] for x in xs}
    det = {(x, m): 0 for x in xs for m in ms}
    ncells = []
    for i in range(trials):
        cfg = make_config(label=f"trial-{i}"); c = commit(cfg)
        h = publish(cfg, run_log(cfg))
        ncells.append(sum(len(cp["pos"]) for cp in h["cps"][:-1]))
        honest += check_bookkeeping(h, c)[0] and check_cells(h, 100, vrng)[0]
        weight_caught += not check_bookkeeping(publish(cfg, run_log(cfg, boost_at=10)), c)[0]
        for x in xs:
            tr = publish(cfg, run_log(cfg, fake_at=10, fake=x))
            bias[x].append(tr["claimed"] / truth)
            ok_book = check_bookkeeping(tr, c)[0]
            for m in ms:
                det[(x, m)] += (not ok_book) or (not check_cells(tr, m, vrng)[0])
    N = int(np.mean(ncells))
    print(f"honest accepted: {honest}/{trials}")
    print(f"weight-tamper (boost x10 at step 10) caught by bookkeeping alone: {weight_caught}/{trials}")
    print(f"model-move cells in the log (N): about {N}")
    print("fake-moves attack at step 10: x walkers get a forged up-move")
    print("  x  | mean est/truth | " + " | ".join(f"m={m:<4d}" for m in ms))
    for x in xs:
        print(f" {x:3d} | {np.mean(bias[x]):14.2f} | " + " | ".join(f"{det[(x,m)]/trials:6.0%}" for m in ms))

