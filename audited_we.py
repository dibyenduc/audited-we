import hashlib, json, math, random, sys
import numpy as np
from toy_we import T, P_UP, THRESH, step_rng, exact_prob

def H(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()

def make_config(n_walkers=64, per_bin=4, thresh=THRESH, label="run"):
    return {"T": T, "p_up": P_UP, "thresh": thresh, "n_walkers": n_walkers,
            "per_bin": per_bin, "label": label}

def commit(config) -> bytes:
    return H(json.dumps(config, sort_keys=True).encode())

def we_step(pos, w, root, t, per_bin):
    rng = step_rng(root, t, "move")
    pos = pos + np.where(rng.random(len(pos)) < P_UP, 1, -1)
    rs = step_rng(root, t, "resample")
    new_pos, new_w = [], []
    for b in np.unique(pos):
        idx = np.where(pos == b)[0]
        total = w[idx].sum()
        kids = rs.choice(idx, size=per_bin, p=w[idx] / total)
        new_pos.extend(pos[kids]); new_w.extend([total / per_bin] * per_bin)
    return np.array(new_pos, dtype=np.int64), np.array(new_w)

def encode(t, pos, w):
    return {"t": t, "pos": [int(x) for x in pos], "w": [float(x).hex() for x in w]}

def decode(cp):
    return np.array(cp["pos"], dtype=np.int64), np.array([float.fromhex(x) for x in cp["w"]])

def leaf(cp) -> bytes:
    return H(json.dumps(cp, sort_keys=True).encode())

def merkle_levels(leaves):
    levels = [list(leaves)]
    while len(levels[-1]) > 1:
        cur = levels[-1]
        if len(cur) % 2: cur = cur + [cur[-1]]
        levels.append([H(cur[i] + cur[i + 1]) for i in range(0, len(cur), 2)])
    return levels

def merkle_proof(levels, i):
    proof = []
    for lev in levels[:-1]:
        lev = lev + [lev[-1]] if len(lev) % 2 else lev
        proof.append((lev[i ^ 1], i % 2 == 0))
        i //= 2
    return proof

def merkle_check(leaf_hash, proof, root):
    h = leaf_hash
    for sib, sib_on_right in proof:
        h = H(h + sib) if sib_on_right else H(sib + h)
    return h == root

def estimate(cp, thresh):
    pos, w = decode(cp)
    return float(w[pos >= thresh].sum())

def run_and_log(config, start_t=1, start_state=None, cps=None):
    root = commit(config)
    if cps is None:
        n = config["n_walkers"]
        pos, w = np.zeros(n, dtype=np.int64), np.full(n, 1.0 / n)
        cps = [encode(0, pos, w)]
    else:
        pos, w = decode(cps[-1])
    for t in range(len(cps), config["T"] + 1):
        pos, w = we_step(pos, w, root, t, config["per_bin"])
        cps.append(encode(t, pos, w))
    return cps

def publish(config, cps, claimed=None):
    levels = merkle_levels([leaf(c) for c in cps])
    est = estimate(cps[-1], config["thresh"]) if claimed is None else claimed
    return {"config": config, "cps": cps, "merkle_root": levels[-1][0], "claimed": est}

def verify(tr, expected_commitment, k_random, vrng):
    cfg, cps = tr["config"], tr["cps"]
    if commit(cfg) != expected_commitment:
        return False, "config does not match published commitment"
    root = expected_commitment
    levels = merkle_levels([leaf(c) for c in cps])
    if levels[-1][0] != tr["merkle_root"]:
        return False, "merkle root mismatch"
    for c in cps:
        if abs(sum(float.fromhex(x) for x in c["w"]) - 1.0) > 1e-9:
            return False, f"mass not conserved at step {c['t']}"
    if abs(estimate(cps[-1], cfg["thresh"]) - tr["claimed"]) > 0:
        return False, "claimed estimate differs from recomputation"
    steps = [cfg["T"]] + vrng.sample(range(1, cfg["T"]), k_random)
    for t in steps:
        i = t
        for j in (i - 1, i):
            if not merkle_check(leaf(cps[j]), merkle_proof(levels, j), tr["merkle_root"]):
                return False, f"merkle proof failed at {j}"
        pos, w = decode(cps[t - 1])
        npos, nw = we_step(pos, w, root, t, cfg["per_bin"])
        gp, gw = decode(cps[t])
        if not (np.array_equal(npos, gp) and np.array_equal(nw, gw)):
            return False, f"replay mismatch at step {t}"
    return True, "ok"

def attack_reweight(config, tamper_t, boost=10.0):
    cps = run_and_log(config)
    pos, w = decode(cps[tamper_t])
    top = pos >= np.sort(pos)[-max(1, len(pos) // 8)]
    w2 = w.copy(); w2[top] *= boost; w2 /= w2.sum()
    cps2 = cps[:tamper_t] + [encode(tamper_t, pos, w2)]
    return publish(config, run_and_log(config, cps=cps2))

def attack_forged_estimate(config):
    cps = run_and_log(config)
    return publish(config, cps, claimed=estimate(cps[-1], config["thresh"]) * 3 + 1e-6)

def attack_threshold_swap(config):
    cfg2 = dict(config, thresh=config["thresh"] - 2)
    return publish(cfg2, run_and_log(cfg2))

def attack_seed_reroll(config, tries=20):
    best = None
    for i in range(tries):
        alt = dict(config, label=f"{config['label']}-reroll{i}")
        cps = run_and_log(alt)
        e = estimate(cps[-1], alt["thresh"])
        if best is None or e > best[0]: best = (e, cps)
    return publish(config, best[1])

if __name__ == "__main__":
    truth = exact_prob()
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    ks = [0, 1, 2, 4, 8, 16, 28]
    vrng = random.Random(12345)
    honest_ok = 0
    biases = []
    det = {k: 0 for k in ks}
    others = {"forged estimate": 0, "threshold swap": 0, "seed re-roll": 0}
    for i in range(trials):
        cfg = make_config(label=f"trial-{i}")
        c = commit(cfg)
        ok, _ = verify(publish(cfg, run_and_log(cfg)), c, 2, vrng)
        honest_ok += ok
        tr = attack_reweight(cfg, tamper_t=10)
        biases.append(tr["claimed"] / truth)
        for k in ks:
            det[k] += not verify(tr, c, k, vrng)[0]
        others["forged estimate"] += not verify(attack_forged_estimate(cfg), c, 0, vrng)[0]
        others["threshold swap"] += not verify(attack_threshold_swap(cfg), c, 0, vrng)[0]
        others["seed re-roll"] += not verify(attack_seed_reroll(cfg, 5), c, 0, vrng)[0]
    print(f"honest transcripts accepted (k=2): {honest_ok}/{trials}")
    for name, n in others.items():
        print(f"attack {name:16s} detected: {n}/{trials}  (k=0 random checks)")
    print(f"mean (attacked estimate / truth): {np.mean(biases):.2f}")
    print("mid-run reweight attack (tamper at step 10, honest continuation)")
    print("  k random steps | detected | theory k/(T-1)")
    for k in ks:
        print(f"  {k:14d} | {det[k]/trials:7.1%} | {k/(T-1):7.1%}")

