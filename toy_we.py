import hashlib, math
import numpy as np

T, P_UP, THRESH = 30, 0.3, 12

def exact_prob():
    need = math.ceil((THRESH + T) / 2)
    return sum(math.comb(T, k) * P_UP**k * (1 - P_UP)**(T - k) for k in range(need, T + 1))

def step_rng(root: bytes, step: int, tag: str = "move") -> np.random.Generator:
    digest = hashlib.sha256(root + f"|{tag}|{step}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "big"))

def monte_carlo(n, root: bytes):
    rng = step_rng(root, 0, "mc")
    ups = rng.random((n, T)) < P_UP
    final = ups.sum(axis=1) * 2 - T
    return float(np.mean(final >= THRESH))

def weighted_ensemble(n_walkers, root: bytes, per_bin=4):
    moves = 0
    pos = np.zeros(n_walkers, dtype=np.int64)
    w = np.full(n_walkers, 1.0 / n_walkers)
    for t in range(T):
        rng = step_rng(root, t, "move")
        moves += len(pos)
        pos = pos + np.where(rng.random(len(pos)) < P_UP, 1, -1)
        rs = step_rng(root, t, "resample")
        new_pos, new_w = [], []
        for b in np.unique(pos):
            idx = np.where(pos == b)[0]
            total = w[idx].sum()
            kids = rs.choice(idx, size=per_bin, p=w[idx] / total)
            new_pos.extend(pos[kids]); new_w.extend([total / per_bin] * per_bin)
        pos, w = np.array(new_pos), np.array(new_w)
    return float(w[pos >= THRESH].sum()), float(w.sum()), moves

if __name__ == "__main__":
    truth = exact_prob()
    print(f"exact probability   : {truth:.3e}")
    runs = 400
    res = [weighted_ensemble(64, f"we-{i}".encode()) for i in range(runs)]
    we = [r[0] for r in res]
    budget = int(np.mean([r[2] for r in res]))
    n_mc = budget // T
    print(f"budget (moves/run)  : {budget}  -> MC uses {n_mc} paths")
    mc = [monte_carlo(n_mc, f"mc-{i}".encode()) for i in range(runs)]
    we_mean, we_se = np.mean(we), np.std(we, ddof=1) / math.sqrt(runs)
    print(f"WE mean             : {we_mean:.3e} +/- {we_se:.1e} (z vs truth = {(we_mean-truth)/we_se:+.2f})")
    print(f"WE relative std/run : {np.std(we, ddof=1)/truth:.2f}")
    print(f"MC runs with zero hits: {sum(m == 0 for m in mc)} of {runs}")
    print(f"WE runs with zero hits: {sum(x == 0 for x in we)} of {runs}")
    print(f"mass conserved      : {weighted_ensemble(64, b'check')[1]:.12f}")

