import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 1491
ms = np.array([0, 25, 50, 100, 200, 400, 800])
obs = {
    1: [0, 2, 3, 6, 14, 26, 56],
    2: [0, 0, 6, 7, 26, 45, 69],
    4: [0, 4, 9, 23, 47, 74, 97],
    8: [0, 22, 32, 36, 72, 92, 100],
    16: [0, 23, 43, 70, 93, 100, 100],
}
bias = {1: 0.77, 2: 0.88, 4: 1.06, 8: 1.70, 16: 1.91}
grid = np.linspace(0, 800, 200)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for (x, y), c in zip(obs.items(), plt.cm.viridis(np.linspace(0, 0.9, len(obs)))):
    ax[0].plot(ms, y, "o", color=c)
    ax[0].plot(grid, 100 * (1 - (1 - x / N) ** grid), "-", color=c, label=f"x={x} forged")
ax[0].set_xlabel("m = model-move cells replayed (of about 1491)")
ax[0].set_ylabel("detection rate (%)")
ax[0].set_title("Detection: dots measured, lines 1-(1-x/N)^m")
ax[0].legend()
ax[0].grid(alpha=0.3)

xs = list(bias)
ax[1].bar([str(x) for x in xs], [bias[x] for x in xs], color="tab:orange")
ax[1].axhline(0.89, color="gray", ls="--", label="honest WE mean about 0.89")
ax[1].set_xlabel("x = forged moves at step 10")
ax[1].set_ylabel("mean estimate / true probability")
ax[1].set_title("Bias the attacker gains")
ax[1].legend()
plt.tight_layout()
plt.savefig("output/detection_vs_bias.png", dpi=150)

