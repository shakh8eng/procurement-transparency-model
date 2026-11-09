"""
Transparency, Entry and Prices in Public Procurement: A Simulated Auction Model for Central Asia
Shohrukh Tojidinov, ECON8001 Independent Research Paper, Marmara University

Simulation code for Section 5 (Progress 1).

Model:
  costs c ~ Uniform[c_L, c_H], first-price procurement auction (lowest bid wins)
  bid:            b(c)  = c + (c_H - c) / n
  expected price: P(n)  = c_L + 2 (c_H - c_L) / (n + 1)
  expected profit: pi(n) = (c_H - c_L) / (n (n + 1))
  entry cost:     k(tau) = k0 (1 - theta * tau)
  entry:          n* = largest whole n with pi(n) >= k(tau)   (integer version)
                  pi(n) = k(tau)                               (continuous version, used for calibration)

Run:  python simulation.py
Outputs: figure1.pdf, figure1.png, results_tau.csv, printed tables
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

C_H = 100.0


# ---------- model formulas ----------
def bid(c, n, c_L=60.0, c_H=C_H):
    return c + (c_H - c) / n


def price(n, c_L, c_H=C_H):
    return c_L + 2 * (c_H - c_L) / (n + 1)


def profit(n, c_L, c_H=C_H):
    return (c_H - c_L) / (n * (n + 1))


def winner_cost(n, c_L, c_H=C_H):
    return c_L + (c_H - c_L) / (n + 1)


def entry_cost(tau, k0, theta):
    return k0 * (1 - theta * tau)


def n_integer(k, c_L, c_H=C_H):
    """Largest whole n with pi(n) >= k (at least one firm)."""
    n = 1
    while profit(n + 1, c_L, c_H) >= k:
        n += 1
    return n


def n_continuous(tau, n_N, theta):
    """Solve pi(n) = k(tau) with k0 calibrated to n_N: n(n+1) = n_N(n_N+1)/(1 - theta*tau)."""
    x = n_N * (n_N + 1) / (1 - theta * tau)
    return (-1 + np.sqrt(1 + 4 * x)) / 2


# ---------- calibration ----------
def calibrate(n_N, n_D, c_L, c_H=C_H):
    k0 = (c_H - c_L) / (n_N * (n_N + 1))
    theta = 1 - n_N * (n_N + 1) / (n_D * (n_D + 1))
    return k0, theta


# ---------- Monte Carlo check ----------
def monte_carlo(n, c_L, reps=200_000, seed=1):
    rng = np.random.default_rng(seed)
    costs = rng.uniform(c_L, C_H, size=(reps, n))
    bids = bid(costs, n, c_L)
    winner = bids.argmin(axis=1)
    paid = bids.min(axis=1)
    firm0_profit = np.where(winner == 0, paid - costs[:, 0], 0.0)
    return paid.mean(), firm0_profit.mean()


def main():
    # 1) Monte Carlo check of the formulas
    print("Monte Carlo check (c_L = 60, 200,000 tenders each)")
    print(f"{'n':>3} {'P sim':>8} {'P formula':>10} {'pi sim':>8} {'pi formula':>11}")
    max_gap = 0.0
    for n in range(2, 7):
        p_sim, pi_sim = monte_carlo(n, 60.0)
        p_f, pi_f = price(n, 60.0), profit(n, 60.0)
        max_gap = max(max_gap, abs(p_sim - p_f))
        print(f"{n:>3} {p_sim:8.2f} {p_f:10.2f} {pi_sim:8.3f} {pi_f:11.3f}")
    print(f"largest price gap: {max_gap:.3f}\n")

    # 2) Illustrative example from Section 3 (integer entry)
    print("Section 3 example: c_L=60, k0=5, theta=0.76 (integer entry)")
    print(f"{'tau':>5} {'k':>6} {'n*':>3} {'P':>7} {'winner cost':>12} {'entry costs':>12} {'firm profit':>12}")
    for tau in [0, 0.25, 0.5, 0.75, 1.0]:
        k = entry_cost(tau, 5, 0.76)
        n = n_integer(k, 60.0)
        print(f"{tau:5.2f} {k:6.2f} {n:3d} {price(n, 60):7.2f} {winner_cost(n, 60):12.2f} "
              f"{n * k:12.2f} {n * (profit(n, 60) - k):12.2f}")
    print()

    # 3) Calibration illustration: Kazakhstan national tenders 2021 vs 2024
    n_N, n_D = 3.16, 4.2
    k0_60, theta = calibrate(n_N, n_D, 60.0)
    print(f"Calibration (Kazakhstan 2021 -> 2024): theta = {theta:.3f}, k0 (c_L=60) = {k0_60:.3f}\n")

    taus = np.linspace(0, 1, 101)
    n_path = n_continuous(taus, n_N, theta)
    c_L_values = [50.0, 60.0, 70.0]

    print("Results along tau (c_L = 60)")
    print(f"{'tau':>5} {'bidders':>8} {'price':>7} {'saving %':>9}")
    rows = []
    p0 = price(n_N, 60.0)
    for tau in [0, 0.25, 0.5, 0.75, 1.0]:
        n = n_continuous(tau, n_N, theta)
        p = price(n, 60.0)
        print(f"{tau:5.2f} {n:8.2f} {p:7.2f} {100 * (p0 - p) / p0:9.2f}")
    print()

    print("Sensitivity: price at tau=0 and tau=1 for different c_L")
    print(f"{'c_L':>5} {'P(0)':>7} {'P(1)':>7} {'saving %':>9}")
    for c_L in c_L_values:
        a, b = price(n_N, c_L), price(n_D, c_L)
        print(f"{c_L:5.0f} {a:7.2f} {b:7.2f} {100 * (a - b) / a:9.2f}")

    with open("results_tau.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau", "bidders"] + [f"price_cL{int(c)}" for c in c_L_values])
        for tau, n in zip(taus, n_path):
            w.writerow([round(tau, 2), round(n, 4)] + [round(price(n, c), 4) for c in c_L_values])

    # 4) Figure 1
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))

    ax1.plot(taus, n_path, color=colors[0], lw=2)
    ax1.set_xlabel(r"Transparency $\tau$")
    ax1.set_ylabel("Number of bidders")
    ax1.set_title("(a) Number of bidders", loc="left", fontsize=10, color=ink)

    for c_L, col in zip(c_L_values, colors):
        p = price(n_path, c_L)
        ax2.plot(taus, p, color=col, lw=2, label=f"$c_L$ = {int(c_L)}")
        ax2.annotate(f"$c_L$ = {int(c_L)}", (1.0, p[-1]), xytext=(4, 0), textcoords="offset points",
                     va="center", fontsize=9, color=ink)
    ax2.set_xlabel(r"Transparency $\tau$")
    ax2.set_ylabel("Expected price (% of $c_H$)")
    ax2.set_title("(b) Expected price", loc="left", fontsize=10, color=ink)
    ax2.set_xlim(0, 1.18)
    ax2.legend(frameon=False, fontsize=8, loc="lower left")

    for ax in (ax1, ax2):
        ax.grid(True, color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    ax1.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig("figure1.pdf")
    fig.savefig("figure1.png", dpi=200)
    print("\nSaved figure1.pdf, figure1.png, results_tau.csv")


if __name__ == "__main__":
    main()
