"""Toy simulation of the MetaRSI-v1 scheduling mechanics (arXiv:2609.06396v1).

This is NOT a reproduction of the paper. The environment below is synthetic and
hand-designed; its only purpose is to make three ideas from Sections 3.1 and
4.3-4.4 concrete and runnable:

1. The improved object is a triple S = (D, theta, H).
2. Operator order is constrained: after any behaviour change (H or M), Model-RSI
   needs fresh data first, so H->M and M->M are inadmissible while D->M is fine.
3. The objective is cost-aware: J = dQ - lambda*C - mu*Reg - nu*Gap.

Three schedulers are compared under the same step budget:
    fixed   : D -> M -> H, repeated
    random  : uniformly random admissible operator
    greedy  : one-step lookahead on J per unit cost

Usage:
    uv run python paper/rsi/rsi_loop_sim.py
"""

from __future__ import annotations

import dataclasses
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
IMG_DIR = HERE / "images"

OPERATORS = ("D", "H", "M")
COST = {"D": 1.0, "H": 1.5, "M": 4.0}
LAMBDA, MU, NU = 0.02, 1.0, 0.5


@dataclasses.dataclass
class State:
    """S = (D, theta, H) plus bookkeeping the admissibility rule needs."""

    data: float = 0.0        # amount of usable, fresh training data
    theta: float = 0.0       # capability internalised in weights
    harness: float = 0.0     # capability carried by prompts/tools/memory
    needs_fresh_data: bool = True  # set by H or M, cleared by D
    base_skill: float = 0.0  # untouched capability, used for the regression term

    def copy(self) -> "State":
        return dataclasses.replace(self)


def quality(s: State) -> float:
    """Deployed accuracy in [0, 100]. Harness and weights are partially redundant."""
    raw = 0.55 * s.theta + 0.45 * s.harness - 0.15 * s.theta * s.harness
    return 100.0 * (1.0 - np.exp(-2.0 * raw))


def admissible(s: State, op: str) -> bool:
    if op == "M":
        return not s.needs_fresh_data and s.data > 0.05
    return True


def apply(s: State, op: str, rng: np.random.Generator) -> State:
    """One operator application with mild stochasticity."""
    n = s.copy()
    noise = rng.normal(0.0, 0.02)
    if op == "D":
        # Data yield grows with what the harness currently exposes.
        n.data += 0.25 + 0.2 * s.harness + noise
        n.needs_fresh_data = False
    elif op == "H":
        n.harness = min(1.0, s.harness + 0.18 * (1.0 - s.harness) + noise)
        n.needs_fresh_data = True
    elif op == "M":
        gain = 0.5 * min(s.data, 0.6)
        n.theta = min(1.0, s.theta + gain + noise)
        n.base_skill -= 0.02 * gain  # small forgetting on untouched skills
        n.data = 0.0                 # consumed; old data is now stale
        n.needs_fresh_data = True
    return n


def objective(s0: State, s1: State, op: str, rng: np.random.Generator) -> float:
    """J for a single step: gain minus cost, regression and feedback gap."""
    dq = quality(s1) - quality(s0)
    reg = max(0.0, s0.base_skill - s1.base_skill) * 100.0
    gap = abs(rng.normal(0.0, 0.5)) if op == "H" else abs(rng.normal(0.0, 0.2))
    return dq - LAMBDA * COST[op] - MU * reg - NU * gap


def choose_fixed(step: int, s: State, rng: np.random.Generator) -> str:
    plan = ("D", "M", "H")
    op = plan[step % 3]
    return op if admissible(s, op) else "D"


def choose_random(step: int, s: State, rng: np.random.Generator) -> str:
    legal = [op for op in OPERATORS if admissible(s, op)]
    return str(rng.choice(legal))


def choose_greedy(step: int, s: State, rng: np.random.Generator) -> str:
    best_op, best_val = "D", -np.inf
    for op in OPERATORS:
        if not admissible(s, op):
            continue
        probe = np.random.default_rng(rng.integers(1 << 31))
        val = objective(s, apply(s, op, probe), op, probe) / COST[op]
        if val > best_val:
            best_op, best_val = op, val
    return best_op


SCHEDULERS = {"fixed": choose_fixed, "random": choose_random, "greedy": choose_greedy}


def run(scheduler: str, steps: int, seed: int) -> tuple[np.ndarray, float, list[str]]:
    rng = np.random.default_rng(seed)
    s = State()
    curve = [quality(s)]
    total_cost, trace = 0.0, []
    for t in range(steps):
        op = SCHEDULERS[scheduler](t, s, rng)
        assert admissible(s, op), f"{scheduler} proposed inadmissible {op}"
        s = apply(s, op, rng)
        total_cost += COST[op]
        trace.append(op)
        curve.append(quality(s))
    return np.array(curve), total_cost, trace


def main(steps: int = 12, seeds: int = 50) -> None:
    IMG_DIR.mkdir(exist_ok=True)
    print(f"## Toy scheduler comparison ({steps} steps, {seeds} seeds)\n")
    print("| Scheduler | Final Q (mean ± sd) | Total cost | Q per cost | Example trace |")
    print("|---|---:|---:|---:|---|")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name in SCHEDULERS:
        curves, costs, trace0 = [], [], None
        for seed in range(seeds):
            c, cost, trace = run(name, steps, seed)
            curves.append(c)
            costs.append(cost)
            trace0 = trace0 or trace
        curves = np.stack(curves)
        final = curves[:, -1]
        print(
            f"| {name} | {final.mean():.1f} ± {final.std():.1f} | {np.mean(costs):.1f} "
            f"| {final.mean() / np.mean(costs):.2f} | {'→'.join(trace0)} |"
        )
        mean, sd = curves.mean(0), curves.std(0)
        ax.plot(mean, label=name)
        ax.fill_between(range(len(mean)), mean - sd, mean + sd, alpha=0.15)
    ax.set_xlabel("Step")
    ax.set_ylabel("Synthetic deployed quality Q")
    ax.set_title("Toy RSI loop: admissible-order schedulers")
    ax.legend()
    fig.tight_layout()
    out = IMG_DIR / "toy_scheduler.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nPlot written to {out}")
    print("\nReminder: the environment is synthetic. This shows the mechanics, not the paper's result.")


if __name__ == "__main__":
    main()
