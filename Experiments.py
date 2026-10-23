"""
Experiments.py - runs GP.py several times per setting and saves charts + statistics.

    python3 Experiments.py 1          # dataset 1 (also 2 or 3)
    python3 Experiments.py compare    # after running each dataset: compare them

Output goes to charts/.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

import GP as gp

# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
BASELINE_SEEDS = list(range(1, 11))   # 10 runs of the main setting (speed + variation)
COMPARISON_SEEDS = [1, 2, 3, 4, 5]    # 5 runs of each other setting
TARGET_ERROR = {1: None, 2: None, 3: None}   # None = automatic (see find_target)

# each setting changes one thing from the baseline
CONFIGS = {
    "Baseline":        dict(tournament=True,  elitism=True,  penalty=True),
    "No elitism":      dict(tournament=True,  elitism=False, penalty=True),
    "Roulette":        dict(tournament=False, elitism=True,  penalty=True),
    "No size penalty": dict(tournament=True,  elitism=True,  penalty=False),
}

OUT_DIR = "charts"
COLORS = {"Baseline": "#2a78d6", "No elitism": "#eb6834",
          "Roulette": "#1baf7a", "No size penalty": "#eda100"}
DATASET_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
INK, MUTED = "#0b0b0b", "#898781"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#c3c2b7",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e1e0d9",
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "legend.frameon": False,
})


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------
def test_predictions(best, X, y):
    with np.errstate(all="ignore"):
        pred = best.coefficient * best.evaluate(X) + best.shift
    rmse = np.sqrt(np.mean((y - pred) ** 2)) if np.all(np.isfinite(pred)) else np.inf
    return rmse, pred


def run_all(X_train, y_train, X_test, y_test):
    rows, finals, models = [], [], {}
    for name, cfg in CONFIGS.items():
        for seed in (BASELINE_SEEDS if name == "Baseline" else COMPARISON_SEEDS):
            best, history = gp.symbolic_regression(X_train, y_train, seed=seed, verbose=False, **cfg)
            rmse, _ = test_predictions(best, X_test, y_test)
            rows += [dict(config=name, seed=seed, **h) for h in history]
            finals.append(dict(config=name, seed=seed, train_error=best.error, test_rmse=rmse,
                               size=best.size(), formula=best.to_infix(),
                               coefficient=best.coefficient, shift=best.shift,
                               seconds=history[-1]["seconds"]))
            models[(name, seed)] = best
            print(f"{name:16s} seed {seed:2d}: train {best.error:.4g}, test RMSE {rmse:.4g}, "
                  f"size {best.size()}, {history[-1]['seconds']:.0f}s")
    return pd.DataFrame(rows), pd.DataFrame(finals), models


# ---------------------------------------------------------------------------
# convergence speed and variation
# ---------------------------------------------------------------------------
def find_target(final):
    """Training error that counts as 'found the answer'. Automatic: within 5% of the
    training error of the baseline run with the best TEST error (using the lowest
    training error instead would reward runs that overfit)."""
    if TARGET_ERROR.get(gp.DATASET) is not None:
        return TARGET_ERROR[gp.DATASET]
    base = final[(final.config == "Baseline") & np.isfinite(final.test_rmse)]
    return base.loc[base.test_rmse.idxmin(), "train_error"] * 1.05


def speed_table(runs, target):
    """Per run: generation it reached the target (NaN = never) and when it stopped improving."""
    rows = []
    for (config, seed), run in runs.groupby(["config", "seed"]):
        final_err = run.best_ever_error.iloc[-1]
        hit = run[run.best_ever_error <= target]
        plateau = run[run.best_ever_error <= final_err * 1.01]
        rows.append(dict(config=config, seed=seed,
                         gens_to_target=hit.generation.iloc[0] if len(hit) else np.nan,
                         seconds_to_target=hit.seconds.iloc[0] if len(hit) else np.nan,
                         gens_to_plateau=plateau.generation.iloc[0]))
    return pd.DataFrame(rows)


def summarize(final, speed):
    """Per setting: success rate plus median and spread across runs."""
    merged = final.merge(speed, on=["config", "seed"])
    out = []
    for config in CONFIGS:
        g = merged[merged.config == config]
        te = g.train_error.replace(np.inf, np.nan).dropna()
        tr = g.test_rmse.replace(np.inf, np.nan).dropna()
        out.append(dict(
            config=config, runs=len(g),
            success_rate=g.gens_to_target.notna().mean(),
            gens_to_target_median=g.gens_to_target.median(),
            gens_to_target_iqr=g.gens_to_target.quantile(0.75) - g.gens_to_target.quantile(0.25),
            gens_to_plateau_median=g.gens_to_plateau.median(),
            train_error_median=te.median(),
            train_error_cv=te.std() / te.mean() if len(te) > 1 and te.mean() > 0 else np.nan,
            test_rmse_median=tr.median(), test_rmse_min=tr.min(), test_rmse_max=tr.max(),
            size_median=g["size"].median(), seconds_median=g.seconds.median()))
    return pd.DataFrame(out)


def print_summary(summary, target):
    print(f"\nTarget training error: {target:.4g}")
    print(f"{'setting':16s} {'runs':>4s} {'success':>8s} {'gens to target':>15s} "
          f"{'gens to plateau':>16s} {'train cv':>9s} {'test RMSE median [min, max]':>30s} {'sec':>5s}")
    for _, r in summary.iterrows():
        print(f"{r.config:16s} {r.runs:4d} {100 * r.success_rate:7.0f}% "
              f"{r.gens_to_target_median:9.0f} ±{r.gens_to_target_iqr / 2:3.0f} "
              f"{r.gens_to_plateau_median:16.0f} {r.train_error_cv:9.2f} "
              f"{r.test_rmse_median:14.4g} [{r.test_rmse_min:.3g}, {r.test_rmse_max:.3g}] "
              f"{r.seconds_median:5.0f}")
    print("  ± = half the interquartile range; train cv = std / mean of final training error")


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------
def save(fig, name):
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"dataset{gp.DATASET}_{name}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("saved", path)


def line_chart(runs, configs, column, ylabel, title, name, log=True):
    """Median across runs (line) and middle 50% of runs (band) for each setting."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for c in sorted(configs, key=lambda c: c == "Baseline"):       # Baseline drawn on top
        g = runs[runs.config == c].groupby("generation")[column]
        ax.fill_between(g.median().index, g.quantile(0.25), g.quantile(0.75),
                        color=COLORS[c], alpha=0.15, linewidth=0)
        ax.plot(g.median().index, g.median(), color=COLORS[c], label=c,
                linestyle="-" if c == "Baseline" else (0, (5, 3)))
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(c) for c in configs]
    ax.legend([handles[i] for i in order], configs)
    if log:
        ax.set_yscale("log")
    ax.set(xlabel="Generation", ylabel=ylabel, title=title)
    save(fig, name)


def convergence_chart(runs):
    """Every baseline run's best-so-far error: shows speed and run-to-run spread."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    base = runs[runs.config == "Baseline"]
    for _, run in base.groupby("seed"):
        ax.plot(run.generation, run.best_ever_error, color=COLORS["Baseline"], alpha=0.5, linewidth=1.5)
    med = base.groupby("generation").best_ever_error.median()
    ax.plot(med.index, med, color=INK, label="Median of runs")
    ax.set_yscale("log")
    ax.set(xlabel="Generation", ylabel="Best training error (log scale)",
           title=f"Convergence across {base.seed.nunique()} runs")
    ax.legend()
    save(fig, "convergence")


def time_to_target_chart(speed):
    """Share of runs that have found the answer by each generation."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    gens = np.arange(gp.GENERATIONS)
    for c in sorted(CONFIGS, key=lambda c: c == "Baseline"):
        g = speed[speed.config == c]
        ax.step(gens, [(g.gens_to_target <= k).mean() for k in gens], where="post",
                color=COLORS[c], label=f"{c} ({len(g)} runs)",
                linestyle="-" if c == "Baseline" else (0, (5, 3)))
    handles, labels = ax.get_legend_handles_labels()
    order = [next(i for i, l in enumerate(labels) if l.startswith(c)) for c in CONFIGS]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower right")
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set(xlabel="Generation", ylabel="Runs that reached the target",
           title="How quickly runs reach the target error")
    save(fig, "time_to_target")


def per_run_chart(values, ylabel, title, name, log=False, never_y=None):
    """One dot per run for each setting, bar = median. NaN values are drawn as
    hollow dots at never_y (runs that never reached the target)."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    rng = np.random.default_rng(0)
    for i, c in enumerate(CONFIGS):
        v = values[values.config == c].value.replace(np.inf, np.nan)
        ok = v.dropna()
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(ok)), ok, s=60, color=COLORS[c],
                   edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
        if never_y is not None and v.isna().any():
            n = v.isna().sum()
            ax.scatter(i + rng.uniform(-0.12, 0.12, n), np.full(n, never_y), s=60,
                       facecolor="none", edgecolor=COLORS[c], linewidth=1.5, zorder=3)
        if len(ok):
            ax.plot([i - 0.25, i + 0.25], [ok.median()] * 2, color=INK, zorder=4)
    if never_y is not None:
        ax.axhline(gp.GENERATIONS, color=MUTED, linewidth=1, linestyle="--")
        ax.text(len(CONFIGS) - 0.5, never_y, "never reached", color=MUTED, ha="right", va="center")
    if log:
        ax.set_yscale("log")
    ax.set_xticks(range(len(CONFIGS)), list(CONFIGS))
    ax.grid(axis="x", visible=False)
    ax.set(ylabel=ylabel, title=title)
    save(fig, name)


def model_charts(best, X, y):
    """Predicted vs. true, and residuals (dataset 2: relative error vs x3)."""
    _, pred = test_predictions(best, X, y)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(y, pred, s=10, color=COLORS["Baseline"], alpha=0.5, linewidth=0)
    lo, hi = min(y.min(), np.nanmin(pred)), max(y.max(), np.nanmax(pred))
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1.2, linestyle="--", label="Perfect prediction")
    if gp.DATASET == 2:
        ax.set(xscale="log", yscale="log")
    ax.set(xlabel="True f (test set)", ylabel="Predicted f", title="Best model on held-out data")
    ax.legend(loc="upper left")
    save(fig, "pred_vs_actual")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    if gp.DATASET == 2:
        ax.scatter(X["x3"], 100 * (pred - y) / np.maximum(y, 0.005), s=10,
                   color=COLORS["Baseline"], alpha=0.5, linewidth=0)
        ax.set(xscale="log", xlabel="x3 (log scale)", ylabel="Relative error (%)",
               title="Prediction error grows as x3 gets small")
    else:
        ax.scatter(X["x"], pred - y, s=10, color=COLORS["Baseline"], alpha=0.5, linewidth=0)
        ax.set(xlabel="x", ylabel="Prediction − true f", title="Residuals on the test set")
    ax.axhline(0, color=MUTED, linewidth=1)
    save(fig, "residuals")


# ---------------------------------------------------------------------------
# compare datasets
# ---------------------------------------------------------------------------
def compare_datasets():
    found = [d for d in (1, 2, 3) if os.path.exists(f"{OUT_DIR}/speed_dataset{d}.csv")]
    if not found:
        print("Run Experiments.py for at least one dataset first.")
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    print(f"{'dataset':8s} {'runs':>4s} {'success':>8s} {'gens to target':>15s} "
          f"{'gens to plateau':>16s} {'train cv':>9s}")
    for d in found:
        speed = pd.read_csv(f"{OUT_DIR}/speed_dataset{d}.csv")
        speed = speed[speed.config == "Baseline"]
        gens = np.arange(pd.read_csv(f"{OUT_DIR}/runs_dataset{d}.csv").generation.max() + 1)
        ax.step(gens, [(speed.gens_to_target <= k).mean() for k in gens], where="post",
                color=DATASET_COLORS[d], label=f"Dataset {d} ({len(speed)} runs)")
        r = pd.read_csv(f"{OUT_DIR}/summary_dataset{d}.csv").iloc[0]      # Baseline row
        print(f"{d:<8d} {int(r.runs):4d} {100 * r.success_rate:7.0f}% "
              f"{r.gens_to_target_median:9.0f} ±{r.gens_to_target_iqr / 2:3.0f} "
              f"{r.gens_to_plateau_median:16.0f} {r.train_error_cv:9.2f}")
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set(xlabel="Generation", ylabel="Runs that reached the target", title="Convergence speed by dataset")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/compare_time_to_target.png", dpi=200)
    print("saved charts/compare_time_to_target.png")


# ---------------------------------------------------------------------------
def main(dataset):
    os.makedirs(OUT_DIR, exist_ok=True)
    X_train, y_train, X_test, y_test, _ = gp.initialize(dataset)
    print(f"Dataset {dataset}: {len(y_train)} training rows, {len(y_test)} test rows")

    runs, final, models = run_all(X_train, y_train, X_test, y_test)
    target = find_target(final)
    speed = speed_table(runs, target)
    summary = summarize(final, speed)
    for table, name in [(runs, "runs"), (final, "final"), (speed, "speed"), (summary, "summary")]:
        table.to_csv(f"{OUT_DIR}/{name}_dataset{dataset}.csv", index=False)
    print_summary(summary, target)

    convergence_chart(runs)
    line_chart(runs, ["Baseline", "No elitism"], "best_error",
               "Best error in generation (log scale)", "Effect of elitism", "elitism")
    line_chart(runs, ["Baseline", "Roulette"], "best_ever_error",
               "Best training error (log scale)", "Tournament vs. roulette selection", "selection")
    line_chart(runs, ["Baseline", "No size penalty"], "mean_size",
               "Mean tree size (nodes)", "Bloat: average tree size", "bloat", log=False)
    line_chart(runs, list(CONFIGS), "unique_fraction",
               "Unique formulas (share of population)", "Population diversity", "diversity", log=False)
    time_to_target_chart(speed)
    per_run_chart(speed.rename(columns={"gens_to_target": "value"}), "Generations to reach target",
                  "Convergence speed per run (bar = median)", "speed_spread",
                  never_y=gp.GENERATIONS * 1.08)
    tr = final.test_rmse.replace(np.inf, np.nan).dropna()
    per_run_chart(final.rename(columns={"test_rmse": "value"}), "Test RMSE",
                  "Test error of each run's best model (bar = median)", "final_test_error",
                  log=len(tr) > 0 and tr.max() / max(tr.min(), 1e-300) > 20)

    # the assignment's procedure: overall best model = baseline run with the lowest test error
    base = final[final.config == "Baseline"]
    best_seed = int(base.loc[base.test_rmse.idxmin(), "seed"])
    best = models[("Baseline", best_seed)]
    model_charts(best, X_test, y_test)
    print(f"\nBest model (baseline seed {best_seed}): "
          f"{best.coefficient:.6g} * {best.to_infix()} + {best.shift:.6g}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else str(gp.DATASET)
    if arg == "compare":
        compare_datasets()
    else:
        main(int(arg))