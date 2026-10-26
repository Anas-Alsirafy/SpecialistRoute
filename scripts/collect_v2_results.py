# ═══════════════════════════════════════════════════════════════════════════════
# COLLECT ALL v2 RESULTS INTO ONE EXCEL FILE
# Reads only what the main notebook already wrote (no TensorFlow needed):
#   <output_dir>/<prefix>_<Component>/progress.json, *_seed*_history.csv, config.txt
#   <output_dir>/<prefix>_pipeline/pipeline_summary_seed*.json, theoretical_seed*.json,
#                                   pipeline_log_seed*.csv
#   <output_dir>/<prefix>_run_all_state.json
#   <dataset reports>/manifest.csv  (to split restored vs original test images)
# Run:  /usr/bin/python3 collect_v2_results.py
# ═══════════════════════════════════════════════════════════════════════════════
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

COLLECT = {
    "output_dir": "/home/auc/PlantDoc/Experiments",
    "prefix": "v2",
    "manifest": "/home/auc/PlantDoc/Datasets/reports/manifest.csv",
    "out_file": "/home/auc/PlantDoc/Experiments/v2_all_results.xlsx",
}
ORDER = ["Classifier", "Apple", "Bell_pepper", "Corn", "Potato", "Tomato", "Grape", "Rest"]

warnings = []


def ci95(values):
    v = np.asarray([x for x in values if pd.notna(x)], dtype=float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    if len(v) == 1 or v.std(ddof=1) == 0:
        return v.mean(), 0.0, v.mean(), v.mean(), len(v)
    lo, hi = stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=stats.sem(v))
    return v.mean(), v.std(ddof=1), lo, hi, len(v)


def summarize(df, group_cols, metric_cols):
    rows = []
    for key, g in df.groupby(group_cols, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        for m in metric_cols:
            mean, std, lo, hi, n = ci95(g[m])
            rows.append({**dict(zip(group_cols, key)), "metric": m, "mean": mean, "std": std,
                         "ci95_low": lo, "ci95_high": hi, "n_seeds": n, "values": list(np.round(g[m].tolist(), 4))})
    return pd.DataFrame(rows)


def history_stats(h):
    out = {"epochs_run": len(h)}
    if "val_loss" in h and h["val_loss"].notna().any():
        i = int(h["val_loss"].idxmin())
        out.update(hist_selected_epoch=i + 1, val_loss_min=h["val_loss"].min(),
                   val_acc_at_selected=h["val_accuracy"].iloc[i] if "val_accuracy" in h else np.nan,
                   train_acc_at_selected=h["accuracy"].iloc[i] if "accuracy" in h else np.nan)
    if "val_accuracy" in h:
        j = int(h["val_accuracy"].idxmax())
        out.update(best_val_acc=h["val_accuracy"].max(), best_val_acc_epoch=j + 1,
                   final_val_acc=h["val_accuracy"].iloc[-1],
                   val_acc_last20_mean=h["val_accuracy"].iloc[-20:].mean())
    if "lr" in h:
        lr0 = h["lr"].iloc[0]
        cut = h.index[h["lr"] < lr0 * 0.999]
        out.update(initial_lr=lr0, final_lr=h["lr"].iloc[-1], first_lr_cut_epoch=int(cut[0]) + 1 if len(cut) else None,
                   n_lr_cuts=int((h["lr"].diff() < 0).sum()))
    return out


def main(cfg):
    out_dir, prefix = cfg["output_dir"], cfg["prefix"]
    comp_dirs = {os.path.basename(d)[len(prefix) + 1:]: d for d in glob.glob(os.path.join(out_dir, f"{prefix}_*"))
                 if os.path.isdir(d) and not d.endswith((f"{prefix}_pipeline", f"{prefix}_superseded"))}
    comps = [c for c in ORDER if c in comp_dirs] + sorted(c for c in comp_dirs if c not in ORDER)
    print(f"[Collect] components found: {comps}")

    # ── per component × seed: final metrics + training-history statistics ───────
    per_seed, histories, configs = [], [], []
    for comp in comps:
        d = comp_dirs[comp]
        prog_path = os.path.join(d, "progress.json")
        if not os.path.exists(prog_path):
            warnings.append(f"{comp}: no progress.json")
            continue
        prog = json.load(open(prog_path))
        model = prog.get("model_name", "?")
        for r in prog["per_seed_results"]:
            row = {"component": comp, "model": model, **{k: v for k, v in r.items() if k != "checkpoint_path"}}
            hp = os.path.join(d, f"{model}_seed{r['seed']}_history.csv")
            if os.path.exists(hp):
                h = pd.read_csv(hp)
                row.update(history_stats(h))
                histories.append(h.assign(component=comp, seed=r["seed"]))
            else:
                warnings.append(f"{comp} seed {r['seed']}: no history file {os.path.basename(hp)}")
            per_seed.append(row)
        cp = os.path.join(d, "config.txt")
        if os.path.exists(cp):
            for line in open(cp):
                if ":" in line and line.startswith("  "):
                    k, v = line.strip().split(":", 1)
                    configs.append({"component": comp, "key": k.strip(), "value": v.strip()})
    per_seed = pd.DataFrame(per_seed)
    comp_metrics = [m for m in ["test_accuracy", "test_macro_f1", "test_balanced_accuracy", "val_accuracy"]
                    if m in per_seed.columns]
    comp_summary = summarize(per_seed, ["component"], comp_metrics) if len(per_seed) else pd.DataFrame()

    # ── pipeline per seed ────────────────────────────────────────────────────────
    pdir = os.path.join(out_dir, f"{prefix}_pipeline")
    restored = set()
    if os.path.exists(cfg["manifest"]):
        man = pd.read_csv(cfg["manifest"])
        restored = set(man.loc[man["status"].astype(str).str.startswith("restored"), "rel"])
    else:
        warnings.append("manifest not found — restored/original split skipped")
    pipe_rows, species_rows, logs = [], [], []
    for sp in sorted(glob.glob(os.path.join(pdir, "pipeline_summary_seed*.json"))):
        seed = int(sp.split("seed")[-1].split(".")[0])
        s = json.load(open(sp))
        row = {"seed": seed, "final_accuracy": s["final_accuracy"], "routing_accuracy": s["routing_accuracy"],
               "final_correct": s["final_correct"], "n": s["n"]}
        tp = os.path.join(pdir, f"theoretical_seed{seed}.json")
        if os.path.exists(tp):
            t = json.load(open(tp))
            row.update(theory_perfect_routing=t.get("perfect_routing"), theory_simple=t.get("simple"),
                       theory_class_aware=t.get("class_aware"))
        lp = os.path.join(pdir, f"pipeline_log_seed{seed}.csv")
        if os.path.exists(lp):
            log = pd.read_csv(lp)
            log["rel"] = log["image"].str.split("/test/").str[-1]
            log["restored"] = log["rel"].isin(restored)
            for flag, g in log.groupby("restored"):
                tag = "restored" if flag else "original"
                row[f"final_acc_{tag}"] = g["final_correct"].mean()
                row[f"n_{tag}"] = len(g)
            for (spc, flag), g in log.groupby(["true_species", "restored"]):
                species_rows.append({"seed": seed, "true_species": spc, "restored": flag, "images": len(g),
                                     "routing_correct": int(g["routing_correct"].sum()),
                                     "final_correct": int(g["final_correct"].sum())})
            logs.append(log.assign(seed=seed))
        pipe_rows.append(row)
    pipe = pd.DataFrame(pipe_rows).sort_values("seed") if pipe_rows else pd.DataFrame()
    pipe_metrics = [c for c in ["final_accuracy", "routing_accuracy", "final_acc_original", "final_acc_restored",
                                "theory_simple", "theory_class_aware", "theory_perfect_routing"] if c in pipe.columns]
    pipe_summary = summarize(pipe.assign(all="pipeline"), ["all"], pipe_metrics) if len(pipe) else pd.DataFrame()

    # ── run state ────────────────────────────────────────────────────────────────
    state_rows = []
    sp = os.path.join(out_dir, f"{prefix}_run_all_state.json")
    if os.path.exists(sp):
        st = json.load(open(sp))
        for k, v in st.get("runs", {}).items():
            state_rows.append({"run": k, **{kk: vv for kk, vv in v.items() if kk != "traceback"}})
        for k, v in st.get("evaluations", {}).items():
            state_rows.append({"run": f"pipeline/{k}", "status": v})

    readme = pd.DataFrame({"sheet": [
        "component_per_seed", "component_summary", "pipeline_per_seed", "pipeline_summary", "pipeline_species",
        "histories", "pipeline_logs", "configs", "run_state", "warnings"], "contents": [
        "Final val_*/test_* metrics per component and seed, plus training-curve statistics",
        "Mean ± std and 95% t-interval per component over seeds",
        "Assembled pipeline per seed: final/routing accuracy, theory, original-216 vs restored-20",
        "Mean ± std and 95% t-interval of the pipeline over seeds",
        "Per seed × true species × restored flag: images, routed correctly, final correct",
        "Every epoch of every run (accuracy, loss, val_accuracy, val_loss, lr)",
        "Every test image × seed: routing, prediction, correctness, restored flag",
        "config.txt of each component (last seed run)", "Run-all status per run", "Missing files, if any"]})

    with pd.ExcelWriter(cfg["out_file"], engine="openpyxl") as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        per_seed.to_excel(xw, sheet_name="component_per_seed", index=False)
        comp_summary.to_excel(xw, sheet_name="component_summary", index=False)
        pipe.to_excel(xw, sheet_name="pipeline_per_seed", index=False)
        pipe_summary.to_excel(xw, sheet_name="pipeline_summary", index=False)
        pd.DataFrame(species_rows).to_excel(xw, sheet_name="pipeline_species", index=False)
        (pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()).to_excel(xw, sheet_name="histories", index=False)
        (pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()).to_excel(xw, sheet_name="pipeline_logs", index=False)
        pd.DataFrame(configs).to_excel(xw, sheet_name="configs", index=False)
        pd.DataFrame(state_rows).to_excel(xw, sheet_name="run_state", index=False)
        pd.DataFrame({"warning": warnings or ["none"]}).to_excel(xw, sheet_name="warnings", index=False)

    print(f"[Collect] {len(per_seed)} component runs, {len(pipe)} pipeline seeds, "
          f"{sum(len(h) for h in histories)} history rows -> {cfg['out_file']}")
    for w in warnings:
        print(f"  ! {w}")
    if len(comp_summary):
        t = comp_summary[comp_summary["metric"] == "test_accuracy"]
        print("\nTest accuracy per component (mean ± std over seeds):")
        for _, r in t.iterrows():
            print(f"  {r['component']:11s} {100 * r['mean']:6.2f}% ± {100 * r['std']:5.2f}%  (n={r['n_seeds']})")
    if len(pipe_summary):
        r = pipe_summary[pipe_summary["metric"] == "final_accuracy"].iloc[0]
        print(f"\nAssembled pipeline: {100 * r['mean']:.2f}% ± {100 * r['std']:.2f}%  "
              f"(95% CI [{100 * r['ci95_low']:.2f}%, {100 * r['ci95_high']:.2f}%], n={r['n_seeds']})")


if __name__ == "__main__":
    main(COLLECT)
