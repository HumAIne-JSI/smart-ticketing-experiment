import pandas as pd
import json
import glob
import os
import statistics

BASE_DIR = r"C:\Users\rokkl\Desktop\IJS\smart_ticketing_experiment"
DB_DIR = os.path.join(BASE_DIR, "database", "workshop")
INSTANCES_DIR = os.path.join(DB_DIR, "instances")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "workshop")
EXCEL_PATH = os.path.join(BASE_DIR, "data", "User Request_last_team_ANON.xlsx")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load GT from Excel (Team->Name column)
df_excel = pd.read_excel(EXCEL_PATH)
df_excel = df_excel.drop_duplicates(subset="Ref", keep="first")
gt_lookup = {}
for _, row in df_excel.iterrows():
    ref = row["Ref"]
    label = row.get("Team->Name")
    if pd.notna(label):
        gt_lookup[ref] = label

# 2. Load user name mapping
with open(os.path.join(DB_DIR, "users.json"), encoding="utf-8") as f:
    users_list = json.load(f)
user_map = {u["user_id"]: u["username"] for u in users_list}

# 3. Discover all instance IDs
instance_ids = sorted(
    int(d.name) for d in os.scandir(INSTANCES_DIR) if d.is_dir() and d.name.isdigit()
)

for inst_id in instance_ids:
    duckdb_dir = os.path.join(INSTANCES_DIR, str(inst_id), "duckdb")

    # Load al_instances
    with open(os.path.join(duckdb_dir, "al_instances.json"), encoding="utf-8") as f:
        al_instances = json.load(f)

    inst_info = al_instances[0] if al_instances else {}

    # Load label_decisions
    with open(os.path.join(duckdb_dir, "label_decisions.json"), encoding="utf-8") as f:
        label_decisions = json.load(f)

    # Load metrics
    with open(os.path.join(duckdb_dir, "metrics.json"), encoding="utf-8") as f:
        metrics_list = json.load(f)

    # -- Build result --
    result = {
        "al_instance_id": inst_id,
        "workshop_folder": "workshop",
        "model_name": inst_info.get("model_name"),
        "query_strategy": inst_info.get("query_strategy"),
        "labeler_user_id": inst_info.get("user_id"),
        "labeler_username": user_map.get(inst_info.get("user_id"), ""),
        "created_at": inst_info.get("created_at"),
    }

    if not label_decisions:
        result["label_decisions_stats"] = {
            "n_total": 0,
            "n_with_model_prediction": 0,
            "n_with_ground_truth": 0,
            "n_latency_used": 0,
            "n_latency_excluded_zero": 0,
            "n_skipped_for_training": 0,
            "n_skipped_null_model": 0,
        }
        result["computed_metrics"] = {
            "average_latency_ms": None,
            "model_labeler_agreement_pct": None,
            "model_gt_agreement_pct": None,
            "labeler_gt_agreement_pct": None,
        }
        result["agreement_counts"] = {
            "model_labeler": {"match": 0, "total": 0},
            "model_gt": {"match": 0, "total": 0},
            "labeler_gt": {"match": 0, "total": 0},
        }
    else:
        # Filter out skipped_for_training = True
        active = [d for d in label_decisions if not d.get("skipped_for_training")]
        n_skipped_total = len(label_decisions) - len(active)

        n_total = len(active)

        # Latency: exclude 0 or None
        valid_latencies = [
            d["latency_ms"] for d in active
            if d.get("latency_ms") is not None and d["latency_ms"] > 0
        ]
        n_latency_excluded = n_total - len(valid_latencies)
        avg_latency = round(statistics.mean(valid_latencies), 2) if valid_latencies else None

        # For model-based comparisons: keep only rows with non-null model_prediction
        with_model_pred = [d for d in active if d.get("model_prediction") is not None]
        n_with_mp = len(with_model_pred)
        n_null_mp = n_total - n_with_mp

        # Model/Labeler agreement
        ml_match = sum(1 for d in with_model_pred if d["label"] == d["model_prediction"])
        ml_total = len(with_model_pred)
        ml_pct = round(100.0 * ml_match / ml_total, 2) if ml_total > 0 else None

        # Model/GT agreement (only rows where both model_pred and GT exist)
        mp_gt_rows = [d for d in with_model_pred if d["ref"] in gt_lookup]
        mp_gt_match = sum(1 for d in mp_gt_rows if d["model_prediction"] == gt_lookup[d["ref"]])
        mp_gt_total = len(mp_gt_rows)
        mp_gt_pct = round(100.0 * mp_gt_match / mp_gt_total, 2) if mp_gt_total > 0 else None

        # Labeler/GT agreement (all active rows where GT exists)
        lab_gt_rows = [d for d in active if d["ref"] in gt_lookup]
        lab_gt_match = sum(1 for d in lab_gt_rows if d["label"] == gt_lookup[d["ref"]])
        lab_gt_total = len(lab_gt_rows)
        lab_gt_pct = round(100.0 * lab_gt_match / lab_gt_total, 2) if lab_gt_total > 0 else None

        result["label_decisions_stats"] = {
            "n_total": n_total,
            "n_with_model_prediction": n_with_mp,
            "n_with_ground_truth": lab_gt_total,
            "n_latency_used": len(valid_latencies),
            "n_latency_excluded_zero": n_latency_excluded,
            "n_skipped_for_training": n_skipped_total,
            "n_skipped_null_model": n_null_mp,
        }

        result["computed_metrics"] = {
            "average_latency_ms": avg_latency,
            "model_labeler_agreement_pct": ml_pct,
            "model_gt_agreement_pct": mp_gt_pct,
            "labeler_gt_agreement_pct": lab_gt_pct,
        }

        result["agreement_counts"] = {
            "model_labeler": {"match": ml_match, "total": ml_total},
            "model_gt": {"match": mp_gt_match, "total": mp_gt_total},
            "labeler_gt": {"match": lab_gt_match, "total": lab_gt_total},
        }

    # Macro F1 from metrics.json (return as-is per iteration)
    macro_f1_list = []
    for m in metrics_list:
        macro_f1_list.append({
            "iteration_id": m.get("iteration_id"),
            "num_labeled": m.get("num_labeled"),
            "f1_score": m.get("f1_score"),
            "accuracy": m.get("accuracy"),
            "precision_macro": m.get("precision_macro"),
            "recall_macro": m.get("recall_macro"),
            "created_at": m.get("created_at"),
        })

    # Sort by iteration_id
    macro_f1_list.sort(key=lambda x: (x["iteration_id"] or 0))
    result["macro_f1_by_iteration"] = macro_f1_list

    # Write output file
    out_path = os.path.join(OUTPUT_DIR, f"{inst_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Print summary
    m = result["computed_metrics"]
    n = result["label_decisions_stats"]
    f1s = [x["f1_score"] for x in macro_f1_list]
    print(
        f"Instance {inst_id:>2}: decisions={n['n_total']:>2}  "
        f"lat_avg={m['average_latency_ms'] or 'N/A':>10}  "
        f"mod/lab={m['model_labeler_agreement_pct'] or 'N/A':>6}  "
        f"mod/GT={m['model_gt_agreement_pct'] or 'N/A':>6}  "
        f"lab/GT={m['labeler_gt_agreement_pct'] or 'N/A':>6}  "
        f"f1_iters={len(f1s)}  "
        f"f1_final={f1s[-1] if f1s else 'N/A':.4f}"
    )

print(f"\nOutput files in: {OUTPUT_DIR}/")
