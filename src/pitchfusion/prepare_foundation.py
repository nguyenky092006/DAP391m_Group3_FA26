"""Generate the complete B1-B5 evidence package from frozen ViSEC artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
CHANNELS = (
    "log_f0",
    "voicing_confidence",
    "log_energy",
    "delta_log_f0",
    "delta_log_energy",
)


def draw_heatmap(frame: pd.DataFrame, title: str, path: Path, cmap: str, fmt: str) -> None:
    values = frame.to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(8, 4.8))
    image = axis.imshow(values, aspect="auto", cmap=cmap)
    axis.set_xticks(range(len(frame.columns)), frame.columns, rotation=30, ha="right")
    axis.set_yticks(range(len(frame.index)), frame.index)
    axis.set_title(title)
    threshold = float(np.nanmean(values))
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, format(values[row, column], fmt), ha="center", va="center", color="white" if values[row, column] > threshold else "black")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def load_manifest(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path, dtype={"sample_id": "string", "speaker_id": "string"})
    if frame["sample_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate sample_id values")
    eligible = frame[frame["split"] != "excluded"].copy()
    if len(frame) != 5280 or len(eligible) != 4992:
        raise ValueError(f"Expected 5,280 source and 4,992 eligible rows; found {len(frame)} and {len(eligible)}")
    return frame, eligible


def create_b2(frame: pd.DataFrame, eligible: pd.DataFrame, output: Path) -> dict[str, object]:
    distributions: list[pd.DataFrame] = []
    for column in ("emotion", "accent_clean", "gender_clean"):
        table = eligible[column].fillna("missing").value_counts(dropna=False).rename_axis("value").reset_index(name="samples")
        table.insert(0, "field", column)
        table["percent"] = 100.0 * table["samples"] / len(eligible)
        distributions.append(table)
    write_csv(pd.concat(distributions, ignore_index=True), output / "b2_distributions.csv")
    durations = eligible["duration_sec_measured"].astype(float)
    summary = {
        "source_rows": len(frame),
        "eligible_rows": len(eligible),
        "excluded_rows": int((frame["split"] == "excluded").sum()),
        "speakers": int(eligible["speaker_id"].nunique()),
        "missing_embedded_audio": int((~frame["embedded_audio_present"].astype(bool)).sum()),
        "unreadable_audio": int((~frame["audio_readable"].astype(bool)).sum()),
        "duplicate_waveform_rows": int(frame["duplicate_content_group"].notna().sum()),
        "possible_clipping": int(eligible["possible_clipping"].astype(bool).sum()),
        "high_silence": int((eligible["high_silence_ratio"].astype(float) >= 0.60).sum()),
        "sample_rates": {str(k): int(v) for k, v in frame["sample_rate_hz"].value_counts().items()},
        "channels": {str(k): int(v) for k, v in frame["num_channels"].value_counts().items()},
        "duration_seconds": {
            "min": float(durations.min()),
            "median": float(durations.median()),
            "mean": float(durations.mean()),
            "p95": float(durations.quantile(0.95)),
            "max": float(durations.max()),
        },
    }
    atomic_json(output / "b2_data_quality_summary.json", summary)
    dictionary = pd.DataFrame(
        [
            {"field": name, "dtype": str(frame[name].dtype), "role": (
                "target" if name == "emotion" else
                "split_only" if name in {"speaker_id", "split", "cv_fold"} else
                "analysis_only" if name in {"accent_clean", "gender_clean", "duration_sec_measured"} else
                "audit_or_lineage"
            )}
            for name in frame.columns
        ]
    )
    write_csv(dictionary, output / "b2_data_dictionary.csv")
    return summary


QUERIES = {
    "emotion_distribution_by_accent": """
        WITH counts AS (
          SELECT accent_clean AS accent, emotion, COUNT(*) AS samples
          FROM eligible GROUP BY accent_clean, emotion
        )
        SELECT accent, emotion, samples,
               ROUND(100.0 * samples / SUM(samples) OVER (PARTITION BY accent), 2) AS percent_within_accent,
               RANK() OVER (PARTITION BY accent ORDER BY samples DESC) AS frequency_rank
        FROM counts ORDER BY accent, frequency_rank, emotion
    """,
    "duration_by_emotion": """
        SELECT emotion, COUNT(*) AS samples,
               ROUND(AVG(duration_sec_measured), 3) AS mean_seconds,
               ROUND(MIN(duration_sec_measured), 3) AS min_seconds,
               ROUND(MAX(duration_sec_measured), 3) AS max_seconds
        FROM eligible GROUP BY emotion ORDER BY emotion
    """,
    "speaker_concentration": """
        WITH counts AS (
          SELECT accent_clean AS accent, speaker_id, COUNT(*) AS samples
          FROM eligible GROUP BY accent_clean, speaker_id
        ), ranked AS (
          SELECT *, RANK() OVER (PARTITION BY accent ORDER BY samples DESC) AS speaker_rank,
                 SUM(samples) OVER (PARTITION BY accent) AS accent_samples
          FROM counts
        )
        SELECT accent, speaker_id, samples, accent_samples,
               ROUND(100.0 * samples / accent_samples, 2) AS percent_of_accent, speaker_rank
        FROM ranked WHERE speaker_rank <= 5 ORDER BY accent, speaker_rank
    """,
    "quality_by_accent_emotion": """
        SELECT accent_clean AS accent, emotion, COUNT(*) AS samples,
               ROUND(AVG(duration_sec_measured), 3) AS mean_duration_seconds,
               ROUND(100.0 * AVG(possible_clipping), 2) AS clipping_percent,
               ROUND(100.0 * AVG(CASE WHEN high_silence_ratio >= 0.60 THEN 1.0 ELSE 0.0 END), 2) AS high_silence_percent
        FROM eligible GROUP BY accent_clean, emotion ORDER BY accent, emotion
    """,
    "split_coverage": """
        SELECT split, accent_clean AS accent, emotion, COUNT(*) AS samples,
               COUNT(DISTINCT speaker_id) AS speakers
        FROM eligible GROUP BY split, accent_clean, emotion ORDER BY split, accent, emotion
    """,
}


def create_b3(eligible: pd.DataFrame, output: Path, figures: Path, database: Path) -> dict[str, object]:
    database.parent.mkdir(parents=True, exist_ok=True)
    if database.exists():
        database.unlink()
    connection = sqlite3.connect(database)
    eligible.to_sql("eligible", connection, index=False)
    results: dict[str, pd.DataFrame] = {}
    for name, query in QUERIES.items():
        results[name] = pd.read_sql_query(query, connection)
        write_csv(results[name], output / f"b3_{name}.csv")
    connection.close()

    figures.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(eligible["duration_sec_measured"], bins=50, color="#4c78a8", edgecolor="white")
    plt.xlabel("Duration (seconds)")
    plt.title("ViSEC eligible utterance duration")
    plt.tight_layout()
    plt.savefig(figures / "b3_duration_distribution.png", dpi=180)
    plt.close()

    cross = pd.crosstab(eligible["accent_clean"], eligible["emotion"])
    draw_heatmap(cross, "Emotion counts by accent", figures / "b3_emotion_accent_heatmap.png", "Blues", ".0f")

    grouped = [group["duration_sec_measured"].to_numpy() for _, group in eligible.groupby(["emotion", "accent_clean"], sort=True)]
    labels = [f"{emotion}\n{accent}" for (emotion, accent), _ in eligible.groupby(["emotion", "accent_clean"], sort=True)]
    plt.figure(figsize=(11, 5)); plt.boxplot(grouped, tick_labels=labels, showfliers=False)
    plt.ylabel("Duration (seconds)"); plt.title("Duration by emotion and accent"); plt.xticks(rotation=35, ha="right")
    plt.tight_layout(); plt.savefig(figures / "b3_duration_emotion_accent.png", dpi=180); plt.close()
    return {
        "queries": list(QUERIES),
        "eligible_rows": len(eligible),
        "median_duration_seconds": float(eligible["duration_sec_measured"].median()),
        "accent_counts": {str(k): int(v) for k, v in eligible["accent_clean"].value_counts().items()},
    }


def create_prosody_summary(index_path: Path, root: Path, output_path: Path) -> pd.DataFrame:
    if output_path.exists():
        return pd.read_parquet(output_path)
    index = pd.read_csv(index_path, dtype={"sample_id": "string"})
    rows: list[dict[str, object]] = []
    for number, row in enumerate(index.itertuples(index=False), start=1):
        path = Path(str(row.artifact_path))
        path = path if path.is_absolute() else root / path
        with np.load(path, allow_pickle=False) as stored:
            values = np.asarray(stored["values"], dtype=np.float32)
        record: dict[str, object] = {"sample_id": str(row.sample_id)}
        for channel_index, channel in enumerate(CHANNELS):
            record[f"{channel}_mean"] = float(values[channel_index].mean())
            record[f"{channel}_std"] = float(values[channel_index].std())
        rows.append(record)
        if number % 500 == 0 or number == len(index):
            print(f"B4 prosody summaries {number}/{len(index)}", flush=True)
    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.stem + ".tmp.parquet")
    result.to_parquet(temporary, index=False)
    temporary.replace(output_path)
    return result


def create_dashboard(data: pd.DataFrame, path: Path) -> None:
    records = data[["sample_id", "emotion", "accent_clean", "tsne_x", "tsne_y", "log_f0_mean", "log_energy_mean"]].round(5).to_dict("records")
    payload = json.dumps(records, ensure_ascii=False)
    html = """<!doctype html><html><head><meta charset='utf-8'><title>ViSEC B4 Feature Dashboard</title>
<style>body{font-family:Segoe UI,Arial;margin:24px;color:#172033}select{margin:0 16px 16px 4px;padding:6px}svg{border:1px solid #ccd3df;background:#fff}table{border-collapse:collapse;margin-top:18px}th,td{border:1px solid #ddd;padding:6px 10px;text-align:right}th:first-child,td:first-child{text-align:left}.dot{opacity:.65}</style></head><body>
<h1>ViSEC Feature Dashboard</h1><label>Emotion <select id='emotion'></select></label><label>Accent <select id='accent'></select></label><p id='status'></p><svg id='plot' width='900' height='520'></svg><table><thead><tr><th>Group</th><th>Samples</th><th>Mean log-F0</th><th>Mean log-energy</th></tr></thead><tbody id='summary'></tbody></table>
<script>const rows=__DATA__;const emotions=['all',...new Set(rows.map(r=>r.emotion))];const accents=['all',...new Set(rows.map(r=>r.accent_clean))];for(const [id,vals] of [['emotion',emotions],['accent',accents]]){const s=document.getElementById(id);vals.forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;s.appendChild(o)});s.onchange=draw}function draw(){const e=document.getElementById('emotion').value,a=document.getElementById('accent').value,d=rows.filter(r=>(e==='all'||r.emotion===e)&&(a==='all'||r.accent_clean===a));document.getElementById('status').textContent=`${d.length} samples`;const svg=document.getElementById('plot');svg.innerHTML='';const xs=d.map(r=>r.tsne_x),ys=d.map(r=>r.tsne_y),minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys),colors={angry:'#d62728',happy:'#f2b701',neutral:'#4c78a8',sad:'#7a5195'};d.forEach(r=>{const c=document.createElementNS('http://www.w3.org/2000/svg','circle');c.setAttribute('cx',30+840*(r.tsne_x-minx)/(maxx-minx||1));c.setAttribute('cy',490-450*(r.tsne_y-miny)/(maxy-miny||1));c.setAttribute('r',3);c.setAttribute('fill',colors[r.emotion]||'#555');c.setAttribute('class','dot');const t=document.createElementNS('http://www.w3.org/2000/svg','title');t.textContent=`${r.sample_id} | ${r.emotion} | ${r.accent_clean}`;c.appendChild(t);svg.appendChild(c)});const groups={};d.forEach(r=>{const k=`${r.accent_clean} / ${r.emotion}`;(groups[k]??=[]).push(r)});document.getElementById('summary').innerHTML=Object.entries(groups).map(([k,v])=>`<tr><td>${k}</td><td>${v.length}</td><td>${(v.reduce((s,r)=>s+r.log_f0_mean,0)/v.length).toFixed(3)}</td><td>${(v.reduce((s,r)=>s+r.log_energy_mean,0)/v.length).toFixed(3)}</td></tr>`).join('')}draw()</script></body></html>""".replace("__DATA__", payload)
    atomic_text(path, html)


def create_b4(eligible: pd.DataFrame, output: Path, figures: Path, dashboard: Path, root: Path) -> dict[str, object]:
    prosody = create_prosody_summary(
        root / "data/interim/features/multi_prosody_v1/feature_index.csv",
        root,
        root / "data/interim/features/multi_prosody_v1/sample_summaries.parquet",
    )
    features = pd.read_parquet(root / "data/interim/features/m1_features_v1.parquet")
    merged = eligible[["sample_id", "emotion", "accent_clean", "gender_clean"]].merge(prosody, on="sample_id", validate="one_to_one").merge(features, on="sample_id", validate="one_to_one")
    feature_columns = [column for column in features.columns if column != "sample_id"]
    sample = merged.sample(n=min(2000, len(merged)), random_state=391).reset_index(drop=True)
    matrix = StandardScaler().fit_transform(sample[feature_columns].to_numpy(np.float32))
    reduced = PCA(n_components=min(50, matrix.shape[1]), random_state=391).fit_transform(matrix)
    embedding = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", max_iter=1000, random_state=391).fit_transform(reduced)
    sample["tsne_x"] = embedding[:, 0]
    sample["tsne_y"] = embedding[:, 1]
    write_csv(sample[["sample_id", "emotion", "accent_clean", "tsne_x", "tsne_y"]], output / "b4_tsne_projection.csv")

    figures.mkdir(parents=True, exist_ok=True)
    emotion_order = sorted(merged["emotion"].unique())
    violin_values = [merged.loc[merged["emotion"] == emotion, "log_f0_mean"].to_numpy() for emotion in emotion_order]
    plt.figure(figsize=(8, 5)); plt.violinplot(violin_values, showmeans=True, showmedians=True)
    plt.xticks(range(1, len(emotion_order) + 1), emotion_order); plt.ylabel("Mean log-F0")
    plt.title("Mean log-F0 profile by emotion")
    plt.tight_layout(); plt.savefig(figures / "b4_log_f0_violin.png", dpi=180); plt.close()
    profile = merged.groupby("emotion")[[f"{c}_mean" for c in CHANNELS]].mean()
    standardized = (profile - profile.mean()) / profile.std(ddof=0).replace(0, 1)
    draw_heatmap(standardized, "Standardized multi-prosody profile by emotion", figures / "b4_prosody_profile_heatmap.png", "coolwarm", ".2f")
    plt.figure(figsize=(8, 6)); colors = {"angry": "#d62728", "happy": "#f2b701", "neutral": "#4c78a8", "sad": "#7a5195"}
    for emotion in sorted(sample["emotion"].unique()):
        group = sample[sample["emotion"] == emotion]
        plt.scatter(group["tsne_x"], group["tsne_y"], s=18, alpha=.65, label=emotion, color=colors.get(emotion))
    plt.legend(); plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2"); plt.title("t-SNE of 174-feature M1 representation")
    plt.tight_layout(); plt.savefig(figures / "b4_tsne_emotion_accent.png", dpi=180); plt.close()
    create_dashboard(sample, dashboard)
    summary = {
        "prosody_samples": len(prosody),
        "prosody_channels": list(CHANNELS),
        "m1_feature_dimensions": len(feature_columns),
        "tsne_samples": len(sample),
        "normalization_policy": "multi-prosody normalization is fitted on train only for modelling; visual summaries are exploratory",
    }
    atomic_json(output / "b4_feature_visualization_summary.json", summary)
    return summary


def create_b5(eligible: pd.DataFrame, output: Path) -> dict[str, object]:
    split_counts = eligible.groupby("split").agg(samples=("sample_id", "size"), speakers=("speaker_id", "nunique")).reset_index()
    write_csv(split_counts, output / "b5_split_counts.csv")
    coverage = eligible.groupby(["split", "accent_clean", "emotion"]).agg(samples=("sample_id", "size"), speakers=("speaker_id", "nunique")).reset_index()
    write_csv(coverage, output / "b5_split_coverage.csv")
    speaker_sets = {name: set(group["speaker_id"].astype(str)) for name, group in eligible.groupby("split")}
    overlap = {f"{left}_{right}": len(speaker_sets[left] & speaker_sets[right]) for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    cv = eligible[eligible["split"] == "train"].groupby("cv_fold").agg(samples=("sample_id", "size"), speakers=("speaker_id", "nunique")).reset_index()
    write_csv(cv, output / "b5_grouped_cv.csv")
    summary = {"split_version": str(eligible["split_version"].dropna().iloc[0]), "seed": int(eligible["split_seed"].dropna().iloc[0]), "speaker_overlap": overlap, "coverage_cells": len(coverage), "expected_coverage_cells": 36, "status": "PASS" if not any(overlap.values()) and len(coverage) == 36 else "FAIL"}
    atomic_json(output / "b5_partition_summary.json", summary)
    if summary["status"] != "PASS":
        raise ValueError(f"B5 partition validation failed: {summary}")
    return summary


def create_docs(b2: dict[str, object], b3: dict[str, object], b4: dict[str, object], b5: dict[str, object], docs: Path) -> None:
    atomic_text(docs / "B2_data_understanding.md", f"""# B2 Data Understanding\n\nThe pinned ViSEC release contains {b2['source_rows']:,} source records; {b2['eligible_rows']:,} are eligible and {b2['excluded_rows']:,} remain traceable as excluded. The eligible set contains {b2['speakers']} speaker IDs. No accent, gender, or speaker identifier is used as a model input.\n\nMachine-readable outputs are in `reports/foundation/b2_data_dictionary.csv`, `b2_distributions.csv`, and `b2_data_quality_summary.json`. Raw audio is unchanged.\n""")
    atomic_text(docs / "B3_eda_sql.md", f"""# B3 EDA and SQL\n\nEDA covers all {b3['eligible_rows']:,} eligible utterances. Median duration is {b3['median_duration_seconds']:.3f} seconds. Accent counts are {b3['accent_counts']}. Five named SQL queries cover emotion distribution, duration, speaker concentration, quality, and split coverage.\n\nThe reproducible notebook is `notebooks/B3_eda_sql.ipynb`; SQL is in `sql/B3_rq_queries.sql`; figures and tables are under `reports/figures/foundation` and `reports/foundation`. SQL is used only for metadata analysis.\n""")
    atomic_text(docs / "B4_feature_visualization.md", f"""# B4 Feature and Advanced Visualization\n\nThe feature package contains {b4['prosody_samples']:,} five-channel temporal prosody artifacts and a {b4['m1_feature_dimensions']}-dimensional M1 summary table. The three advanced figures are a log-F0 violin plot, standardized multi-prosody heatmap, and t-SNE projection of {b4['tsne_samples']:,} deterministic samples.\n\nThe interactive dashboard is `reports/dashboard/b4_feature_dashboard.html`. Model normalization remains train-only; full-data visualization transforms are exploratory and never reused for training.\n""")
    atomic_text(docs / "B5_partition_report.md", f"""# B5 Dataset Partition\n\nStatus: **{b5['status']}**. Split `{b5['split_version']}` uses seed {b5['seed']}, has {b5['coverage_cells']}/{b5['expected_coverage_cells']} split-accent-emotion cells, and zero speaker overlap across train, validation, and test. The same frozen split is required for M1-M5. Test remains reserved.\n\nMachine-readable split, coverage, grouped-CV, and validation evidence is stored in `data/manifests` and `reports/foundation`.\n""")


def create_notebook(path: Path) -> None:
    notebook = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# B3 ViSEC EDA and SQL\n", "Generated evidence is read from `reports/foundation`.\n"]}, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["from pathlib import Path\n", "import pandas as pd\n", "ROOT = Path('..')\n", "pd.read_csv(ROOT / 'reports/foundation/b3_emotion_distribution_by_accent.csv')\n"]}, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["pd.read_csv(ROOT / 'reports/foundation/b3_quality_by_accent_emotion.csv')\n"]}], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.14"}}, "nbformat": 4, "nbformat_minor": 5}
    atomic_json(path, notebook)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/split_manifest_v2_enriched.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = ROOT / "reports/foundation"
    figures = ROOT / "reports/figures/foundation"
    frame, eligible = load_manifest(args.manifest)
    print("B2 data understanding", flush=True); b2 = create_b2(frame, eligible, reports)
    print("B3 EDA and SQL", flush=True); b3 = create_b3(eligible, reports, figures, ROOT / "data/processed/visec_metadata.sqlite")
    print("B4 feature visualization", flush=True); b4 = create_b4(eligible, reports, figures, ROOT / "reports/dashboard/b4_feature_dashboard.html", ROOT)
    print("B5 partition validation", flush=True); b5 = create_b5(eligible, reports)
    create_docs(b2, b3, b4, b5, ROOT / "docs/research")
    create_notebook(ROOT / "notebooks/B3_eda_sql.ipynb")
    completion = {"status": "PASS", "steps": {"B1": "scope and RQ evidence matrix", "B2": b2, "B3": b3, "B4": b4, "B5": b5}, "lineage": {"manifest_sha256": sha256(args.manifest), "m1_features_sha256": sha256(ROOT / "data/interim/features/m1_features_v1.parquet")}}
    atomic_json(reports / "b1_b5_completion.json", completion)
    print(json.dumps({"status": "PASS", "output": str(reports)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
