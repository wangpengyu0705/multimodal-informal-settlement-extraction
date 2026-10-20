#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT = Path.home() / "projects" / "nairobi_v2"

INPUT = (
    PROJECT
    / "results/final_test_v2/seed_42/final_test_metrics_v2.csv"
)

OUT_DIR = (
    PROJECT
    / "results/publication_tables_v2/seed_42/"
      "final_main_performance_table"
)

OUT_CSV = OUT_DIR / "main_performance_table_final.csv"
OUT_MD = OUT_DIR / "main_performance_table_final.md"
OUT_TEX = OUT_DIR / "main_performance_table_final.tex"
OUT_JSON = OUT_DIR / "main_performance_table_final_audit.json"

EXPECTED_COVERED_N = 5744
EXPECTED_ALL_N = 22356

KEEP = [
    "pr_auc",
    "informal_precision",
    "informal_recall",
    "informal_f1",
    "accuracy",
]

DISPLAY_NAME = {
    "S2-only": "S2-only",
    "GSV Mean": "GSV-only",
    "Fusion Mean": "Fusion",
    "S2-only everywhere": "S2-only",
    "Conditional: Fusion if GSV, otherwise S2": "Final mapping",
}


def fmt(v):
    if pd.isna(v) or str(v).strip() == "":
        return "—"
    return f"{float(v):.3f}"


def load_final_metrics():
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)

    required = {
        "scope",
        "model",
        "n",
        *KEEP,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in Step15 metrics: {sorted(missing)}"
        )

    for col in KEEP:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    covered = df.loc[
        (df["scope"] == "GSV-covered test")
        & df["model"].isin(
            [
                "S2-only",
                "GSV Mean",
                "Fusion Mean",
            ]
        )
    ].copy()

    full = df.loc[
        (df["scope"] == "All-valid test")
        & df["model"].isin(
            [
                "S2-only everywhere",
                "Conditional: Fusion if GSV, otherwise S2",
            ]
        )
    ].copy()

    if len(covered) != 3:
        raise ValueError(
            f"Covered section expected 3 rows, found {len(covered)}"
        )
    if not (
        covered["n"].astype(int)
        == EXPECTED_COVERED_N
    ).all():
        raise ValueError(
            "Covered section population is not 5,744 cells"
        )

    if len(full) != 2:
        raise ValueError(
            f"Full-test section expected 2 rows, found {len(full)}"
        )
    if not (
        full["n"].astype(int)
        == EXPECTED_ALL_N
    ).all():
        raise ValueError(
            "Full-test section population is not 22,356 cells"
        )

    covered["display"] = covered["model"].map(
        DISPLAY_NAME
    )
    full["display"] = full["model"].map(
        DISPLAY_NAME
    )

    covered["_order"] = covered["display"].map(
        {
            "S2-only": 0,
            "GSV-only": 1,
            "Fusion": 2,
        }
    )
    full["_order"] = full["display"].map(
        {
            "S2-only": 0,
            "Final mapping": 1,
        }
    )

    covered = covered.sort_values("_order")
    full = full.sort_values("_order")

    # Guard against accidentally reporting a mixed-model PR-AUC.
    final_row = full.loc[
        full["display"] == "Final mapping"
    ]
    if len(final_row) != 1:
        raise ValueError(
            "Could not uniquely identify Final mapping row"
        )
    if final_row["pr_auc"].notna().any():
        raise ValueError(
            "Final mapping unexpectedly contains PR-AUC. "
            "The frozen protocol does not construct a mixed-model probability."
        )

    return covered, full


def save_csv(covered, full):
    rows = []

    for section, n, frame in [
        (
            "A. GSV-covered held-out Test cells",
            EXPECTED_COVERED_N,
            covered,
        ),
        (
            "B. All valid held-out Test cells",
            EXPECTED_ALL_N,
            full,
        ),
    ]:
        for _, r in frame.iterrows():
            rows.append(
                {
                    "Section": section,
                    "Population_n": n,
                    "Mapping approach": r["display"],
                    "PR-AUC": r["pr_auc"],
                    "Precision": r["informal_precision"],
                    "Recall": r["informal_recall"],
                    "F1": r["informal_f1"],
                    "Accuracy": r["accuracy"],
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(
        OUT_CSV,
        index=False,
    )
    return out


def save_markdown(covered, full):
    lines = [
        "**Table X. Predictive performance on the held-out Test set.**",
        "",
        "**A. GSV-covered held-out Test cells (n = 5,744)**",
        "",
        "| Mapping approach | PR-AUC | Precision | Recall | F1 | Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for _, r in covered.iterrows():
        lines.append(
            f"| {r['display']} | "
            f"{fmt(r['pr_auc'])} | "
            f"{fmt(r['informal_precision'])} | "
            f"{fmt(r['informal_recall'])} | "
            f"{fmt(r['informal_f1'])} | "
            f"{fmt(r['accuracy'])} |"
        )

    lines.extend(
        [
            "",
            "**B. All valid held-out Test cells (n = 22,356)**",
            "",
            "| Mapping approach | PR-AUC | Precision | Recall | F1 | Accuracy |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for _, r in full.iterrows():
        lines.append(
            f"| {r['display']} | "
            f"{fmt(r['pr_auc'])} | "
            f"{fmt(r['informal_precision'])} | "
            f"{fmt(r['informal_recall'])} | "
            f"{fmt(r['informal_f1'])} | "
            f"{fmt(r['accuracy'])} |"
        )

    lines.extend(
        [
            "",
            (
                "*Note:* Precision, recall and F1 refer to the "
                "informal-settlement class. PR-AUC is not reported for "
                "Final mapping because no single mixed probability score "
                "was constructed across the Fusion and S2-only branches."
            ),
        ]
    )

    text = "\n".join(lines)
    OUT_MD.write_text(
        text,
        encoding="utf-8",
    )
    return text


def save_latex(covered, full):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Predictive performance on the held-out Test set.}",
        r"\label{tab:final_performance}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Mapping approach & PR-AUC & Precision & Recall & F1 & Accuracy \\",
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{A. GSV-covered held-out Test cells (n = 5,744)}} \\",
    ]

    for _, r in covered.iterrows():
        lines.append(
            f"{r['display']} & "
            f"{fmt(r['pr_auc'])} & "
            f"{fmt(r['informal_precision'])} & "
            f"{fmt(r['informal_recall'])} & "
            f"{fmt(r['informal_f1'])} & "
            f"{fmt(r['accuracy'])} \\\\"
        )

    lines.extend(
        [
            r"\addlinespace",
            r"\multicolumn{6}{l}{\textit{B. All valid held-out Test cells (n = 22,356)}} \\",
        ]
    )

    for _, r in full.iterrows():
        lines.append(
            f"{r['display']} & "
            f"{fmt(r['pr_auc'])} & "
            f"{fmt(r['informal_precision'])} & "
            f"{fmt(r['informal_recall'])} & "
            f"{fmt(r['informal_f1'])} & "
            f"{fmt(r['accuracy'])} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.98\linewidth}",
            r"\footnotesize",
            (
                r"\textit{Note:} Precision, recall and F1 refer to the "
                r"informal-settlement class. PR-AUC is not reported for "
                r"Final mapping because no single mixed probability score "
                r"was constructed across the Fusion and S2-only branches."
            ),
            r"\end{minipage}",
            r"\end{table}",
        ]
    )

    OUT_TEX.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    covered, full = load_final_metrics()

    save_csv(
        covered,
        full,
    )
    markdown = save_markdown(
        covered,
        full,
    )
    save_latex(
        covered,
        full,
    )

    audit = {
        "status": "PASSED",
        "source_step": "Step15",
        "source_file": str(INPUT),
        "scientific_recomputation": False,
        "metrics_recomputed": False,
        "section_a_population": {
            "description": "GSV-covered held-out Test cells",
            "n": EXPECTED_COVERED_N,
        },
        "section_b_population": {
            "description": "All valid held-out Test cells",
            "n": EXPECTED_ALL_N,
        },
        "display_names": {
            "column": "Mapping approach",
            "final_row": "Final mapping",
        },
        "reported_metrics": [
            "PR-AUC",
            "Precision",
            "Recall",
            "F1",
            "Accuracy",
        ],
        "outputs": {
            "csv": str(OUT_CSV),
            "markdown": str(OUT_MD),
            "latex": str(OUT_TEX),
        },
    }

    OUT_JSON.write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 100)
    print("Nairobi v2 — Final Main-Text Performance Table")
    print("=" * 100)
    print("STATUS: PASSED")
    print()
    print(markdown)
    print()
    print("Outputs:")
    print(" ", OUT_CSV)
    print(" ", OUT_MD)
    print(" ", OUT_TEX)
    print(" ", OUT_JSON)


if __name__ == "__main__":
    main()
