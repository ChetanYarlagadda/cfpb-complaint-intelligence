"""All report figures, drawn from the result tables in reports/results/."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from complaintiq import plotting
from complaintiq.plotting import INK, INK_2, METHOD_COLOR, MODEL_COLOR, MUTED, SERIES, SURFACE

# Takeaway titles, written after reading the results.
TITLES = {
    "volume": "Template campaigns made up 54% of 2025-26 complaints; one Cash App template was filed 26,464 times",
    "bursts": "Embedding topics flag the Jan 2025 Cash App and Zelle campaigns, a Zelle relapse in Oct 2025 and smaller surges",
    "per_class": "TF-IDF + LR beats fine-tuned DistilBERT on 9 of 13 issues; vague issues stay at F1 <= 0.15",
    "confusion": "Fraud, unauthorized and 'other' transaction complaints are mostly confused with each other",
    "coverage": "Auto-routing only the most confident 25% lifts TF-IDF accuracy from 0.44 to 0.71",
    "protocol": "Random CV overstates the temporal test by 0.045 macro-F1; scoring template copies drops it to 0.265",
    "search": "Sentence embeddings beat TF-IDF by 0.06 same-issue P@10; the cross-encoder adds nothing significant",
    "injection": "Surges of +200% over a topic's expected count are flagged 89% of the time; small topics need more",
}

SHORT_ISSUE = {
    "Confusing or misleading advertising or marketing": "Misleading advertising",
    "Confusing or missing disclosures": "Confusing/missing disclosures",
    "Fraud or scam": "Fraud or scam",
    "Lost or stolen money order": "Lost/stolen money order",
    "Managing, opening, or closing your mobile wallet account": "Managing/closing wallet account",
    "Money was not available when promised": "Money not available when promised",
    "Other service problem": "Other service problem",
    "Other transaction problem": "Other transaction problem",
    "Problem with customer service": "Customer service",
    "Trouble accessing funds in your mobile or digital wallet": "Trouble accessing wallet funds",
    "Unauthorized transactions or other transaction problem": "Unauthorized/other transaction",
    "Unexpected or other fees": "Unexpected or other fees",
    "Wrong amount charged or received": "Wrong amount charged/received",
}


def short(issue: str) -> str:
    return SHORT_ISSUE.get(issue, issue)


def _suptitle(fig: plt.Figure, text: str) -> None:
    fig.suptitle(text, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)


def fig_volume(volume: pd.DataFrame, groups: pd.DataFrame, path: Path) -> Path:
    """Monthly complaints vs distinct near-duplicate groups (top), share in template campaigns (bottom)."""
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 5.8), sharex=True, gridspec_kw={"height_ratios": [2.4, 1], "hspace": 0.08})
    x = pd.PeriodIndex(volume.month, freq="M").to_timestamp()
    ax.plot(x, volume.complaints, color=SERIES[0], lw=1.8, drawstyle="steps-mid")
    ax.plot(x, volume.distinct_groups, color=SERIES[2], lw=1.8, drawstyle="steps-mid")
    ax.set_yscale("log")
    ax.set_ylim(volume.distinct_groups.min() * 0.6, volume.complaints.max() * 2.5)
    ax.set_ylabel("Complaints per month (log)")
    peak = int(volume.complaints.idxmax())
    lines = [f"{volume.month[peak]}: {volume.complaints[peak]:,} complaints"]
    lines += [f"{g.top_company}: one template, {g.n_complaints:,} copies" for g in groups.head(2).itertuples()]
    ax.annotate("\n".join(lines), xy=(x[peak], volume.complaints[peak]), xytext=(-18, -4), textcoords="offset points",
                ha="right", va="top", fontsize=8.5, color=INK_2, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    share = 100 * volume.in_campaign_groups / volume.complaints
    ax2.bar(x, share, width=24, color=SERIES[1])
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("% template\ncopies")
    ax2.set_xlabel("Month received")
    ax2.grid(axis="x", visible=False)
    fig.legend(
        handles=[
            Line2D([], [], color=SERIES[0], lw=1.8, label="all complaints with a narrative"),
            Line2D([], [], color=SERIES[2], lw=1.8, label="distinct complaints (one per near-duplicate group)"),
            Patch(color=SERIES[1], label="share in template groups (>= 100 near-identical complaints)"),
        ],
        loc="lower center", ncol=3, fontsize=8.2, bbox_to_anchor=(0.5, -0.04),
    )
    _suptitle(fig, TITLES["volume"])
    fig.subplots_adjust(top=0.92, bottom=0.14)
    plotting.save(fig, path)
    return path


def fig_bursts(cells: pd.DataFrame, topics: pd.DataFrame, path: Path) -> Path:
    """Topics x months heatmap of observed / expected share; burst months marked."""
    cells = cells.copy()
    flagged = sorted(cells[cells.burst].topic.unique(), key=lambda t: cells[(cells.topic == t) & cells.burst].month.min())
    months = sorted(cells.month.unique())
    cells.loc[cells.expected < 2, "ratio"] = np.nan  # too little history for a meaningful ratio
    mat = (
        cells[cells.topic.isin(flagged)]
        .pivot(index="topic", columns="month", values="ratio")
        .reindex(index=flagged, columns=months)
    )
    cmap = LinearSegmentedColormap.from_list("bo", [SERIES[0], SURFACE, SERIES[1]])
    cmap.set_bad(plotting.GRID)
    fig, ax = plt.subplots(figsize=(11, 0.36 * len(flagged) + 1.6))
    im = ax.imshow(np.log2(mat.to_numpy().clip(1 / 8, 8)), aspect="auto", cmap=cmap, vmin=-3, vmax=3, interpolation="nearest")
    ax.grid(False)
    for r, t in enumerate(flagged):
        sub = cells[(cells.topic == t) & cells.burst]
        for m, organic in zip(sub.month, sub.burst_in_distinct_groups):
            c = months.index(m)
            ax.plot(c, r, "o", ms=5.5, mfc=INK if organic else "none", mec=INK, mew=1.2)
    labels = topics.set_index("topic").label
    ax.set_yticks(range(len(flagged)), [f"{t}: {labels[t]}" for t in flagged], fontsize=8)
    ticks = [i for i, m in enumerate(months) if m.endswith("-01")]
    ax.set_xticks(ticks, [months[i][:4] for i in ticks])
    ax.set_xlabel("Month received (each column is one month)")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_ticks([-3, -1, 0, 1, 3], labels=["1/8", "1/2", "1", "2", "8"])
    cb.set_label("share / median share of the previous 6 months", color=INK_2)
    cb.outline.set_visible(False)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", mfc=INK, mec=INK, label="burst, also in distinct complaints"),
            Line2D([], [], marker="o", ls="", mfc="none", mec=INK, mew=1.2, label="burst only in raw volume (template copies)"),
            Patch(color=plotting.GRID, label="baseline < 2 complaints"),
        ],
        loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=8,
    )
    _suptitle(fig, TITLES["bursts"])
    fig.subplots_adjust(top=1 - 0.75 / (0.36 * len(flagged) + 1.6))
    plotting.save(fig, path)
    return path


def fig_per_class(per_class: pd.DataFrame, results: pd.DataFrame, path: Path) -> Path:
    """Per-issue F1 of both routers, issues ordered by test support."""
    pc = per_class.sort_values("test_n").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(pc) + 1.6))
    y = np.arange(len(pc))
    ax.barh(y + 0.18, pc.f1_tfidf, height=0.34, color=MODEL_COLOR["TF-IDF + LR"])
    ax.barh(y - 0.18, pc.f1_distilbert, height=0.34, color=MODEL_COLOR["DistilBERT (fine-tuned)"])
    ax.set_yticks(y, [f"{short(i)}  (n={n:,})" for i, n in zip(pc.issue, pc.test_n)], fontsize=8.5)
    ax.set_xlim(0, max(0.7, pc[["f1_tfidf", "f1_distilbert"]].to_numpy().max() + 0.05))
    ax.set_xlabel("F1 on the 2025-26 test complaints")
    ax.grid(axis="y", visible=False)
    test = results[results.protocol == "temporal test (deduplicated)"].set_index("model").macro_f1
    ax.legend(
        handles=[Patch(color=c, label=f"{m} (macro-F1 {test[m]:.3f})") for m, c in MODEL_COLOR.items()],
        loc="lower right", fontsize=8.5,
    )
    ax.set_title(TITLES["per_class"], loc="left")
    plotting.save(fig, path)
    return path


def fig_confusion(conf: pd.DataFrame, model: str, path: Path) -> Path:
    """Row-normalised confusion matrix of the model chosen on validation."""
    labels = list(dict.fromkeys(conf.true))
    mat = conf.pivot(index="true", columns="predicted", values="count").reindex(index=labels, columns=labels).to_numpy(float)
    share = mat / mat.sum(axis=1, keepdims=True).clip(min=1)
    cmap = LinearSegmentedColormap.from_list("blue", [SURFACE, SERIES[0]])
    fig, ax = plt.subplots(figsize=(8.8, 7.6))
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if share[i, j] >= 0.05:
                ax.text(j, i, f"{share[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color=SURFACE if share[i, j] > 0.55 else INK)
    names = [short(lab) for lab in labels]
    ax.set_xticks(range(len(labels)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), names, fontsize=8)
    ax.set_xlabel("Predicted issue")
    ax.set_ylabel("Issue the consumer selected")
    ax.set_title(f"{TITLES['confusion']}\n{model}, share of each true issue (cells >= 0.05 labelled)", loc="left", fontsize=10.5)
    plotting.save(fig, path)
    return path


def fig_coverage(rc: pd.DataFrame, path: Path) -> Path:
    """Accuracy when only the most confident share of complaints is auto-routed."""
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for model, color in MODEL_COLOR.items():
        s = rc[rc.model == model]
        ax.plot(s.coverage * 100, s.accuracy, "-o", color=color, ms=4, mec=SURFACE, mew=1, label=model)
    ax.set_xlabel("Complaints auto-routed (% most confident)")
    ax.set_ylabel("Accuracy on the auto-routed complaints")
    ax.set_xlim(0, 102)
    ax.legend(loc="lower left")
    ax.set_title(TITLES["coverage"], loc="left")
    plotting.save(fig, path)
    return path


def fig_protocol(results: pd.DataFrame, path: Path) -> Path:
    """Macro-F1 of the same models under different evaluation protocols."""
    order = ["random 5-fold CV inside 2023-2024", "validation (late 2024)", "temporal test (deduplicated)", "temporal test incl. template copies"]
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    width = 0.36
    for i, (model, color) in enumerate(MODEL_COLOR.items()):
        s = results[results.model == model].set_index("protocol").reindex(order)
        x = np.arange(len(order)) + (i - 0.5) * width
        ax.bar(x, s.macro_f1.fillna(0), width=width - 0.04, color=color, label=model)
        for xi, v in zip(x, s.macro_f1):
            ax.text(xi, (0 if np.isnan(v) else v) + 0.01, "n/a" if np.isnan(v) else f"{v:.3f}", ha="center", fontsize=8, color=INK_2 if not np.isnan(v) else MUTED)
    ax.set_xticks(range(len(order)), ["random 5-fold CV\n(2023-24)", "validation\n(Nov-Dec 2024)", "test 2025-26\n(deduplicated)", "test 2025-26 incl.\ntemplate copies"])
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, results.macro_f1.max() + 0.08)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_title(TITLES["protocol"], loc="left")
    plotting.save(fig, path)
    return path


def fig_search(summary: pd.DataFrame, path: Path) -> Path:
    """Same-issue precision@k of each retrieval method, with the random-retrieval expectation."""
    ks = [1, 5, 10]
    methods = [m for m in METHOD_COLOR if m in set(summary.method) and m != "Random (label prior)"]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    width = 0.8 / len(methods)
    s = summary.set_index("method")
    for i, m in enumerate(methods):
        x = np.arange(len(ks)) + (i - (len(methods) - 1) / 2) * width
        vals = [s.loc[m, f"p@{k}"] for k in ks]
        err = [[v - s.loc[m, f"p@{k}_ci_low"] for k, v in zip(ks, vals)], [s.loc[m, f"p@{k}_ci_high"] - v for k, v in zip(ks, vals)]]
        ax.bar(x, vals, width=width - 0.03, color=METHOD_COLOR[m], label=m)
        ax.errorbar(x, vals, yerr=err, fmt="none", ecolor=INK_2, elinewidth=1, capsize=2)
    rnd = s.loc["Random (label prior)", "p@10"]
    ax.axhline(rnd, color=MUTED, ls="--", lw=1.2)
    ax.set_xticks(range(len(ks)), [f"P@{k}" for k in ks])
    ax.set_ylabel("Share of retrieved cases with the same issue")
    ax.set_ylim(0, max(0.5, summary[[f"p@{k}_ci_high" for k in ks]].to_numpy().max() + 0.05))
    ax.grid(axis="x", visible=False)
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=MUTED, ls="--", lw=1.2, label=f"random retrieval ({rnd:.2f})"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8.5)
    ax.set_title(TITLES["search"], loc="left")
    plotting.save(fig, path)
    return path


def fig_injection(inj: pd.DataFrame, path: Path) -> Path:
    """Recovery rate of synthetic surges by size and by the topic's expected monthly count."""
    inj = inj.copy()
    bins = [0, 15, 50, np.inf]
    names = ["expected < 15 / month", "15-50", "> 50"]
    inj["size"] = pd.cut(inj.expected, bins=bins, labels=names)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for i, name in enumerate(names):
        s = inj[inj["size"] == name].groupby("multiplier").detected.agg(["mean", "size"])
        if len(s):
            ax.plot(s.index * 100, s["mean"], "-o", color=SERIES[i], mec=SURFACE, mew=1, label=f"{name} (n={int(s['size'].iloc[0])} cells)")
    overall = inj.groupby("multiplier").detected.mean()
    ax.plot(overall.index * 100, overall, "--", color=INK_2, lw=1.2, label="all injected cells")
    ax.axvline(100, color=MUTED, ls=":", lw=1)
    ax.text(104, 0.03, "share doubles: the >= 2x rule", fontsize=8, color=MUTED)
    ax.set_xlabel("Injected extra complaints (% of the topic's expected monthly count)")
    ax.set_ylabel("Share of injected surges flagged")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="lower right", fontsize=8.5, title="topic size", title_fontsize=8.5)
    ax.set_title(TITLES["injection"], loc="left", fontsize=10.5)
    plotting.save(fig, path)
    return path


def make_all(results: Path, figs: Path) -> list[Path]:
    """Redraw every figure from the CSVs in `results`."""
    plotting.apply_style()
    read = lambda f: pd.read_csv(results / f)  # noqa: E731
    routing = read("routing_results.csv")
    top_conf = read("routing_top_confusions.csv")
    return [
        fig_volume(read("monthly_volume.csv"), read("near_duplicate_groups.csv"), figs / "01_volume_and_campaigns.png"),
        fig_bursts(read("topic_months.csv"), read("topics.csv"), figs / "02_emerging_topic_bursts.png"),
        fig_per_class(read("routing_per_class.csv"), routing, figs / "03_routing_per_issue_f1.png"),
        fig_confusion(read("routing_confusion.csv"), top_conf.model.iloc[0], figs / "04_routing_confusion.png"),
        fig_coverage(read("routing_risk_coverage.csv"), figs / "05_routing_risk_coverage.png"),
        fig_protocol(routing, figs / "06_evaluation_protocol.png"),
        fig_search(read("search_results.csv"), figs / "07_similar_case_precision.png"),
        fig_injection(read("burst_injection.csv"), figs / "08_burst_injection_recovery.png"),
    ]
