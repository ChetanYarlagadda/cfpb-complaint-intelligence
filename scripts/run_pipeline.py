"""Run every experiment end to end.

Writes reports/results/*.csv|json and reports/figures/*.png and logs each configuration
to MLflow (./mlruns). Expensive intermediates (MinHash signatures, embeddings, fine-tuned
weights, predictions, cross-encoder rankings, live-API responses) are cached in data/cache/
under keys that include their settings, so re-runs only recompute what changed.

    python scripts/run_pipeline.py            # full run
    python scripts/run_pipeline.py --quick    # smoke test on a small sample; outputs go to data/cache/smoke/
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sentence_transformers  # noqa: E402
import sklearn  # noqa: E402
import spacy  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

from complaintiq import entities, finetune  # noqa: E402
from complaintiq.data import PRODUCT, SNAPSHOT_REPO, download_snapshot, live_crosscheck, load_complaints  # noqa: E402
from complaintiq.dedup import group_summary, minhash_check, minhash_signatures, near_duplicate_groups  # noqa: E402
from complaintiq.embeddings import CROSS_ENCODER_ID, EMBEDDERS, Embedder, Reranker, ids_key  # noqa: E402
from complaintiq.emerging import ctfidf_terms, detect_bursts, injection_recovery, kmeans_topics, monthly_matrix  # noqa: E402
from complaintiq.figures import make_all  # noqa: E402
from complaintiq.routing import build_splits, iid_cv_macro_f1, risk_coverage, top_confusions, tune_tfidf_lr  # noqa: E402
from complaintiq.search import dense_search, precision_at_k, random_precision, rerank, tfidf_search  # noqa: E402
from complaintiq.stats import bootstrap_macro_f1_diff, confusion, holm, macro_f1, paired_bootstrap, per_class_f1  # noqa: E402
from complaintiq.tracking import init_mlflow, log_artifacts, log_result  # noqa: E402

torch.set_num_threads(2)

CACHE = ROOT / "data" / "cache"
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "reports" / "results"
FIGS = ROOT / "reports" / "figures"

WINDOW = ("2021-01-01", "2026-05-31")  # narratives thin out after May 2026 (publication lag)
TRAIN = ("2023-01-01", "2024-12-31")  # issue taxonomy is stable from 2023
TEST = ("2025-01-01", "2026-05-31")
MIN_TRAIN_PER_CLASS = 100
CS = [1.0, 4.0, 16.0]
FT = finetune.FinetuneConfig()
N_TOPICS = 30
BURST = {"window": 6, "min_ratio": 2.0, "min_count": 30, "alpha": 0.01}
INJECT_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
N_INJECT_CELLS = 200
N_QUERIES = 1000
CAND_DEPTH = 50
KS = [1, 5, 10]
N_ENTITY_DOCS = 400
N_LIVE_CHECK = 100
CAMPAIGN_GROUP_SIZE = 100  # near-duplicate groups at least this large are treated as template campaigns


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


class Timer:
    """Wall-clock per stage; stages flagged gpu=True add to the GPU total."""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self.gpu: set[str] = set()

    def run(self, name: str, fn, *args, gpu: bool = False, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        self.stages[name] = self.stages.get(name, 0.0) + time.perf_counter() - t0
        if gpu and torch.cuda.is_available():
            self.gpu.add(name)
        return out


# --------------------------------------------------------------------------- data


def load(quick: bool) -> pd.DataFrame:
    path = download_snapshot(RAW)
    df = load_complaints(path, *WINDOW)
    if quick:
        df = df.sample(n=6000, random_state=0).sort_values(["date", "complaint_id"]).reset_index(drop=True)
    log(f"data: {len(df):,} complaints with narratives, {df.date.min():%Y-%m-%d} to {df.date.max():%Y-%m-%d}")
    return df


def add_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Near-duplicate group per complaint, plus a check of the MinHash estimates against exact Jaccard."""
    path = CACHE / f"minhash__128x5__{ids_key(df.complaint_id.tolist())}.npy"
    if path.exists():
        sig = np.load(path)
    else:
        sig = minhash_signatures(df.narrative.tolist(), num_perm=128, k=5)
        np.save(path, sig)
    df["group"] = near_duplicate_groups(sig, bands=32, threshold=0.5)
    size = df.groupby("group").complaint_id.transform("size")
    df["campaign"] = size >= CAMPAIGN_GROUP_SIZE
    log(f"dedup: {df.group.nunique():,} near-duplicate groups; {df.campaign.sum():,} complaints in groups of >= {CAMPAIGN_GROUP_SIZE}")
    pairs = minhash_check(df, sig, n_pairs=800)
    pairs["error"] = pairs.estimated - pairs.exact
    check = pairs.groupby("pair").agg(
        n=("exact", "size"), mean_exact=("exact", "mean"), mean_estimated=("estimated", "mean"), mean_error=("error", "mean"),
        sd_error=("error", "std"), share_exact_ge_0_5=("exact", lambda e: float((e >= 0.5).mean())),
    ).reset_index()
    check["corr_all_pairs"] = float(np.corrcoef(pairs.exact, pairs.estimated)[0, 1])
    return df, check


def monthly_volume(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("month")
    out = pd.DataFrame(
        {
            "complaints": g.size(),
            "distinct_groups": g.group.nunique(),
            "in_campaign_groups": g.campaign.sum(),
        }
    )
    out.index = out.index.astype(str)
    return out.reset_index()


# --------------------------------------------------------------------------- routing


def run_routing(df: pd.DataFrame, quick: bool, timer: Timer) -> dict:
    sp = build_splits(df, *TRAIN, *TEST, val_frac=0.1, min_train_per_class=MIN_TRAIN_PER_CLASS if not quick else 10)
    labels, k = sp.labels, len(sp.labels)
    log(f"routing: {k} issues; train {len(sp.train):,}, val {len(sp.val):,}, test {len(sp.test):,} (test incl. template copies {len(sp.test_all):,})")

    cs, folds = (CS, 5) if not quick else ([2.0], 2)
    tfidf_path = CACHE / "models" / f"tfidf_lr__{ids_key(sp.train.complaint_id.tolist() + sp.val.complaint_id.tolist())}__C{'-'.join(map(str, cs))}__cv{folds}.joblib"
    if tfidf_path.exists():
        tfidf, tuning, cv_scores = joblib.load(tfidf_path)
    else:
        tfidf, tuning = timer.run("tfidf_lr_s", tune_tfidf_lr, sp.train.text.tolist(), sp.train.y.to_numpy(), sp.val.text.tolist(),
                                  sp.val.y.to_numpy(), cs, k)
        cv_scores = timer.run("tfidf_cv_s", iid_cv_macro_f1, sp.train.text.tolist(), sp.train.y.to_numpy(),
                              float(tuning.loc[tuning.val_macro_f1.idxmax(), "C"]), k, folds=folds)
        tfidf_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump((tfidf, tuning, cv_scores), tfidf_path)
    best_c = float(tuning.loc[tuning.val_macro_f1.idxmax(), "C"])
    y_test = sp.test.y.to_numpy()
    proba_tfidf = tfidf.predict_proba(sp.test.text.tolist())
    pred_tfidf = proba_tfidf.argmax(axis=1)
    log(f"routing: TF-IDF + LR (C={best_c}) test macro-F1 {macro_f1(y_test, pred_tfidf, k):.4f}")
    pred_all = tfidf.predict(sp.test_all.text.tolist())

    # ---- DistilBERT
    cfg = FT if not quick else finetune.FinetuneConfig(epochs=1)
    key = finetune.cache_key(cfg, sp.train.complaint_id.tolist(), labels)
    model_dir = CACHE / "models" / f"distilbert-{key}"
    _, history = timer.run("distilbert_train_s", finetune.finetune, cfg, sp.train.text.tolist(), sp.train.y.to_numpy(),
                           sp.val.text.tolist(), sp.val.y.to_numpy(), labels, model_dir, log=log, gpu=True)
    pred_path = CACHE / "preds" / f"distilbert-{key}__{ids_key(sp.test.complaint_id.tolist())}.npy"
    if pred_path.exists():
        proba_bert = np.load(pred_path)
    else:
        model, tok = finetune.load_finetuned(model_dir)
        proba_bert = timer.run("distilbert_predict_s", finetune.predict_proba, model, tok, sp.test.text.tolist(), cfg.max_length, gpu=True)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(pred_path, proba_bert)
        del model
        torch.cuda.empty_cache()
    pred_bert = proba_bert.argmax(axis=1)
    log(f"routing: DistilBERT test macro-F1 {macro_f1(y_test, pred_bert, k):.4f}")

    # ---- tables
    def top2(proba: np.ndarray) -> float:
        return float((np.argsort(-proba, axis=1)[:, :2] == y_test[:, None]).any(axis=1).mean())

    # Truncation check: does DistilBERT trail only on narratives longer than its 256-token window?
    tok = transformers.AutoTokenizer.from_pretrained(model_dir)
    n_tokens = np.array([len(x) for x in tok(sp.test.text.tolist(), verbose=False)["input_ids"]])
    by_length = []
    for name, mask in [(f"<= {cfg.max_length} tokens (read in full)", n_tokens <= cfg.max_length),
                       (f"> {cfg.max_length} tokens (truncated)", n_tokens > cfg.max_length)]:
        by_length.append({"length": name, "n": int(mask.sum()), "share": float(mask.mean()),
                          "macro_f1_tfidf": macro_f1(y_test[mask], pred_tfidf[mask], k), "macro_f1_distilbert": macro_f1(y_test[mask], pred_bert[mask], k),
                          "accuracy_tfidf": float((pred_tfidf[mask] == y_test[mask]).mean()), "accuracy_distilbert": float((pred_bert[mask] == y_test[mask]).mean())})

    best_epoch = max(history, key=lambda h: h["val_macro_f1"])
    rows = [
        {"model": "TF-IDF + LR", "protocol": "temporal test (deduplicated)", "macro_f1": macro_f1(y_test, pred_tfidf, k),
         "accuracy": float((pred_tfidf == y_test).mean()), "top2_accuracy": top2(proba_tfidf), "n": len(y_test)},
        {"model": "DistilBERT (fine-tuned)", "protocol": "temporal test (deduplicated)", "macro_f1": macro_f1(y_test, pred_bert, k),
         "accuracy": float((pred_bert == y_test).mean()), "top2_accuracy": top2(proba_bert), "n": len(y_test)},
        {"model": "TF-IDF + LR", "protocol": "validation (late 2024)", "macro_f1": float(tuning.val_macro_f1.max()), "n": len(sp.val)},
        {"model": "DistilBERT (fine-tuned)", "protocol": "validation (late 2024)", "macro_f1": best_epoch["val_macro_f1"], "n": len(sp.val)},
        {"model": "TF-IDF + LR", "protocol": "random 5-fold CV inside 2023-2024", "macro_f1": float(np.mean(cv_scores)),
         "macro_f1_sd": float(np.std(cv_scores)), "n": len(sp.train)},
        {"model": "TF-IDF + LR", "protocol": "temporal test incl. template copies", "macro_f1": macro_f1(sp.test_all.y.to_numpy(), pred_all, k),
         "accuracy": float((pred_all == sp.test_all.y.to_numpy()).mean()), "n": len(sp.test_all)},
    ]
    results = pd.DataFrame(rows)
    sig = bootstrap_macro_f1_diff(y_test, pred_tfidf, pred_bert, k, n_boot=2000 if not quick else 200)

    support = np.bincount(y_test, minlength=k)
    per_class = pd.DataFrame(
        {
            "issue": labels,
            "train_n": np.bincount(sp.train.y, minlength=k),
            "test_n": support,
            "f1_tfidf": per_class_f1(y_test, pred_tfidf, k),
            "f1_distilbert": per_class_f1(y_test, pred_bert, k),
        }
    )
    best_name, best_pred, best_proba = (
        ("DistilBERT (fine-tuned)", pred_bert, proba_bert)
        if best_epoch["val_macro_f1"] >= tuning.val_macro_f1.max()
        else ("TF-IDF + LR", pred_tfidf, proba_tfidf)
    )
    cm = confusion(y_test, best_pred, k)
    cm_long = pd.DataFrame([{"true": labels[i], "predicted": labels[j], "count": int(cm[i, j])} for i in range(k) for j in range(k)])
    rc = pd.concat(
        [risk_coverage(y_test, proba_tfidf).assign(model="TF-IDF + LR"), risk_coverage(y_test, proba_bert).assign(model="DistilBERT (fine-tuned)")]
    )

    (CACHE / "api").mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": tfidf, "labels": labels}, CACHE / "api" / "tfidf_lr.joblib")

    for _, r in results.iterrows():
        log_result(f"routing/{r.model}/{r.protocol}", {"model": r.model, "protocol": r.protocol, "C": best_c if "TF" in r.model else None,
                                                        **({k_: v for k_, v in asdict(cfg).items()} if "Distil" in r.model else {})},
                   {"macro_f1": r.macro_f1})
    return {
        "splits": sp,
        "results": results,
        "tuning": tuning,
        "history": pd.DataFrame(history),
        "significance": pd.DataFrame([{"comparison": "DistilBERT - TF-IDF + LR", "metric": "macro_f1", **sig}]),
        "per_class": per_class,
        "by_length": pd.DataFrame(by_length),
        "confusion": cm_long,
        "top_confusions": top_confusions(cm, labels).assign(model=best_name),
        "risk_coverage": rc,
        "best_model": best_name,
        "best_c": best_c,
        "finetune_cfg": asdict(cfg),
        "history_key": key,
        "dropped_issues": sp.dropped.reset_index(),
    }


# --------------------------------------------------------------------------- emerging issues


def run_emerging(df: pd.DataFrame, minilm: Embedder, quick: bool, timer: Timer) -> dict:
    reps = df.drop_duplicates("group").reset_index(drop=True)  # earliest complaint of each group
    emb = timer.run("embed_minilm_s", minilm.encode_cached, reps.text.tolist(), reps.complaint_id.tolist(), gpu=True)
    k = N_TOPICS if not quick else 8
    topic_path = CACHE / f"topics__k{k}__{ids_key(reps.complaint_id.tolist())}.npy"
    if topic_path.exists():
        topics = np.load(topic_path)
    else:
        topics = timer.run("kmeans_s", kmeans_topics, emb, k)
        np.save(topic_path, topics)
    reps["topic"] = topics
    df["topic"] = df.group.map(dict(zip(reps.group, reps.topic)))
    terms = ctfidf_terms(reps.text.tolist(), topics, n_terms=6, min_df=5 if not quick else 2)

    g = df.groupby("topic")
    topic_table = pd.DataFrame(
        {
            "label": [", ".join(terms[t][:4]) for t in sorted(terms)],
            "top_terms": [", ".join(terms[t]) for t in sorted(terms)],
            "complaints": g.size(),
            "distinct_groups": reps.groupby("topic").size(),
            "share_in_campaign_groups": g.campaign.mean(),
            "top_issue": g.issue.agg(lambda s: s.value_counts().index[0]),
            "top_company": g.company.agg(lambda s: s.value_counts().index[0]),
            "share_2025_on": g.date.agg(lambda s: float((s >= TEST[0]).mean())),
        }
    )
    topic_table.index.name = "topic"

    months = pd.period_range(WINDOW[0], WINDOW[1], freq="M")
    counts = monthly_matrix(df.month, df.topic.to_numpy(), months)
    groups = monthly_matrix(reps.month, reps.topic.to_numpy(), months)
    cells = detect_bursts(counts, **BURST)
    pooled = detect_bursts(counts, baseline="pooled", **BURST)
    ablation = pd.DataFrame(
        [{"baseline": name, "burst_topic_months": int(c.burst.sum()),
          "bursts_feb_to_jul_2025": int((c.burst & c.month.between(pd.Period("2025-02", "M"), pd.Period("2025-07", "M"))).sum())}
         for name, c in [("median of monthly shares (used)", cells), ("pooled share of the window", pooled)]]
    )
    cells_groups = detect_bursts(groups, **BURST)
    cells["distinct_groups"] = [int(groups.loc[t, m]) for t, m in zip(cells.topic, cells.month)]
    cells["burst_in_distinct_groups"] = cells_groups.set_index(["topic", "month"]).loc[list(zip(cells.topic, cells.month)), "burst"].to_numpy()
    cells["label"] = cells.topic.map(topic_table.label)

    issue_codes = {lab: i for i, lab in enumerate(sorted(df.issue.unique()))}
    issue_counts = monthly_matrix(df.month, df.issue.map(issue_codes).to_numpy(), months)
    issue_cells = detect_bursts(issue_counts, **BURST)
    issue_cells["issue"] = issue_cells.topic.map({v: k_ for k_, v in issue_codes.items()})

    inject = timer.run(
        "injection_s", injection_recovery, counts, N_INJECT_CELLS if not quick else 10, INJECT_MULTIPLIERS,
        months[months >= pd.Period("2022-01", "M")], seed=0, **BURST,
    )
    for frame in (cells, issue_cells):
        frame["month"] = frame.month.astype(str)
    n_bursts = int(cells.burst.sum())
    log(f"emerging: {k} topics, {n_bursts} burst topic-months; issue-label baseline flags {int(issue_cells.burst.sum())}")
    log_result("emerging/kmeans+binomial", {"n_topics": k, **BURST}, {"bursts": n_bursts,
               "injection_recall_at_2x": float(inject[inject.multiplier == 2.0].detected.mean())})
    return {"topics": topic_table.reset_index(), "cells": cells, "issue_cells": issue_cells, "injection": inject, "ablation": ablation,
            "reps": reps, "rep_emb": emb}


# --------------------------------------------------------------------------- similar-case search


def run_search(routing: dict, emerging: dict, minilm: Embedder, quick: bool, timer: Timer) -> dict:
    sp = routing["splits"]
    corpus = pd.concat([sp.train, sp.val], ignore_index=True)
    queries = sp.test.sample(n=min(N_QUERIES if not quick else 50, len(sp.test)), random_state=0).reset_index(drop=True)
    yc, yq = corpus.y.to_numpy(), queries.y.to_numpy()
    sub_codes = {s: i for i, s in enumerate(sorted(set(corpus.sub_product) | set(queries.sub_product)))}
    sc, sq = corpus.sub_product.map(sub_codes).to_numpy(), queries.sub_product.map(sub_codes).to_numpy()
    depth = min(CAND_DEPTH, len(corpus))

    runs: dict[str, np.ndarray] = {}
    runs["TF-IDF"], _ = timer.run("search_tfidf_s", tfidf_search, corpus.text.tolist(), queries.text.tolist(), depth)

    rep_index = {cid: i for i, cid in enumerate(emerging["reps"].complaint_id)}
    need = [cid for cid in pd.concat([corpus.complaint_id, queries.complaint_id]) if cid not in rep_index]
    text_of = dict(zip(pd.concat([corpus.complaint_id, queries.complaint_id]), pd.concat([corpus.text, queries.text])))
    extra = timer.run("embed_minilm_s", minilm.encode_cached, [text_of[c] for c in need], need, gpu=True) if need else np.zeros((0, 384), np.float32)
    extra_index = {cid: i for i, cid in enumerate(need)}

    def minilm_rows(ids: pd.Series) -> np.ndarray:
        return np.vstack([emerging["rep_emb"][rep_index[c]] if c in rep_index else extra[extra_index[c]] for c in ids])

    emb_c, emb_q = minilm_rows(corpus.complaint_id), minilm_rows(queries.complaint_id)
    runs["MiniLM-L6"], _ = dense_search(emb_c, emb_q, depth)

    bge = Embedder(EMBEDDERS["bge-small"], CACHE)
    bge_c = timer.run("embed_bge_s", bge.encode_cached, corpus.text.tolist(), corpus.complaint_id.tolist(), gpu=True)
    bge_q = timer.run("embed_bge_s", bge.encode_cached, queries.text.tolist(), queries.complaint_id.tolist(), gpu=True)
    runs["BGE-small"], _ = dense_search(bge_c, bge_q, depth)
    del bge

    ce_path = CACHE / f"ce_rerank__bge-small__d{depth}__{ids_key(corpus.complaint_id.tolist())}__{ids_key(queries.complaint_id.tolist())}.npy"
    if ce_path.exists():
        runs["BGE-small + cross-encoder"] = np.load(ce_path)
    else:
        rr = Reranker()
        runs["BGE-small + cross-encoder"] = timer.run("rerank_s", rerank, rr, queries.text.tolist(), corpus.text.tolist(),
                                                      runs["BGE-small"], depth, gpu=True)
        np.save(ce_path, runs["BGE-small + cross-encoder"])
        del rr
    torch.cuda.empty_cache()

    per_query = {"Random (label prior)": {k_: random_precision(yq, yc) for k_ in KS}}
    per_query_sub = {"Random (label prior)": random_precision(sq, sc)}
    for name, idx in runs.items():
        per_query[name] = precision_at_k(idx, yq, yc, KS)
        per_query_sub[name] = precision_at_k(idx, sq, sc, [10])[10]

    rows = []
    rng = np.random.default_rng(0)
    for name, pk in per_query.items():
        row = {"method": name}
        for k_ in KS:
            boots = rng.choice(pk[k_], size=(2000, len(pk[k_]))).mean(axis=1)
            row[f"p@{k_}"] = float(pk[k_].mean())
            row[f"p@{k_}_ci_low"], row[f"p@{k_}_ci_high"] = np.percentile(boots, [2.5, 97.5])
        row["same_subproduct_p@10"] = float(per_query_sub[name].mean())
        rows.append(row)
        log_result(f"search/{name}", {"method": name, "n_queries": len(queries), "corpus": len(corpus)}, {f"p@{k_}": row[f"p@{k_}"] for k_ in KS})
    summary = pd.DataFrame(rows)

    pairs = [("TF-IDF", "MiniLM-L6"), ("TF-IDF", "BGE-small"), ("MiniLM-L6", "BGE-small"), ("BGE-small", "BGE-small + cross-encoder"),
             ("Random (label prior)", "TF-IDF")]
    tests = [(a, b, 10) for a, b in pairs] + [("BGE-small", "BGE-small + cross-encoder", 1)]
    sig = pd.DataFrame([{"baseline": a, "system": b, "metric": f"p@{k_}", **paired_bootstrap(per_query[a][k_], per_query[b][k_])} for a, b, k_ in tests])
    sig["p_holm"] = holm(sig.p_value.to_numpy())

    best = summary[summary.method != "Random (label prior)"].sort_values("p@10", ascending=False).method.iloc[0]
    examples = []
    for qi in range(3):
        q = queries.iloc[qi]
        for rank, ci in enumerate(runs[best][qi, :3], start=1):
            c = corpus.iloc[ci]
            examples.append({"query_id": q.complaint_id, "query_issue": q.issue, "query_start": q.text[:220], "rank": rank,
                             "match_id": c.complaint_id, "match_issue": c.issue, "match_date": f"{c.date:%Y-%m-%d}", "match_start": c.text[:220]})

    api_dir = CACHE / "api"
    corpus[["complaint_id", "date", "issue", "sub_product", "company", "text"]].to_parquet(api_dir / "corpus.parquet")
    np.save(api_dir / "corpus_bge-small.npy", bge_c)
    model_dir = CACHE / "models" / f"distilbert-{routing['history_key']}"
    (api_dir / "config.json").write_text(json.dumps({
        "embedder": "bge-small", "n_corpus": len(corpus), "labels": sp.labels,
        "distilbert_dir": model_dir.relative_to(ROOT).as_posix(), "distilbert_max_length": routing["finetune_cfg"]["max_length"],
    }, indent=2))
    log(f"search: best P@10 = {summary['p@10'].max():.3f} ({best}); random {summary.loc[0, 'p@10']:.3f}")
    return {"summary": summary, "significance": sig, "examples": pd.DataFrame(examples), "n_corpus": len(corpus), "n_queries": len(queries)}


# --------------------------------------------------------------------------- entities


def run_entities(routing: dict, quick: bool) -> dict:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    sample = routing["splits"].test.sample(n=N_ENTITY_DOCS if not quick else 30, random_state=1).reset_index(drop=True)
    ents = entities.extract(nlp, sample.narrative.tolist())
    per_doc, summary = entities.spot_check(sample, ents)
    label_counts = pd.Series([lab for doc in ents for lab, _ in doc]).value_counts().rename_axis("label").reset_index(name="count")
    examples = pd.DataFrame(
        [{"complaint_id": cid, "company": comp, "entities": "; ".join(f"{lab}: {t}" for lab, t in doc[:12])}
         for cid, comp, doc in zip(sample.complaint_id[:5], sample.company[:5], ents[:5])]
    )
    log(f"entities: {len(sample)} narratives; " + ", ".join(f"{r.check} = {r.value:.3f}" for r in summary.itertuples()))
    return {"summary": summary, "label_counts": label_counts, "examples": examples, "per_doc": per_doc}


# --------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="smoke test on a 6,000-complaint sample; outputs to data/cache/smoke/")
    parser.add_argument("--skip-live-check", action="store_true", help="don't call the live CFPB API")
    args = parser.parse_args()

    global RESULTS, FIGS, CACHE
    if args.quick:
        RESULTS = FIGS = ROOT / "data" / "cache" / "smoke"
        CACHE = ROOT / "data" / "cache" / "smoke_cache"
    for d in (CACHE, RESULTS, FIGS):
        d.mkdir(parents=True, exist_ok=True)
    init_mlflow(RESULTS if args.quick else ROOT)
    timer = Timer()
    t_all = time.perf_counter()

    df = timer.run("load_s", load, args.quick)
    df, minhash = timer.run("dedup_s", add_groups, df)
    volume = monthly_volume(df)
    groups = group_summary(df).head(25)

    live = None
    if not args.skip_live_check:
        try:
            live = timer.run("live_check_s", live_crosscheck, df, N_LIVE_CHECK if not args.quick else 3, ROOT / "data" / "cache" / "live_api_cache.json")
            log(f"live check: {live.drop(columns='complaint_id').mean().round(3).to_dict()}")
        except Exception as exc:  # the API being down shouldn't sink the run
            log(f"live check skipped: {exc}")

    routing = run_routing(df, args.quick, timer)
    minilm = Embedder(EMBEDDERS["minilm"], CACHE)
    emerging = run_emerging(df, minilm, args.quick, timer)
    search = run_search(routing, emerging, minilm, args.quick, timer)
    del minilm
    torch.cuda.empty_cache()
    ents = timer.run("entities_s", run_entities, routing, args.quick)

    # ---- write tables
    w = lambda frame, name, fmt="%.4f": frame.to_csv(RESULTS / name, index=False, float_format=fmt)  # noqa: E731
    w(volume, "monthly_volume.csv")
    w(groups.reset_index(), "near_duplicate_groups.csv")
    w(minhash, "minhash_check.csv")
    if live is not None:
        w(live, "live_api_crosscheck.csv")
    w(routing["results"], "routing_results.csv")
    w(routing["tuning"], "routing_tfidf_tuning.csv")
    w(routing["history"], "routing_distilbert_history.csv")
    w(routing["significance"], "routing_significance.csv")
    w(routing["per_class"], "routing_per_class.csv")
    w(routing["by_length"], "routing_by_length.csv")
    w(routing["confusion"], "routing_confusion.csv")
    w(routing["top_confusions"], "routing_top_confusions.csv")
    w(routing["risk_coverage"], "routing_risk_coverage.csv")
    w(routing["dropped_issues"], "routing_dropped_issues.csv")
    w(emerging["topics"], "topics.csv")
    w(emerging["cells"], "topic_months.csv")
    w(emerging["cells"][emerging["cells"].burst].sort_values(["month", "ratio"], ascending=[True, False]), "bursts.csv")
    w(emerging["issue_cells"][emerging["issue_cells"].burst], "issue_label_bursts.csv")
    w(emerging["injection"], "burst_injection.csv")
    w(emerging["ablation"], "burst_baseline_ablation.csv")
    w(search["summary"], "search_results.csv")
    w(search["significance"], "search_significance.csv")
    w(search["examples"], "search_examples.csv")
    w(ents["summary"], "entity_spotcheck.csv")
    w(ents["label_counts"], "entity_label_counts.csv")
    w(ents["examples"], "entity_examples.csv")

    timer.stages["total_s"] = time.perf_counter() - t_all
    sp = routing["splits"]
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quick": args.quick,
        "data": {
            "source": f"huggingface.co/datasets/{SNAPSHOT_REPO} (mirror of the CFPB bulk export, as of 2026-08-11)",
            "product": PRODUCT,
            "window": list(WINDOW),
            "complaints": int(len(df)),
            "issues_in_window": int(df.issue.nunique()),
            "near_duplicate_groups": int(df.group.nunique()),
            "complaints_in_campaign_groups": int(df.campaign.sum()),
            "campaign_group_size": CAMPAIGN_GROUP_SIZE,
            "live_check": None if live is None else {k: float(v) for k, v in live.drop(columns="complaint_id").mean().items()} | {"n": int(len(live))},
        },
        "routing": {
            "train": list(TRAIN), "test": list(TEST), "n_train": len(sp.train), "n_val": len(sp.val), "n_test": len(sp.test),
            "n_test_incl_copies": len(sp.test_all), "labels": sp.labels, "best_c": routing["best_c"], "best_model_on_val": routing["best_model"],
            "finetune": routing["finetune_cfg"],
        },
        "emerging": {"n_topics": N_TOPICS, "burst": BURST, "injection_multipliers": INJECT_MULTIPLIERS, "injection_cells": N_INJECT_CELLS},
        "search": {"n_corpus": search["n_corpus"], "n_queries": search["n_queries"], "candidate_depth": CAND_DEPTH,
                   "embedders": {k: v.model_id for k, v in EMBEDDERS.items()}, "cross_encoder": CROSS_ENCODER_ID},
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu": platform.processor(),
            "torch_threads": torch.get_num_threads(),
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "scikit_learn": sklearn.__version__,
            "spacy": spacy.__version__,
        },
        "timings_s": {k: round(v, 1) for k, v in timer.stages.items()},
        "gpu_stages_s_this_run": round(sum(timer.stages[s] for s in timer.gpu), 1),
    }
    (RESULTS / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    figs = make_all(RESULTS, FIGS)
    log_artifacts("summary", figs + [RESULTS / "routing_results.csv", RESULTS / "search_results.csv"])
    log(f"done in {timer.stages['total_s'] / 60:.1f} min (GPU stages {meta['gpu_stages_s_this_run'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
