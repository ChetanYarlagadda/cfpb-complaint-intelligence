"""Smoke-test the API in-process against the real pipeline artifacts and record latencies.

    python scripts/api_smoke.py      # needs data/cache/api/ from scripts/run_pipeline.py

Writes reports/results/api_smoke.json.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.engine import ComplaintEngine  # noqa: E402
from app.main import EXAMPLE, create_app  # noqa: E402

N_REPEATS = 20


def timed(client: TestClient, url: str, body: dict) -> tuple[dict, list[float]]:
    """Response of the last call and the wall-clock latency (ms) of each call after one warm-up."""
    client.post(url, json=body)
    lat = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        r = client.post(url, json=body)
        lat.append((time.perf_counter() - t0) * 1000)
        r.raise_for_status()
    return r.json(), lat


if __name__ == "__main__":
    out = {"example": EXAMPLE}
    with TestClient(create_app(lambda: ComplaintEngine(router="tfidf", reranker=True))) as c:
        route, lat_route = timed(c, "/route", {"text": EXAMPLE, "k": 3})
        sim, lat_sim = timed(c, "/similar", {"text": EXAMPLE, "k": 3, "rerank": False})
        sim_rr, lat_sim_rr = timed(c, "/similar", {"text": EXAMPLE, "k": 3, "rerank": True})
    out["route_top3"] = route["issues"]
    out["similar_top3_issues"] = [h["issue"] for h in sim["cases"]]
    out["similar_reranked_top3_issues"] = [h["issue"] for h in sim_rr["cases"]]
    out["median_latency_ms"] = {
        "route_tfidf": float(np.median(lat_route)),
        "similar": float(np.median(lat_sim)),
        "similar_rerank_top20": float(np.median(lat_sim_rr)),
    }
    path = ROOT / "reports" / "results" / "api_smoke.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
