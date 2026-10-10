from fastapi.testclient import TestClient

from app.engine import Case, IssueScore
from app.main import create_app


class FakeEngine:
    router_name = "fake"

    def route(self, text: str, k: int) -> list[IssueScore]:
        return [IssueScore(f"issue {i}", 1.0 / (i + 2)) for i in range(k)]

    def similar(self, text: str, k: int, rerank: bool) -> list[Case]:
        return [Case(str(i), "2024-01-01", "Fraud or scam", "Mobile or digital wallet", "Co", text[:10], 1.0 - i / 10) for i in range(k)]


def client() -> TestClient:
    return TestClient(create_app(FakeEngine))


TEXT = "I sent money with a payment app and never received the item I paid for."


def test_health():
    with client() as c:
        assert c.get("/health").json() == {"status": "ok"}


def test_route_returns_k_issues():
    with client() as c:
        r = c.post("/route", json={"text": TEXT, "k": 2})
        assert r.status_code == 200
        body = r.json()
        assert body["router"] == "fake"
        assert [i["issue"] for i in body["issues"]] == ["issue 0", "issue 1"]


def test_similar_returns_cases():
    with client() as c:
        r = c.post("/similar", json={"text": TEXT, "k": 3, "rerank": True})
        assert r.status_code == 200
        body = r.json()
        assert body["reranked"] is True and len(body["cases"]) == 3


def test_requests_are_validated():
    with client() as c:
        assert c.post("/route", json={"text": "too short"}).status_code == 422
        assert c.post("/similar", json={"text": TEXT, "k": 500}).status_code == 422
