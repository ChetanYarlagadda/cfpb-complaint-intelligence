"""FastAPI service: route a complaint narrative to an issue queue and find similar past cases.

    uvicorn app.main:app --port 8000      # Swagger UI at http://localhost:8000/docs
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from app.engine import Engine, build_default_engine

EXAMPLE = (
    "I sent $450 through Zelle to pay for a puppy advertised online. After I paid, the seller stopped responding. "
    "My bank says it cannot reverse the payment because I authorized it."
)


class RouteRequest(BaseModel):
    text: str = Field(min_length=20, max_length=20_000, examples=[EXAMPLE], description="complaint narrative")
    k: int = Field(3, ge=1, le=13, description="number of issues to return")


class IssueOut(BaseModel):
    issue: str
    probability: float


class RouteResponse(BaseModel):
    router: str
    latency_ms: float
    issues: list[IssueOut]


class SimilarRequest(BaseModel):
    text: str = Field(min_length=20, max_length=20_000, examples=[EXAMPLE], description="complaint narrative")
    k: int = Field(5, ge=1, le=50, description="number of similar complaints")
    rerank: bool = Field(False, description="reorder candidates with the cross-encoder")


class CaseOut(BaseModel):
    complaint_id: str
    date: str
    issue: str
    sub_product: str
    company: str
    snippet: str
    score: float


class SimilarResponse(BaseModel):
    reranked: bool = Field(description="true if the cross-encoder reordered the cases and `score` is its logit; false if `score` is cosine")
    latency_ms: float
    cases: list[CaseOut]


def create_app(engine_factory: Callable[[], Engine]) -> FastAPI:
    """Build the app; the engine is created once at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine_factory()
        yield

    app = FastAPI(
        title="Fintech Complaint Intelligence",
        description="Issue routing and similar-case search for money-transfer and payment-app complaints (CFPB data).",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/route", response_model=RouteResponse)
    def route(req: RouteRequest, request: Request) -> RouteResponse:
        t0 = time.perf_counter()
        engine = request.app.state.engine
        issues = engine.route(req.text, k=req.k)
        return RouteResponse(
            router=engine.router_name,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            issues=[IssueOut(**i.__dict__) for i in issues],
        )

    @app.post("/similar", response_model=SimilarResponse)
    def similar(req: SimilarRequest, request: Request) -> SimilarResponse:
        t0 = time.perf_counter()
        engine = request.app.state.engine
        cases = engine.similar(req.text, k=req.k, rerank=req.rerank)
        return SimilarResponse(
            reranked=req.rerank and engine.can_rerank,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            cases=[CaseOut(**c.__dict__) for c in cases],
        )

    return app


app = create_app(build_default_engine)
