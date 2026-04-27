"""HTTP client for the Astar Island competition API.

Wraps all endpoints at api.ainm.no/astar-island/ with:
- Rate limiting (max 4 req/s, under the 5 req/s server limit)
- Budget tracking (local + server-synced)
- Auth expiry detection (raises on 401)
- Retry with backoff on 429
"""

import os
import time
import logging

import httpx

from dtos import (
    RoundSummary,
    RoundInfo,
    BudgetInfo,
    SimResult,
    AnalysisResult,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://api.ainm.no"


class BudgetExhaustedError(Exception):
    """Raised when the query budget is exhausted."""
    pass


class AuthExpiredError(Exception):
    """Raised when the JWT token has expired or is invalid."""
    pass


class RoundNotActiveError(Exception):
    """Raised when trying to query a round that isn't active."""
    pass


class AstarClient:
    """Client for the Astar Island competition API.

    ┌──────────────────────────────────────────────┐
    │  Request Flow                                 │
    │                                               │
    │  caller ──> rate_limit_wait ──> http_request  │
    │                                  │            │
    │              ┌───────────────────┤            │
    │              │                   │            │
    │           401 → AuthExpiredError │            │
    │           429 → backoff + retry  │            │
    │           400 → RoundNotActive   │            │
    │           2xx → parse + return   │            │
    └──────────────────────────────────────────────┘
    """

    def __init__(self, token: str | None = None):
        self._token = token or os.environ.get("ASTAR_TOKEN", "")
        if not self._token:
            raise ValueError(
                "No auth token provided. Set ASTAR_TOKEN env var or pass token= argument."
            )
        self._session = httpx.Client(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=30.0,
        )
        self._queries_used = 0
        self._queries_max = 50
        self._last_request_time = 0.0
        self._min_interval = 0.25  # 4 req/s (safely under 5 req/s limit)
        self._max_retries = 3

    def _wait_for_rate_limit(self):
        """Sleep if needed to respect rate limits."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def _handle_response(self, resp: httpx.Response, retries_left: int = 0) -> dict:
        """Handle common error codes."""
        if resp.status_code == 401:
            raise AuthExpiredError(
                "JWT token expired or invalid. Refresh your token from app.ainm.no "
                "and update ASTAR_TOKEN."
            )
        if resp.status_code == 429:
            if retries_left > 0:
                wait = 2.0 ** (self._max_retries - retries_left)
                logger.warning(f"Rate limited (429). Retrying in {wait}s...")
                time.sleep(wait)
                return None  # Signal retry
            raise httpx.HTTPStatusError(
                "Rate limit exceeded after retries",
                request=resp.request,
                response=resp,
            )
        if resp.status_code == 400:
            detail = resp.json().get("detail", resp.text)
            raise RoundNotActiveError(f"Bad request: {detail}")
        resp.raise_for_status()
        return resp.json()

    # ── Public endpoints ──────────────────────────────────────

    def get_rounds(self) -> list[RoundSummary]:
        """List all rounds."""
        self._wait_for_rate_limit()
        resp = self._session.get("/astar-island/rounds")
        data = self._handle_response(resp)
        return [RoundSummary(**r) for r in data]

    def get_round_detail(self, round_id: str) -> RoundInfo:
        """Get round details including initial states for all seeds."""
        self._wait_for_rate_limit()
        resp = self._session.get(f"/astar-island/rounds/{round_id}")
        data = self._handle_response(resp)
        return RoundInfo(**data)

    def get_budget(self) -> BudgetInfo:
        """Check remaining query budget for the active round."""
        self._wait_for_rate_limit()
        resp = self._session.get("/astar-island/budget")
        data = self._handle_response(resp)
        budget = BudgetInfo(**data)
        self._queries_used = budget.queries_used
        self._queries_max = budget.queries_max
        return budget

    def simulate(
        self,
        round_id: str,
        seed_index: int,
        viewport_x: int = 0,
        viewport_y: int = 0,
        viewport_w: int = 15,
        viewport_h: int = 15,
    ) -> SimResult:
        """Run one simulation query. Costs 1 from the 50-query budget.

        Raises BudgetExhaustedError if budget is exhausted.
        Retries on 429 with exponential backoff.
        """
        if self._queries_used >= self._queries_max:
            raise BudgetExhaustedError(
                f"Budget exhausted: {self._queries_used}/{self._queries_max} queries used."
            )

        payload = {
            "round_id": round_id,
            "seed_index": seed_index,
            "viewport_x": viewport_x,
            "viewport_y": viewport_y,
            "viewport_w": viewport_w,
            "viewport_h": viewport_h,
        }

        for attempt in range(self._max_retries):
            self._wait_for_rate_limit()
            resp = self._session.post("/astar-island/simulate", json=payload)
            retries_left = self._max_retries - attempt - 1
            data = self._handle_response(resp, retries_left=retries_left)
            if data is not None:
                break
        else:
            raise httpx.HTTPStatusError(
                "Simulate failed after all retries",
                request=resp.request,
                response=resp,
            )

        result = SimResult(**data)
        self._queries_used = result.queries_used
        self._queries_max = result.queries_max
        logger.info(
            f"Query {self._queries_used}/{self._queries_max}: "
            f"seed={seed_index} viewport=({viewport_x},{viewport_y},{viewport_w},{viewport_h}) "
            f"settlements={len(result.settlements)}"
        )
        return result

    def submit(
        self,
        round_id: str,
        seed_index: int,
        prediction: list[list[list[float]]],
    ) -> dict:
        """Submit prediction tensor for one seed.

        prediction: H×W×6 tensor where each cell's 6 probabilities sum to 1.0.
        """
        payload = {
            "round_id": round_id,
            "seed_index": seed_index,
            "prediction": prediction,
        }

        for attempt in range(self._max_retries):
            self._wait_for_rate_limit()
            resp = self._session.post("/astar-island/submit", json=payload)
            retries_left = self._max_retries - attempt - 1
            data = self._handle_response(resp, retries_left=retries_left)
            if data is not None:
                break
        else:
            raise httpx.HTTPStatusError(
                "Submit failed after all retries",
                request=resp.request,
                response=resp,
            )

        logger.info(f"Submitted seed {seed_index}: {data.get('status', 'unknown')}")
        return data

    def get_analysis(self, round_id: str, seed_index: int) -> AnalysisResult:
        """Get ground truth analysis for a completed round."""
        self._wait_for_rate_limit()
        resp = self._session.get(
            f"/astar-island/analysis/{round_id}/{seed_index}"
        )
        data = self._handle_response(resp)
        return AnalysisResult(**data)

    def get_my_rounds(self) -> list[dict]:
        """Get all rounds enriched with team scores and budget info."""
        self._wait_for_rate_limit()
        resp = self._session.get("/astar-island/my-rounds")
        data = self._handle_response(resp)
        return data

    @property
    def queries_used(self) -> int:
        return self._queries_used

    @property
    def queries_remaining(self) -> int:
        return self._queries_max - self._queries_used
