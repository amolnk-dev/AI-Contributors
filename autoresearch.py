"""API utilities: health checks, testing, profiling."""

import time
import httpx


def check_health(base_url: str, timeout: float = 5.0) -> bool:
    """Check if an endpoint is running."""
    try:
        r = httpx.get(f"{base_url}/", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def test_predict(base_url: str, payload: dict, timeout: float = 30.0) -> dict:
    """Send a predict request and return response with timing."""
    start = time.time()
    try:
        r = httpx.post(f"{base_url}/predict", json=payload, timeout=timeout)
        elapsed = time.time() - start
        return {
            "status": r.status_code,
            "response": r.json(),
            "latency_ms": round(elapsed * 1000, 1),
            "ok": r.status_code == 200,
        }
    except Exception as e:
        return {"status": -1, "error": str(e), "ok": False}


def profile_endpoint(base_url: str, payload: dict, n_requests: int = 10) -> dict:
    """Profile endpoint latency over multiple requests."""
    latencies = []
    errors = 0
    for _ in range(n_requests):
        result = test_predict(base_url, payload)
        if result["ok"]:
            latencies.append(result["latency_ms"])
        else:
            errors += 1
    return {
        "n_requests": n_requests,
        "errors": errors,
        "mean_ms": round(sum(latencies) / max(len(latencies), 1), 1),
        "min_ms": round(min(latencies), 1) if latencies else None,
        "max_ms": round(max(latencies), 1) if latencies else None,
        "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0, 1),
    }
