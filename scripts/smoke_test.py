"""Smoke-test a running prediction API end to end.

    uv run python scripts/smoke_test.py http://127.0.0.1:8000

Used in CI against the freshly built container before it's published, and usable against
any deployment. Waits for the service to come up (containers and sleeping free-tier hosts
take a while), then checks each endpoint with a real request.
"""

import argparse
import sys
import time

import requests

from readmission.api.schemas import EXAMPLE_ENCOUNTER


def wait_until_healthy(base_url: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        try:
            response = requests.get(f"{base_url}/health", timeout=10)
            if response.status_code == 200:
                return response.json()
        except requests.RequestException:
            pass
        if time.monotonic() > deadline:
            sys.exit(f"FAIL  {base_url}/health not healthy after {timeout:.0f}s")
        time.sleep(2)


def check(condition: bool, message: str) -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {message}")
    if not condition:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url", help="e.g. http://127.0.0.1:8000")
    parser.add_argument(
        "--timeout", type=float, default=120, help="seconds to wait for /health"
    )
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    health = wait_until_healthy(base, args.timeout)
    check(
        health.get("status") == "ok",
        f"/health ok (model {health.get('model_version')})",
    )

    info = requests.get(f"{base}/model-info", timeout=10).json()
    check(
        set(info.get("risk_tier_thresholds", {})) == {"moderate", "high"},
        f"/model-info reports tier thresholds {info.get('risk_tier_thresholds')}",
    )

    response = requests.post(f"{base}/predict", json=EXAMPLE_ENCOUNTER, timeout=30)
    body = response.json()
    check(response.status_code == 200, "/predict accepts the example encounter")
    check(
        0.0 <= body["readmission_risk"] <= 1.0
        and body["risk_tier"] in {"low", "moderate", "high"},
        f"/predict risk {body['readmission_risk']:.4f} ({body['risk_tier']})",
    )
    check(len(body["top_factors"]) == 3, "/predict explains the score with 3 factors")

    invalid = dict(EXAMPLE_ENCOUNTER, diag_1="I50.9")  # an ICD-10 code
    response = requests.post(f"{base}/predict", json=invalid, timeout=30)
    check(response.status_code == 422, "/predict rejects an ICD-10 code with 422")

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
