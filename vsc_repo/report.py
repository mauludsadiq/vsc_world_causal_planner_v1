from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

@dataclass
class CheckResult:
    name: str
    passed: bool
    metrics: Dict[str, Any]
    message: str = ""

class Reporter:
    def __init__(self) -> None:
        self.results: List[CheckResult] = []

    def add(self, name: str, passed: bool, metrics: Dict[str, Any], message: str = "") -> None:
        self.results.append(CheckResult(name=name, passed=bool(passed), metrics=dict(metrics), message=message))

    def summary(self) -> Dict[str, Any]:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        return {"total": total, "passed": passed, "failed": failed, "all_passed": failed == 0}

    def print(self) -> None:
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            metric_str = " ".join(f"{k}={r.metrics[k]}" for k in sorted(r.metrics.keys()))
            if r.message:
                print(f"[{status}] {r.name}: {metric_str} :: {r.message}")
            else:
                print(f"[{status}] {r.name}: {metric_str}")
        s = self.summary()
        print(f"SUMMARY: total={s['total']} passed={s['passed']} failed={s['failed']} all_passed={s['all_passed']}")

    def write_json(self, out_path: str) -> None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        payload = {"summary": self.summary(), "results": [asdict(r) for r in self.results]}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
