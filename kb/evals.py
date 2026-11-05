"""Evaluation harness: measure retrieval and answer quality on a labeled question set.

Each line of the JSONL file:
  {"question": "...", "expected_source": "refunds.md", "must_include": ["14 days"]}
  {"question": "...", "should_refuse": true}
  {"question": "...", "allow_refusal": true, "must_not_include": ["free forever"]}   # injection test

Metrics
- retrieval_hit_rate: the expected source is among the retrieved chunks
- citation_accuracy: the answer cites the expected source
- answer_contains: every `must_include` phrase appears in the answer
- refusal_accuracy: out-of-scope questions are refused; in-scope ones are not
- grounded_rate: answers whose citations are all valid
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from .rag import Assistant


@dataclass
class CaseResult:
    question: str
    passed: bool
    retrieval_hit: bool | None
    cited_expected: bool | None
    contains: bool | None
    refusal_ok: bool
    answer: str
    sources: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    cases: list[CaseResult]
    metrics: dict

    def summary(self) -> str:
        m = self.metrics
        lines = [f"Cases: {m['cases']}  Passed: {m['passed']} ({m['pass_rate']:.0%})"]
        for k in ("retrieval_hit_rate", "citation_accuracy", "answer_contains", "refusal_accuracy", "grounded_rate"):
            if m.get(k) is not None:
                lines.append(f"  {k:<20} {m[k]:.0%}")
        lines.append(f"  median_latency_ms    {m['median_latency_ms']}")
        lines.append(f"  total_cost_usd       {m['total_cost_usd']:.5f}")
        fails = [c for c in self.cases if not c.passed]
        if fails:
            lines.append("Failures:")
            lines += [f"  - {c.question}\n      -> {c.answer[:160]}" for c in fails]
        return "\n".join(lines)


def _rate(vals: list[bool]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def run_eval(assistant: Assistant, org_id: str, path: str | Path) -> EvalReport:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    cases, hits, cites, contains, refusals, grounded, lat, cost = [], [], [], [], [], [], [], 0.0
    for r in rows:
        should_refuse = bool(r.get("should_refuse"))
        retrieved = [c.source for c, _ in assistant.retrieve(org_id, r["question"])]
        ans = assistant.ask(org_id, r["question"])
        lat.append(ans.latency_ms)
        cost += ans.cost_usd
        grounded.append(ans.grounded)
        refusal_ok = True if r.get("allow_refusal") else ans.refused == should_refuse
        refusals.append(refusal_ok)
        if any(p.lower() in ans.text.lower() for p in r.get("must_not_include", [])):
            refusal_ok = False  # e.g. the answer repeated an injected instruction

        hit = cited = inc = None
        if not should_refuse:
            exp = r.get("expected_source")
            if exp:
                hit = exp in retrieved
                cited = exp in {c.source for c in ans.citations}
                hits.append(hit)
                cites.append(cited)
            if r.get("must_include"):
                inc = all(p.lower() in ans.text.lower() for p in r["must_include"])
                contains.append(inc)

        passed = refusal_ok and ans.grounded and all(v is not False for v in (hit, cited, inc))
        cases.append(CaseResult(r["question"], passed, hit, cited, inc, refusal_ok, ans.text, retrieved))

    metrics = {
        "cases": len(cases),
        "passed": sum(c.passed for c in cases),
        "pass_rate": sum(c.passed for c in cases) / max(1, len(cases)),
        "retrieval_hit_rate": _rate(hits),
        "citation_accuracy": _rate(cites),
        "answer_contains": _rate(contains),
        "refusal_accuracy": _rate(refusals),
        "grounded_rate": _rate(grounded),
        "median_latency_ms": int(statistics.median(lat)) if lat else 0,
        "total_cost_usd": cost,
    }
    return EvalReport(cases, metrics)
