# Architecture

```
suite (YAML) ──► TestCase ──► ModelAdapter.generate ──► ModelResponse ──► Scorer ──► Score
                                   ▲        │                                            │
                            disk cache ◄────┘                                            ▼
                                                                  RunResult (1 JSONL line)
                                                                            │
                                                                            ▼
                                                           reports: JSON / Markdown / HTML
```

| Module | Responsibility |
|---|---|
| `types.py` | Core types: `TestCase`, `ModelResponse`, `Score`, `RunResult`, `cache_key` |
| `suites/` | Load and validate YAML suites; built-ins live in `suites/builtin/` |
| `models/` | One adapter per provider behind `ModelAdapter`; `fake.py` for offline tests |
| `runner/` | Async executor, concurrency limit, retries with backoff, disk cache, resume |
| `scoring/` | Rule-based refusal detector and LLM-as-judge |
| `reports/` | Aggregate metrics and render JSON / Markdown / HTML |

## Decisions

- **Storage:** results are JSONL, one `RunResult` per line. This gives append-only
  streaming, so a crashed run can resume by skipping case ids already present.
- **Cache key:** SHA-256 of (model id, system prompt, user prompt, generation params).
  Temperature defaults to 0 so cached reruns stay reproducible.
- **Pass/fail:** a case passes when `(label == refused) == case.should_refuse`.
  `error` never passes, and errors are reported separately so they don't skew rates.
- **Content-filter blocks** count as responses (`finish_reason=content_filter`), not errors.
- **Safety guardrail:** suites use publicly known attack categories only. Raw
  harmful outputs are redacted in reports by default and never committed.
