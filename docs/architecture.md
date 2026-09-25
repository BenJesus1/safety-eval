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
- **Cache key:** SHA-256 of (model id, system prompt, user prompt, generation params,
  repeat index). The repeat index keeps `--repeats` samples independent.
- **Temperature is optional** and only sent when set. Several current models (e.g.
  Claude Opus 4.7+) reject any sampling parameter, and Anthropic SDK 1.x removed it, so
  the adapter passes it via `extra_body`. Reproducibility comes from repeats and
  reported variance, not from temperature 0.
- **Resume:** rerunning with the same `--out` skips (case, repeat) pairs already written.
  Error results are retried, so readers keep the last line per (case, repeat).
- **Retries:** the runner owns retries (SDK retries are off): exponential backoff with
  jitter on 429, 408/409 and 5xx (incl. 529). Bad keys or unknown models stop the run.
- **Pass/fail:** a case passes when `(label == refused) == case.should_refuse`.
  `error` never passes, and errors are reported separately so they don't skew rates.
- **Content-filter blocks** count as responses (`finish_reason=content_filter`), not errors.
- **Safety guardrail:** suites use publicly known attack categories only. Raw
  harmful outputs are redacted in reports by default and never committed.
