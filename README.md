# safety-eval

Reproducible safety evaluations for LLMs. One command measures **jailbreak
resistance** and **over-refusal** together, so you can see the trade-off between them.

> Status: pre-alpha (Phase 0). The API and CLI are not stable yet.

## Quickstart (dev)

```bash
brew install uv
uv sync
uv run pytest
uv run safety-eval --help
```

Copy `.env.example` to `.env` and add your API keys before running against real models.

## Usage

```bash
uv run safety-eval list-suites
uv run safety-eval list-models
uv run safety-eval run --suite over_refusal --model anthropic:claude-haiku-4-5 --dry-run
uv run safety-eval run --suite over_refusal --model anthropic:claude-haiku-4-5 --repeats 3
```

Results stream to `results/<suite>__<model>.jsonl`, one line per (case, repeat).
Responses are cached in `.safety_eval_cache/`, so reruns are free, and an interrupted
run resumes where it stopped.

## Roadmap

- [ ] v0.1: 3 suites, Anthropic/OpenAI/Ollama adapters, rule + LLM judge scoring, HTML report
- [ ] v0.2: multi-turn attacks, more categories, leaderboard page

See [docs/architecture.md](docs/architecture.md).

## License

MIT
