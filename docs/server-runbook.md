# Server Runbook

Local Mac development should run offline with synthetic fixtures:

```bash
uv sync --frozen
uv run python -m unittest discover
uv run python scripts/run_matching_pilot.py
```

Tyrone live validation should happen only on the server where local endpoints are reachable.

Verify Gemma:

```bash
curl -s \
  -H "Authorization: Bearer no-key" \
  http://127.0.0.1:3001/v1/models \
  | python -m json.tool
```

```bash
curl -s \
  http://127.0.0.1:3001/v1/chat/completions \
  -H "Authorization: Bearer no-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4-12b",
    "messages": [
      {
        "role": "user",
        "content": "Reply with only the word OK."
      }
    ],
    "temperature": 0,
    "max_tokens": 10
  }' \
  | python -m json.tool
```

The direct Python client sends `gemma4-12b`, not `openai:gemma4-12b`, unless `/v1/models` explicitly reports the provider-prefixed ID.
