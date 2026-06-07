# Cosavu — ContextAPI (Stan)

Optimize bloated prompts into lean, **same-intent** prompts before you send them
to a large LLM (GPT / Claude / etc.). ContextAPI sits between your app and your
model: you send it a wordy prompt, it returns a tighter one with the same intent
and fewer tokens — so the downstream call costs less and runs faster.

Powered by Cosavu's **Stan** prompt-optimization models.

## Base URL

```
https://api.cosavu.com
```

## Authentication

Every request needs your Cosavu API token, sent in the `X-API-Token` header:

```
X-API-Token: csvu_your_token_here
```

Keep your token secret. Don't commit it — load it from an environment variable
(`COSAVU_API_KEY`) instead.

## Install

```bash
pip install -r requirements.txt
```

The client (`cosavu_context`) only depends on `requests`.

## Quickstart

```python
from cosavu_context import Cosavu

cosavu = Cosavu()  # reads COSAVU_API_KEY from the environment
result = cosavu.optimize("Hey so um could you maybe help me summarize this ...")

print(result.text)            # the lean prompt -> forward this to your LLM
print(result.tokens_saved)    # e.g. 18
print(f"{result.reduction:.0%}")  # e.g. 34%
```

One-shot helper, no client to manage:

```python
from cosavu_context import optimize

lean = optimize("please could you kindly walk me through ...").text
```

## Models

Pick a tier with `model=` (default is `stan-1.5-mini-thinking`):

| Model                       | Speed    | Best for                                                        |
| --------------------------- | -------- | -------------------------------------------------------------- |
| `stan-1.5-mini-instant`     | fastest  | latency-sensitive paths and testing; aggressive, terse output  |
| `stan-1.5-mini-thinking`    | balanced | general purpose — solid balance of quality and speed (default) |
| `stan-1.5-mini-predictive`  | richest  | complex prompts; also adds helpful implied context             |

Convenience constants are exported:

```python
from cosavu_context import Cosavu, INSTANT, THINKING, PREDICTIVE

cosavu = Cosavu()
result = cosavu.optimize(prompt, model=PREDICTIVE)
```

## The result object

`optimize()` returns an `OptimizedContext`:

| Attribute          | Type    | Description                                   |
| ------------------ | ------- | --------------------------------------------- |
| `text`             | `str`   | The optimized prompt. Send this to your LLM.  |
| `model`            | `str`   | The Stan tier that produced it.               |
| `original_tokens`  | `int`   | Estimated tokens of the input.                |
| `optimized_tokens` | `int`   | Estimated tokens of `text`.                   |
| `tokens_saved`     | `int`   | `original_tokens - optimized_tokens`.         |
| `reduction`        | `float` | Fraction of input tokens removed (0..1).      |
| `compression`      | `float` | Target compression applied (0..1), if known.  |
| `latency_ms`       | `float` | Server-side optimization time.                |

`str(result)` is the optimized text, so you can pass `result` straight into an
f-string.

## Typical usage pattern

```python
from cosavu_context import Cosavu

cosavu = Cosavu()

# 1) optimize the prompt
lean = cosavu.optimize(user_prompt).text

# 2) forward the lean prompt to your model of choice (pseudo-code)
openai.chat.completions.create(
    model="gpt-5",
    messages=[{"role": "user", "content": lean}],
)
# -> you just trimmed the input tokens on that call.
```

## Configuration

| Argument / env          | Default                   | Purpose                          |
| ----------------------- | ------------------------- | -------------------------------- |
| `api_key` / `COSAVU_API_KEY` | —                    | Your Cosavu API token (required).|
| `base_url` / `COSAVU_API_URL`| `https://api.cosavu.com` | Override the API host.        |
| `timeout`               | `120`                     | Per-request timeout (seconds).   |
| `max_retries`           | `2`                       | Retries on 429 / 5xx / transport.|

```python
cosavu = Cosavu(api_key="csvu_...", timeout=60, max_retries=3)
```

The client is also a context manager:

```python
with Cosavu() as cosavu:
    result = cosavu.optimize(prompt)
```

## Errors

All client errors derive from `CosavuError`:

| Exception          | When                                          |
| ------------------ | --------------------------------------------- |
| `AuthError`        | Missing or invalid API token (401 / 403).     |
| `BadRequestError`  | Empty prompt, unknown model, malformed (400). |
| `RateLimitError`   | Rate limit exceeded after retries (429).      |
| `APIError`         | Server or transport failure after retries.    |

```python
from cosavu_context import optimize, CosavuError

try:
    lean = optimize(prompt).text
except CosavuError as e:
    print("optimization failed:", e)
```

## Local development

Point the client at a locally running instance:

```python
cosavu = Cosavu(base_url="http://localhost:2026")
```

Everything else (headers, request/response shapes) stays the same.
