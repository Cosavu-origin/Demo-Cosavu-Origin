"""Rich terminal chat for the OriginChain → ContextAPI → NGen pipeline.

Each turn shows the internals:
  • Source Context        — raw result from the OriginChain tool call
  • Optimized Context     — what Cosavu's ContextAPI sent to the LLM (+ savings)
  • Sampling Parameters   — Stan's temperature / top_p / reasoning mode
  • Thinking              — the answer model's (NGen-4 Mini) reasoning
  • Answer                — the final reply

    pip install -r requirements.txt        # groq + requests + rich (+ Node.js for npx)
    export OC_HOST=... OC_TENANT=... OC_TOKEN=...
    export GROQ_API_KEY=gsk_...            # tool calling
    export COSAVU_API_KEY=csvu_...         # ContextAPI
    export TNSA_API_KEY=tnsa_...           # optional, for NGen
    python chat.py
"""

from __future__ import annotations

import os
import re
import sys
import time

# Windows consoles default to cp1252 and crash on the box-drawing / arrow glyphs
# Rich emits. Force UTF-8 so rendering never errors on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from origin_chain import (
    DEFAULT_MODEL,
    DEFAULT_TOOL_MODEL,
    OriginChainError,
    OriginChainPipeline,
    load_env,
)
from origin_chain.pipeline import ToolStep

console = Console()
MAX_CTX = 2000  # clip very long context blocks for display only

# NGen's closed agent sometimes injects tool-status placeholders into its text.
_PLACEHOLDER_RE = re.compile(
    r"\[\s*(?:rolling dice|searching|loading|fetching|processing|running|thinking|"
    r"generating)[^\]]*\]\s*",
    re.IGNORECASE,
)


def _clean_answer(s: str) -> str:
    return _PLACEHOLDER_RE.sub("", s or "").strip()


def _clip(s: str, n: int = MAX_CTX) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + f"\n… [{len(s) - n} more chars]"


def _context_columns(step: ToolStep) -> Columns:
    source = Panel(
        _clip(step.source_context) or "[dim](empty)[/dim]",
        title="[bold]Source Context[/bold]  [dim](OriginChain)[/dim]",
        border_style="yellow",
    )
    if step.optimized:
        sub = (f"{step.original_tokens} → {step.optimized_tokens} tokens · "
               f"saved {step.tokens_saved}")
        opt = Panel(
            _clip(step.optimized_context) or "[dim](empty)[/dim]",
            title="[bold]Optimized Context[/bold]  [dim](ContextAPI → LLM)[/dim]",
            subtitle=sub,
            border_style="green",
        )
    else:
        opt = Panel(
            "[dim]passthrough — result below the optimize threshold, sent as-is "
            "to avoid dropping data[/dim]",
            title="[bold]Optimized Context[/bold]  [dim](ContextAPI → LLM)[/dim]",
            border_style="dim",
        )
    return Columns([source, opt], equal=True, expand=True)


def _sampling_table(step: ToolStep) -> Table:
    t = Table(title="Sampling Parameters (Stan)", title_style="bold magenta",
              show_header=True, header_style="magenta", expand=False)
    t.add_column("parameter", style="cyan")
    t.add_column("value", style="white")
    s = step.sampling or {}
    rows = [
        ("temperature", s.get("temperature")),
        ("top_p", s.get("top_p")),
        ("reasoning_mode", s.get("reasoning_mode")),
        ("compression", f"{step.compression:.0%}" if step.compression is not None else None),
        ("jepa_recoverability",
         f"{s['jepa_recoverability']:.0%}" if s.get("jepa_recoverability") is not None else None),
    ]
    any_val = False
    for k, v in rows:
        if v is not None:
            t.add_row(k, str(v))
            any_val = True
    if not any_val:
        t.add_row("[dim]n/a[/dim]", "[dim]result not optimized[/dim]")
    return t


def render(question: str, result, elapsed: float | None = None) -> None:
    console.print(Rule(f"[bold cyan]{question}[/bold cyan]", style="cyan"))

    if not result.steps:
        console.print(Panel("[dim]no OriginChain tool was called for this turn[/dim]",
                            border_style="dim"))
    for i, step in enumerate(result.steps, 1):
        args = ", ".join(f"{k}={v!r}" for k, v in (step.arguments or {}).items()) or "(no args)"
        console.print(Panel(Text(args, style="white"),
                            title=f"[bold blue]Tool call {i}[/bold blue] · {step.tool}",
                            border_style="blue"))
        console.print(_context_columns(step))
        console.print(_sampling_table(step))

    if result.thinking:
        console.print(Panel(_clip(result.thinking),
                            title="[bold]Thinking[/bold]  [dim](NGen-4 Mini)[/dim]",
                            border_style="grey50"))

    console.print(Panel(Markdown(_clean_answer(result.answer) or "*(no answer)*"),
                        title="[bold green]Answer[/bold green]  [dim](NGen-4 Mini)[/dim]",
                        border_style="green"))

    calls = len(result.steps)
    saved = result.tokens_saved
    console.print(
        f"[dim]{calls} tool call{'s' if calls != 1 else ''} · "
        f"{saved} token{'s' if saved != 1 else ''} saved by ContextAPI"
        + (f" · {elapsed:.1f}s" if elapsed is not None else "") + "[/dim]\n"
    )


def main() -> None:
    loaded = load_env()  # auto-load origin-api/.env (and parents)

    missing = [k for k in ("OC_HOST", "OC_TENANT", "OC_TOKEN", "GROQ_API_KEY", "COSAVU_API_KEY")
               if not os.environ.get(k)]
    if missing:
        console.print(f"[red]Missing environment variables:[/red] {', '.join(missing)}")
        console.print("Set them in [bold]origin-api/.env[/bold] (see [bold].env.example[/bold]) "
                      "or export them.")
        return

    src = f"  [dim](loaded {loaded[0]})[/dim]" if loaded else ""
    console.print(Panel.fit(
        "[bold]OriginChain Chat[/bold]\n"
        f"tool calls: [cyan]{DEFAULT_TOOL_MODEL}[/cyan] (Groq)   "
        f"answer: [cyan]{DEFAULT_MODEL}[/cyan] (NGen)   optimizer: [cyan]Cosavu ContextAPI[/cyan]\n"
        "Groq performs the OriginChain tool call → ContextAPI optimizes the result → "
        "NGen answers." + src + "\n"
        "[dim]Type a question. Ctrl-C or 'exit' to quit.[/dim]",
        border_style="cyan"))

    try:
        with OriginChainPipeline() as pipe:
            tools = [t.get("name") for t in pipe.mcp.list_tools()]
            console.print(f"[dim]OriginChain tools: {', '.join(tools)}[/dim]\n")
            while True:
                try:
                    question = console.input("[bold cyan]you ›[/bold cyan] ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not question:
                    continue
                if question.lower() in {"exit", "quit", ":q"}:
                    break
                try:
                    t0 = time.time()
                    with console.status("[cyan]thinking…[/cyan]", spinner="dots"):
                        result = pipe.run(question)
                    render(question, result, elapsed=time.time() - t0)
                except OriginChainError as e:
                    console.print(Panel(str(e), title="[red]error[/red]", border_style="red"))
    except OriginChainError as e:
        console.print(Panel(str(e), title="[red]startup error[/red]", border_style="red"))
    console.print("\n[dim]bye[/dim]")


if __name__ == "__main__":
    main()
