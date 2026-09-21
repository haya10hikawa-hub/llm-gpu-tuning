#!/usr/bin/env python3
"""
Agentic evaluation for Qwen3.8-27B on gfx906.

Downstream accuracy benchmarks (HellaSwag etc.) say nothing about whether a
model can drive tools, which is what "agentic" use actually needs. This
measures the four things that break agents in practice:

  A  tool selection      - picks the right tool for an unambiguous request
  B  disambiguation      - picks correctly when several tools look similar
  C  argument extraction - pulls the right values out of the request
  D  restraint           - does NOT call a tool when none is needed

plus the practical cost: latency per turn and how many tokens are burned on
reasoning before the tool call appears.

Talks to llama-server's OpenAI-compatible endpoint.
"""

import argparse
import json
import re
import sys
import time
import urllib.request

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {"type": "object", "properties": {
            "location": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["location"]}}},
    {"type": "function", "function": {
        "name": "get_forecast",
        "description": "Get a multi-day weather forecast for a location.",
        "parameters": {"type": "object", "properties": {
            "location": {"type": "string"},
            "days": {"type": "integer", "description": "Number of days ahead"}},
            "required": ["location", "days"]}}},
    {"type": "function", "function": {
        "name": "search_web",
        "description": "Search the public web and return result snippets.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "search_docs",
        "description": "Search the internal company documentation, not the public web.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "send_email",
        "description": "Send an email.",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string"}, "subject": {"type": "string"},
            "body": {"type": "string"}},
            "required": ["to", "subject", "body"]}}},
    {"type": "function", "function": {
        "name": "create_calendar_event",
        "description": "Create an event on the calendar.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"}, "date": {"type": "string", "description": "YYYY-MM-DD"},
            "start_time": {"type": "string", "description": "HH:MM 24h"},
            "duration_minutes": {"type": "integer"}},
            "required": ["title", "date", "start_time"]}}},
    {"type": "function", "function": {
        "name": "convert_currency",
        "description": "Convert an amount between two currencies.",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number"},
            "from_currency": {"type": "string", "description": "ISO code e.g. USD"},
            "to_currency": {"type": "string"}},
            "required": ["amount", "from_currency", "to_currency"]}}},
    {"type": "function", "function": {
        "name": "get_stock_price",
        "description": "Get the latest share price for a ticker symbol.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string"}},
            "required": ["symbol"]}}},
    {"type": "function", "function": {
        "name": "run_sql",
        "description": "Run a read-only SQL query against the analytics database.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file from the local filesystem.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}},
            "required": ["path"]}}},
]

# expect: tool name, or None meaning "must not call any tool"
# args: subset of arguments that must match (case-insensitive for strings)
TASKS = [
    # --- A: unambiguous tool selection ---
    ("A", "What's the weather in Osaka right now?", "get_weather", {"location": "osaka"}),
    ("A", "Send an email to bob@example.com with the subject Status and the body All green.",
     "send_email", {"to": "bob@example.com", "subject": "status"}),
    ("A", "How much is Apple stock trading at?", "get_stock_price", {"symbol": "aapl"}),
    ("A", "Read the file /etc/hostname for me.", "read_file", {"path": "/etc/hostname"}),
    ("A", "Convert 250 US dollars to Japanese yen.", "convert_currency",
     {"amount": 250, "from_currency": "usd", "to_currency": "jpy"}),
    ("A", "Query the analytics database for the total number of rows in the orders table.",
     "run_sql", {}),
    ("A", "Book a meeting called Design Review on 2026-10-15 starting at 14:30.",
     "create_calendar_event", {"title": "design review", "date": "2026-10-15", "start_time": "14:30"}),
    ("A", "What will the weather in Berlin be like over the next 5 days?",
     "get_forecast", {"location": "berlin", "days": 5}),

    # --- B: disambiguation between similar tools ---
    ("B", "Look up what our internal onboarding docs say about VPN setup.",
     "search_docs", {}),
    ("B", "Search the public internet for reviews of the Radeon VII.",
     "search_web", {}),
    ("B", "I need the forecast for Kyoto for the next three days.",
     "get_forecast", {"location": "kyoto", "days": 3}),
    ("B", "Is it raining in Kyoto at the moment?", "get_weather", {"location": "kyoto"}),
    ("B", "Find the company policy document about expense reports.", "search_docs", {}),
    ("B", "What is the current temperature in Reykjavik in fahrenheit?",
     "get_weather", {"location": "reykjavik", "unit": "fahrenheit"}),

    # --- C: argument extraction precision ---
    ("C", "Convert 1499.99 euros into Swiss francs.", "convert_currency",
     {"amount": 1499.99, "from_currency": "eur", "to_currency": "chf"}),
    ("C", "Put 'Quarterly Planning' on the calendar for March 3rd 2027 at 9am for two hours.",
     "create_calendar_event",
     {"title": "quarterly planning", "date": "2027-03-03", "start_time": "09:00",
      "duration_minutes": 120}),
    ("C", "Give me the weather for Sao Paulo in celsius.", "get_weather",
     {"location": "sao paulo", "unit": "celsius"}),
    ("C", "Get me a 10 day outlook for Nairobi.", "get_forecast",
     {"location": "nairobi", "days": 10}),
    ("C", "What is the share price of ticker MSFT?", "get_stock_price", {"symbol": "msft"}),
    ("C", "Search the web for gfx906 vulkan benchmarks, I want 20 results.",
     "search_web", {"max_results": 20}),

    # --- D: restraint, no tool should be called ---
    ("D", "Explain in one sentence what a roofline model is.", None, {}),
    ("D", "What is 17 multiplied by 23?", None, {}),
    ("D", "Thanks, that was helpful.", None, {}),
    ("D", "Which is generally faster for matrix multiply, FP16 or FP32, on hardware with no matrix cores?",
     None, {}),
]


def call(server, messages, tools, timeout=600):
    body = json.dumps({
        "model": "local", "messages": messages, "tools": tools,
        "tool_choice": "auto", "temperature": 0.0, "max_tokens": 512,
    }).encode()
    req = urllib.request.Request(f"{server}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d, time.time() - t0


def norm(v):
    if isinstance(v, str):
        return v.strip().lower().replace("'", "").replace('"', "")
    return v


def args_match(got, want):
    """Every expected key must be present and equal (numbers compared loosely)."""
    missing = []
    for k, exp in want.items():
        if k not in got:
            missing.append(f"{k}:absent")
            continue
        g, e = norm(got[k]), norm(exp)
        if isinstance(e, (int, float)) and isinstance(g, (int, float, str)):
            try:
                if abs(float(g) - float(e)) > 1e-6:
                    missing.append(f"{k}:{g}!={e}")
            except (TypeError, ValueError):
                missing.append(f"{k}:{g}!={e}")
        elif isinstance(e, str) and isinstance(g, str):
            if e not in g and g not in e:
                missing.append(f"{k}:{g}!={e}")
        elif g != e:
            missing.append(f"{k}:{g}!={e}")
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:11439")
    ap.add_argument("--out", default="/home/ubuntu/Desktop/dirOllamaSetting/results27b/agentic.csv")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    rows, cat = [], {}
    for c, prompt, want_tool, want_args in TASKS:
        try:
            d, dt = call(args.server, [{"role": "user", "content": prompt}], TOOLS)
        except Exception as e:
            rows.append(dict(cat=c, prompt=prompt[:40], tool_ok=0, args_ok=0,
                             called="ERROR", detail=str(e)[:60], sec=0,
                             think_tok=0, total_tok=0))
            continue
        msg = d["choices"][0]["message"]
        usage = d.get("usage", {})
        tc = msg.get("tool_calls") or []
        called = tc[0]["function"]["name"] if tc else None
        raw = tc[0]["function"].get("arguments", "{}") if tc else "{}"
        try:
            got = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except json.JSONDecodeError:
            got = {}
            raw = "INVALID_JSON"

        tool_ok = int(called == want_tool)
        if want_tool is None:
            detail = "" if tool_ok else f"called {called}"
            args_ok = tool_ok
        else:
            bad = args_match(got, want_args) if tool_ok else ["tool-wrong"]
            args_ok = int(tool_ok and not bad)
            detail = ",".join(bad)[:70]

        content = msg.get("content") or ""
        think = len(re.findall(r"\S+", re.sub(r"(?s)^.*?<think>(.*?)</think>.*$", r"\1", content))) \
            if "<think>" in content else 0
        rows.append(dict(cat=c, prompt=prompt[:40], tool_ok=tool_ok, args_ok=args_ok,
                         called=called or "-", detail=detail, sec=round(dt, 1),
                         think_tok=think, total_tok=usage.get("completion_tokens", 0)))
        cat.setdefault(c, []).append((tool_ok, args_ok))

    print(f"{'cat':4s} {'prompt':42s} {'tool':>5s} {'args':>5s} {'called':22s} {'s':>6s} detail")
    for r in rows:
        print(f"{r['cat']:4s} {r['prompt']:42s} {r['tool_ok']:5d} {r['args_ok']:5d} "
              f"{r['called'][:22]:22s} {r['sec']:6.1f} {r['detail']}")

    print()
    for c in sorted(cat):
        t = sum(x[0] for x in cat[c]); a = sum(x[1] for x in cat[c]); n = len(cat[c])
        name = {"A": "A 単純な選択", "B": "B 曖昧な選択", "C": "C 引数抽出", "D": "D 呼ばない判断"}[c]
        print(f"{name:16s} tool {t}/{n} ({100*t/n:5.1f}%)   args {a}/{n} ({100*a/n:5.1f}%)")
    tt = sum(r["tool_ok"] for r in rows); aa = sum(r["args_ok"] for r in rows)
    n = len(rows)
    secs = [r["sec"] for r in rows if r["sec"] > 0]
    print(f"{'合計':16s} tool {tt}/{n} ({100*tt/n:5.1f}%)   args {aa}/{n} ({100*aa/n:5.1f}%)")
    if secs:
        print(f"\n1ターン平均 {sum(secs)/len(secs):.1f}s  最大 {max(secs):.1f}s  "
              f"生成トークン平均 {sum(r['total_tok'] for r in rows)/n:.0f}  "
              f"うち thinking 平均 {sum(r['think_tok'] for r in rows)/n:.0f}")

    import csv as _csv
    with open(args.out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
