#!/usr/bin/env python3
"""
Agentic evaluation, round 2 - the failure modes round 1 could not reach.

Round 1 scored 24/24, which only means single-turn tool selection is easy for
a 27B model. This round targets where agents actually break:

  E  multi-turn chains  - use a tool result to decide the next call
  F  parallel calls     - one turn needs more than one tool
  G  underspecification - a required argument is missing; ask, do not invent
  H  distractors        - the prompt contains values that must NOT be used
  I  error recovery     - the tool fails; adapt instead of repeating

G is the one that causes real damage in production: a model that fabricates
arguments will happily email the wrong person.
"""

import argparse, json, re, time, urllib.request

T = lambda n, d, p, r: {"type": "function", "function": {
    "name": n, "description": d,
    "parameters": {"type": "object", "properties": p, "required": r}}}

TOOLS = [
    T("get_weather", "Get the current weather for a location.",
      {"location": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
      ["location"]),
    T("get_timezone", "Get the IANA timezone for a city.",
      {"city": {"type": "string"}}, ["city"]),
    T("convert_currency", "Convert an amount between two currencies.",
      {"amount": {"type": "number"}, "from_currency": {"type": "string"},
       "to_currency": {"type": "string"}}, ["amount", "from_currency", "to_currency"]),
    T("get_exchange_rate", "Get the exchange rate between two currencies.",
      {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}},
      ["from_currency", "to_currency"]),
    T("send_email", "Send an email to a specific address.",
      {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
      ["to", "subject", "body"]),
    T("lookup_contact", "Look up a person's email address by their name.",
      {"name": {"type": "string"}}, ["name"]),
    T("create_calendar_event", "Create a calendar event.",
      {"title": {"type": "string"}, "date": {"type": "string", "description": "YYYY-MM-DD"},
       "start_time": {"type": "string", "description": "HH:MM 24h"},
       "duration_minutes": {"type": "integer"}}, ["title", "date", "start_time"]),
    T("get_stock_price", "Get the latest share price for a ticker symbol.",
      {"symbol": {"type": "string"}}, ["symbol"]),
    T("run_sql", "Run a read-only SQL query against the analytics database.",
      {"query": {"type": "string"}}, ["query"]),
    T("read_file", "Read a file from the local filesystem.",
      {"path": {"type": "string"}}, ["path"]),
    T("list_directory", "List the files in a directory.",
      {"path": {"type": "string"}}, ["path"]),
]


def call(server, messages, timeout=900):
    body = json.dumps({"model": "local", "messages": messages, "tools": TOOLS,
                       "tool_choice": "auto", "temperature": 0.0,
                       "max_tokens": 600}).encode()
    req = urllib.request.Request(f"{server}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    m = d["choices"][0]["message"]
    calls = []
    for tc in (m.get("tool_calls") or []):
        a = tc["function"].get("arguments", "{}")
        try:
            a = json.loads(a) if isinstance(a, str) else (a or {})
        except json.JSONDecodeError:
            a = {"__invalid_json__": True}
        calls.append((tc["function"]["name"], a, tc.get("id", "x")))
    return m, calls, (m.get("content") or ""), time.time() - t0, d.get("usage", {})


def sub(got, want):
    """want's keys must all be present and matching in got."""
    for k, e in want.items():
        if k not in got:
            return False
        g = got[k]
        if isinstance(e, (int, float)) and not isinstance(e, bool):
            try:
                if abs(float(g) - float(e)) > 1e-6:
                    return False
            except (TypeError, ValueError):
                return False
        elif isinstance(e, str):
            if not isinstance(g, str):
                return False
            if e.lower() not in g.lower() and g.lower() not in e.lower():
                return False
        elif g != e:
            return False
    return True


def run(server):
    R = []

    def rec(cat, name, ok, note, sec, tok):
        R.append(dict(cat=cat, task=name, ok=int(ok), note=note[:80],
                      sec=round(sec, 1), tok=tok))
        print(f"{cat:3s} {name:34s} {'OK ' if ok else 'NG '} {sec:6.1f}s  {note[:70]}")

    # ---------- E: multi-turn chains ----------
    # E1: must look up the contact first, then email that address
    msgs = [{"role": "user", "content":
             "Email Dr. Tanaka to say the benchmark finished. Subject: Benchmark done."}]
    m, c, txt, s1, u1 = call(server, msgs)
    step1_ok = bool(c) and c[0][0] == "lookup_contact" and sub(c[0][1], {"name": "tanaka"})
    if c:
        msgs += [{"role": "assistant", "tool_calls": [
            {"id": c[0][2], "type": "function",
             "function": {"name": c[0][0], "arguments": json.dumps(c[0][1])}}]},
            {"role": "tool", "tool_call_id": c[0][2],
             "content": json.dumps({"email": "tanaka@lab.example.jp"})}]
        m2, c2, t2, s2, u2 = call(server, msgs)
        step2_ok = bool(c2) and c2[0][0] == "send_email" and \
            sub(c2[0][1], {"to": "tanaka@lab.example.jp"})
        note = "" if step2_ok else f"step2={c2[0][0] if c2 else 'none'} {json.dumps(c2[0][1])[:40] if c2 else ''}"
    else:
        step2_ok, s2, u2, note = False, 0, {}, "no call in step 1"
    rec("E", "E1 lookup contact -> send email", step1_ok and step2_ok,
        note if note else "", s1 + s2, u1.get("completion_tokens", 0) + u2.get("completion_tokens", 0))

    # E2: use the returned rate to decide, then convert
    msgs = [{"role": "user", "content":
             "If the USD to JPY rate is above 140, convert 500 USD to JPY. Check the rate first."}]
    m, c, txt, s1, u1 = call(server, msgs)
    ok1 = bool(c) and c[0][0] == "get_exchange_rate"
    if c:
        msgs += [{"role": "assistant", "tool_calls": [
            {"id": c[0][2], "type": "function",
             "function": {"name": c[0][0], "arguments": json.dumps(c[0][1])}}]},
            {"role": "tool", "tool_call_id": c[0][2],
             "content": json.dumps({"rate": 152.4})}]
        m2, c2, t2, s2, u2 = call(server, msgs)
        ok2 = bool(c2) and c2[0][0] == "convert_currency" and \
            sub(c2[0][1], {"amount": 500, "from_currency": "usd", "to_currency": "jpy"})
        note = "" if ok2 else f"step2={c2[0][0] if c2 else 'none'}"
    else:
        ok2, s2, u2, note = False, 0, {}, "no call in step 1"
    rec("E", "E2 check rate -> branch -> convert", ok1 and ok2, note, s1 + s2,
        u1.get("completion_tokens", 0) + u2.get("completion_tokens", 0))

    # E3: negative branch - rate below threshold, must NOT convert
    msgs = [{"role": "user", "content":
             "Only if the EUR to USD rate is above 2.0, convert 100 EUR to USD. Check first."}]
    m, c, txt, s1, u1 = call(server, msgs)
    if c:
        msgs += [{"role": "assistant", "tool_calls": [
            {"id": c[0][2], "type": "function",
             "function": {"name": c[0][0], "arguments": json.dumps(c[0][1])}}]},
            {"role": "tool", "tool_call_id": c[0][2], "content": json.dumps({"rate": 1.08})}]
        m2, c2, t2, s2, u2 = call(server, msgs)
        ok = not c2   # must stop, not convert
        note = "" if ok else f"called {c2[0][0]} although condition was false"
    else:
        ok, s2, u2, note = False, 0, {}, "no call in step 1"
    rec("E", "E3 condition false -> do not call", ok, note, s1 + s2,
        u1.get("completion_tokens", 0) + u2.get("completion_tokens", 0))

    # ---------- F: parallel / multiple calls ----------
    m, c, txt, s, u = call(server, [{"role": "user", "content":
        "Give me the current weather in both Tokyo and London."}])
    names = [x[0] for x in c]
    locs = " ".join(json.dumps(x[1]).lower() for x in c)
    ok = names.count("get_weather") >= 2 or ("tokyo" in locs and "london" in locs)
    rec("F", "F1 weather for two cities", ok, f"calls={len(c)} {names}", s, u.get("completion_tokens", 0))

    m, c, txt, s, u = call(server, [{"role": "user", "content":
        "What time zone is Cairo in, and what is the weather there?"}])
    names = [x[0] for x in c]
    ok = "get_timezone" in names and "get_weather" in names
    rec("F", "F2 timezone + weather", ok, f"calls={len(c)} {names}", s, u.get("completion_tokens", 0))

    # ---------- G: underspecification -> must ask, not invent ----------
    for name, prompt, tool, badkeys in [
        ("G1 email with no recipient", "Send an email saying the build passed.", "send_email", ["to"]),
        ("G2 event with no date", "Put a team sync on the calendar at 10am.", "create_calendar_event", ["date"]),
        ("G3 conversion with no source currency", "Convert 300 into yen for me.", "convert_currency", ["from_currency"]),
    ]:
        m, c, txt, s, u = call(server, [{"role": "user", "content": prompt}])
        if not c:
            ok, note = True, "asked back (no tool call)"
        else:
            got = c[0][1]
            invented = [k for k in badkeys if k in got and got[k] not in (None, "", "unknown")]
            ok = not invented
            note = f"invented: {', '.join(f'{k}={got[k]}' for k in invented)}" if invented else "omitted the required argument"
        rec("G", name, ok, note, s, u.get("completion_tokens", 0))

    # ---------- H: distractors ----------
    m, c, txt, s, u = call(server, [{"role": "user", "content":
        "I paid 45 dollars for lunch yesterday and 12 for coffee. Anyway, convert 900 USD to EUR."}])
    ok = bool(c) and c[0][0] == "convert_currency" and sub(c[0][1], {"amount": 900})
    rec("H", "H1 distractor amounts", ok,
        f"amount={c[0][1].get('amount') if c else 'none'}", s, u.get("completion_tokens", 0))

    m, c, txt, s, u = call(server, [{"role": "user", "content":
        "Last year on 2025-04-01 we had an outage. Schedule a postmortem called Outage Review "
        "for 2026-11-20 at 16:00."}])
    ok = bool(c) and c[0][0] == "create_calendar_event" and sub(c[0][1], {"date": "2026-11-20"})
    rec("H", "H2 distractor past date", ok,
        f"date={c[0][1].get('date') if c else 'none'}", s, u.get("completion_tokens", 0))

    m, c, txt, s, u = call(server, [{"role": "user", "content":
        "My colleague's ticker is TSLA but I want the price of NVDA."}])
    ok = bool(c) and c[0][0] == "get_stock_price" and sub(c[0][1], {"symbol": "nvda"})
    rec("H", "H3 distractor ticker", ok,
        f"symbol={c[0][1].get('symbol') if c else 'none'}", s, u.get("completion_tokens", 0))

    # ---------- I: error recovery ----------
    msgs = [{"role": "user", "content": "Read the file /data/report.txt and tell me what is in it."}]
    m, c, txt, s1, u1 = call(server, msgs)
    if c:
        msgs += [{"role": "assistant", "tool_calls": [
            {"id": c[0][2], "type": "function",
             "function": {"name": c[0][0], "arguments": json.dumps(c[0][1])}}]},
            {"role": "tool", "tool_call_id": c[0][2],
             "content": json.dumps({"error": "ENOENT: no such file or directory"})}]
        m2, c2, t2, s2, u2 = call(server, msgs)
        # good behaviour: list the directory, or report the failure. bad: silently invent contents.
        listed = bool(c2) and c2[0][0] == "list_directory"
        reported = (not c2) and re.search(r"not (found|exist)|no such|error", t2, re.I) is not None
        ok = listed or reported
        note = "recovered via list_directory" if listed else ("reported the failure" if reported else
               f"inappropriate: {c2[0][0] if c2 else t2[:40]}")
    else:
        ok, s2, u2, note = False, 0, {}, "no call in step 1"
    rec("I", "I1 recover from missing file", ok, note, s1 + s2,
        u1.get("completion_tokens", 0) + u2.get("completion_tokens", 0))

    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:11439")
    ap.add_argument("--out", default="/home/ubuntu/Desktop/dirOllamaSetting/results27b/agentic2.csv")
    a = ap.parse_args()
    print(f"{'cat':3s} {'task':34s} {'ok':3s} {'sec':>7s}  note")
    R = run(a.server)
    print()
    names = {"E": "E multi-turn chain", "F": "F parallel calls", "G": "G ask when underspecified",
             "H": "H distractor resistance", "I": "I error recovery"}
    for c in sorted({r["cat"] for r in R}):
        rs = [r for r in R if r["cat"] == c]
        k = sum(r["ok"] for r in rs)
        print(f"{names[c]:26s} {k}/{len(rs)} ({100*k/len(rs):5.1f}%)")
    k = sum(r["ok"] for r in R)
    print(f"{'total':26s} {k}/{len(R)} ({100*k/len(R):5.1f}%)")
    print(f"\nper task mean {sum(r['sec'] for r in R)/len(R):.1f}s  "
          f"completion tokens mean {sum(r['tok'] for r in R)/len(R):.0f}")
    import csv
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(R[0].keys())); w.writeheader(); w.writerows(R)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
