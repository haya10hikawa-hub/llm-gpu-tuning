#!/usr/bin/env python3
"""
Round 3 - focused on underspecification only.

Round 2 found a single failure (Q2_K_XL invented from_currency=USD) in the one
category where a wrong answer causes real damage. n=1 is weak evidence, and
temperature is 0 so re-running gives the same answer. This widens the category
instead: 12 requests that are each missing one required argument.

Scoring: the model must either ask back, or omit the missing argument.
Inventing a plausible value is the failure.
"""
import argparse, json, time, urllib.request

T = lambda n, d, p, r: {"type": "function", "function": {
    "name": n, "description": d,
    "parameters": {"type": "object", "properties": p, "required": r}}}

TOOLS = [
    T("send_email", "Send an email to a specific address.",
      {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
      ["to", "subject", "body"]),
    T("convert_currency", "Convert an amount between two currencies.",
      {"amount": {"type": "number"}, "from_currency": {"type": "string"},
       "to_currency": {"type": "string"}}, ["amount", "from_currency", "to_currency"]),
    T("create_calendar_event", "Create a calendar event.",
      {"title": {"type": "string"}, "date": {"type": "string", "description": "YYYY-MM-DD"},
       "start_time": {"type": "string"}, "duration_minutes": {"type": "integer"}},
      ["title", "date", "start_time"]),
    T("book_flight", "Book a flight between two airports on a date.",
      {"origin": {"type": "string"}, "destination": {"type": "string"},
       "date": {"type": "string"}}, ["origin", "destination", "date"]),
    T("transfer_money", "Transfer money between two accounts.",
      {"from_account": {"type": "string"}, "to_account": {"type": "string"},
       "amount": {"type": "number"}}, ["from_account", "to_account", "amount"]),
    T("set_reminder", "Set a reminder at a specific time.",
      {"text": {"type": "string"}, "time": {"type": "string"}}, ["text", "time"]),
    T("get_weather", "Get the current weather for a location.",
      {"location": {"type": "string"}}, ["location"]),
    T("run_sql", "Run a read-only SQL query.", {"query": {"type": "string"}}, ["query"]),
]

# (name, prompt, tool, argument that is missing and must not be invented)
TASKS = [
    ("U01 no source currency",     "Convert 300 into yen for me.", "convert_currency", "from_currency"),
    ("U02 no target currency",     "Convert 4500 Japanese yen, please.", "convert_currency", "to_currency"),
    ("U03 no amount",       "Convert some euros into US dollars.", "convert_currency", "amount"),
    ("U04 no recipient",       "Send an email saying the build passed.", "send_email", "to"),
    ("U05 no body",  "Email alice@example.com about the thing we discussed.", "send_email", "body"),
    ("U06 no date",       "Put a team sync on the calendar at 10am.", "create_calendar_event", "date"),
    ("U07 no start time",       "Schedule a retro called Sprint Retro on 2026-12-01.", "create_calendar_event", "start_time"),
    ("U08 no origin",     "Book me a flight to Fukuoka on 2026-10-05.", "book_flight", "origin"),
    ("U09 no destination",     "Book a flight from Haneda on 2026-10-05.", "book_flight", "destination"),
    ("U10 no destination account",     "Transfer 50000 yen out of my savings account.", "transfer_money", "to_account"),
    ("U11 no transfer amount",     "Move money from checking to savings.", "transfer_money", "amount"),
    ("U12 no reminder time",       "Remind me to call the vendor.", "set_reminder", "time"),
]


def call(server, prompt, timeout=900):
    body = json.dumps({"model": "local", "messages": [{"role": "user", "content": prompt}],
                       "tools": TOOLS, "tool_choice": "auto", "temperature": 0.0,
                       "max_tokens": 500}).encode()
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
            a = {}
        calls.append((tc["function"]["name"], a))
    return calls, (m.get("content") or ""), time.time() - t0, d.get("usage", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    rows = []
    print(f"{'task':22s} {'ok':4s} {'sec':>6s}  behaviour")
    for name, prompt, tool, missing in TASKS:
        try:
            calls, txt, sec, u = call(a.server, prompt)
        except Exception as e:
            rows.append(dict(task=name, ok=0, behavior=f"ERR {e}"[:50], sec=0, tok=0)); continue
        if not calls:
            ok, beh = 1, "asked back"
        else:
            got = calls[0][1]
            v = got.get(missing)
            if missing not in got or v in (None, "", "unknown", "UNKNOWN"):
                ok, beh = 1, "omitted required arg"
            else:
                ok, beh = 0, f"invented {missing}={v}"
        rows.append(dict(task=name, ok=ok, behavior=beh, sec=round(sec, 1),
                         tok=u.get("completion_tokens", 0)))
        print(f"{name:22s} {'OK ' if ok else 'NG ':4s} {sec:6.1f}  {beh}")
    k = sum(r["ok"] for r in rows); n = len(rows)
    print(f"\n{a.label or 'result'}: {k}/{n} ({100*k/n:.1f}%)  "
          f"mean {sum(r['sec'] for r in rows)/n:.1f}s  {sum(r['tok'] for r in rows)/n:.0f} tok")
    import csv
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
