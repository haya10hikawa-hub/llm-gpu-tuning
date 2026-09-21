"""Local backend slot scheduler (plan section 4.3).

Attaches to an already-running llama-server. It never starts, stops, restarts or
model-swaps one: repeated process start/stop measured up to -40% decode on this
GPU, and a swap costs a 40-75 s reload. The pool therefore has no code path that
can evict the resident model.

Concurrency is bounded by the server's own slot count, read from GET /props --
never hardcoded, because the ceiling is a launch-flag property.
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request

# A 27B on this GPU decodes ~25 t/s. A CPU fallback after a GPU reset drops it to
# ~0.85 t/s. Anything under this floor is not a slow GPU, it is not the GPU.
DEGRADED_TPS = 5.0
MIN_TOKENS_FOR_HEALTH = 16   # too few tokens to judge a rate from


class Degraded(RuntimeError):
    """Local backend is not serving from the GPU. Never fall back silently."""


class LocalPool:
    def __init__(self, base_url, api_key=None, timeout=900):
        self.base = base_url.rstrip("/")
        self.key = api_key or os.environ.get("LLAMA_API_KEY")
        self.timeout = timeout
        self.degraded_reason = None
        self._cv = threading.Condition()
        self._waiting = []          # (priority, seq, event) heap-ish, small N
        self._seq = 0
        props = self._get("/props")
        self.slots = int(props.get("total_slots") or 0)
        gen = props.get("default_generation_settings") or {}
        self.ctx_per_slot = int(gen.get("n_ctx") or 0)
        if self.slots < 1 or self.ctx_per_slot < 1:
            raise RuntimeError(f"refusing to attach: slots={self.slots} ctx={self.ctx_per_slot}")
        self._free = self.slots

    # --- transport -------------------------------------------------------
    def _req(self, path, payload=None):
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = "Bearer " + self.key
        data = json.dumps(payload).encode() if payload is not None else None
        r = urllib.request.Request(self.base + path, data=data, headers=h)
        with urllib.request.urlopen(r, timeout=self.timeout) as resp:
            if resp.status // 100 != 2:            # a 503 is not an OK
                raise RuntimeError(f"{path} -> HTTP {resp.status}")
            return json.loads(resp.read())

    def _get(self, path):
        return self._req(path)

    def slots_busy(self):
        try:
            return sum(1 for s in self._get("/slots") if s.get("is_processing"))
        except Exception:
            return None

    # --- admission -------------------------------------------------------
    def acquire(self, priority=0):
        """priority 1 = an ancestor is blocked on this task; 0 = ordinary. FIFO ties."""
        with self._cv:
            self._seq += 1
            me = (-priority, self._seq)
            self._waiting.append(me)
            while True:
                self._waiting.sort()
                if self._free > 0 and self._waiting[0] == me:
                    self._waiting.remove(me)
                    self._free -= 1
                    return
                self._cv.wait()

    def release(self):
        with self._cv:
            self._free += 1
            self._cv.notify_all()

    # --- request ---------------------------------------------------------
    def complete(self, prompt, n_predict=128, priority=0, **kw):
        if self.degraded_reason:
            raise Degraded(self.degraded_reason)
        self.acquire(priority)
        t0 = time.perf_counter()
        try:
            body = {"prompt": prompt, "n_predict": n_predict,
                    "temperature": 0, "cache_prompt": False}
            body.update(kw)
            d = self._req("/completion", body)
        finally:
            self.release()
        wall = time.perf_counter() - t0
        t = d.get("timings") or {}
        n = int(t.get("predicted_n") or 0)
        if n <= 0:                                   # healthy prefill, zero output
            raise RuntimeError("completed with 0 generated tokens")
        tps = float(t.get("predicted_per_second") or 0.0)
        if n >= MIN_TOKENS_FOR_HEALTH and 0 < tps < DEGRADED_TPS:
            self.degraded_reason = (f"decode {tps:.2f} t/s < {DEGRADED_TPS} floor "
                                    f"- suspect CPU fallback (check DRM ACL on /dev/dri)")
            raise Degraded(self.degraded_reason)
        return {"content": d.get("content", ""), "n": n, "tps": tps, "wall": wall}
