"""
DeepSeek-V4 (flash) HiCache + NUMA node binding end-to-end test.

What this test verifies
-----------------------
1. DeepSeek-V4-flash runs with `--enable-hierarchical-cache` (L2 host pool:
   c4 / c128 / indexer sub pools mirrored on host memory, SWA recompute).
2. `--hicache-numa-node N` pins the whole host KV pool to NUMA node N, so the
   KV cache host memory lands on the specified NUMA node (e.g. a CXL memory
   expander exposed as NUMA node). Verified by:
   a. /sys/devices/system/node/nodeN/meminfo MemUsed delta >= threshold.
   b. /proc/<pid>/numa_maps of the server process tree: large anonymous
      mappings are reported on the target node (informational detail).
3. Prefix cache still hits AFTER device-side eviction (the hit must come
   from the host pool; write_through policy + eviction pressure), observed
   via `usage.prompt_tokens_details.cached_tokens` from `--enable-cache-report`.

Usage
-----
# unit mode: verify only the NUMA allocator (no GPU / no model needed)
python3 test_deepseek_v4_hicache_numa.py --mode unit --numa-node 4 --unit-alloc-gb 4

# e2e mode: launch a DeepSeek-V4-flash server and run the full check
python3 test_deepseek_v4_hicache_numa.py \
    --model /path/to/DeepSeek-V4-flash \
    --numa-node 4 \
    --hicache-size 8 \
    --port 30000

# e2e with deterministic eviction pressure (recommended):
python3 test_deepseek_v4_hicache_numa.py \
    --model /path/to/DeepSeek-V4-flash \
    --numa-node 4 \
    --hicache-size 8 \
    --server-max-total-tokens 30000 \
    --port 30000

Notes
-----
- Requires Linux with sysfs NUMA info and libnuma (numactl package).
- The V4 host pool requires page_size 256, layer_first layout and the
  "kernel" IO backend; all of these are the defaults when HiCache is enabled
  for DeepseekV4ForCausalLM, so no extra flags are needed.
- Exit code 0 = all checks passed, 1 = failure.
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field

import requests

NODE_SYSFS = "/sys/devices/system/node"
KILOBYTE = 1024
# Only mappings at least this large are considered "KV pool candidates" when
# parsing /proc/<pid>/numa_maps (filters out small python/torch allocations).
NUMAMAP_MIN_MAPPING_BYTES = 64 * 1024 * 1024


# ---------------------------------------------------------------------------
# NUMA observation helpers
# ---------------------------------------------------------------------------


def list_numa_nodes() -> list[int]:
    nodes = []
    for entry in os.listdir(NODE_SYSFS):
        m = re.fullmatch(r"node(\d+)", entry)
        if m and os.path.isfile(os.path.join(NODE_SYSFS, entry, "meminfo")):
            nodes.append(int(m.group(1)))
    return sorted(nodes)


def read_node_memused() -> dict[int, int]:
    """Return {node_id: MemUsed_bytes} parsed from sysfs meminfo."""
    result = {}
    for node in list_numa_nodes():
        path = os.path.join(NODE_SYSFS, f"node{node}", "meminfo")
        with open(path) as f:
            for line in f:
                parts = line.split()
                # e.g. "Node 4  MemUsed:       123456 kB"
                if len(parts) >= 4 and parts[2] == "MemUsed:":
                    result[node] = int(parts[3]) * KILOBYTE
                    break
    return result


def parse_numa_maps(pid: int) -> list[dict]:
    """Parse /proc/<pid>/numa_maps into a list of per-mapping dicts.

    numa_maps is one line per VMA, e.g.:
        7f4a1c000000 bind:0 anon=1024 dirty=1024 N0=1024 kernelpagesize_kB=4
    A mapping's resident size is derived from the per-node page counts
    (present pages) times kernelpagesize_kB.
    """
    mappings: list[dict] = []
    current: dict | None = None
    try:
        with open(f"/proc/{pid}/numa_maps") as f:
            text = f.read()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return mappings

    def _apply_tokens(mapping: dict, tokens: list[str]) -> None:
        for tok in tokens:
            if tok.startswith("file="):
                mapping["file"] = True
            elif tok.startswith("anon="):
                mapping["anon"] = True
            else:
                nm = re.fullmatch(r"N(\d+)=(\d+)", tok)
                if nm:
                    mapping["node_pages"][int(nm.group(1))] = int(nm.group(2))
                    continue
                pm = re.fullmatch(r"kernelpagesize_kB=(\d+)", tok)
                if pm:
                    mapping["page_size"] = int(pm.group(1)) * KILOBYTE
        mapping["size_bytes"] = (
            sum(mapping["node_pages"].values()) * mapping["page_size"]
        )

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^[0-9a-f]+\s", line):
            if current is not None:
                mappings.append(current)
            current = {
                "size_bytes": 0,
                "anon": False,
                "file": False,
                "node_pages": {},
                "page_size": 4096,
            }
            _apply_tokens(current, line.split()[1:])
        elif current is not None:
            # Defensive: continuation line for the current mapping.
            _apply_tokens(current, line.split())
    if current is not None:
        mappings.append(current)
    return mappings


def process_tree_pids(root_pid: int) -> list[int]:
    try:
        import psutil
    except ImportError:
        return [root_pid]
    try:
        parent = psutil.Process(root_pid)
        return [p.pid for p in [parent] + parent.children(recursive=True)]
    except psutil.Error:
        return [root_pid]


def summarize_large_anon_maps(
    root_pid: int, min_bytes: int = NUMAMAP_MIN_MAPPING_BYTES
) -> dict[int, int]:
    """Sum pages of large anonymous mappings per NUMA node across a process tree."""
    per_node_bytes: dict[int, int] = {}
    for pid in process_tree_pids(root_pid):
        for mapping in parse_numa_maps(pid):
            if not mapping["anon"] or mapping["file"]:
                continue
            if mapping["size_bytes"] < min_bytes:
                continue
            for node, pages in mapping["node_pages"].items():
                per_node_bytes[node] = (
                    per_node_bytes.get(node, 0) + pages * mapping["page_size"]
                )
    return per_node_bytes


# ---------------------------------------------------------------------------
# Test bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class Report:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.results.append(CheckResult(name, passed, detail))
        tag = "PASS" if passed else "FAIL"
        print(f"[{tag}] {name}" + (f" -- {detail}" if detail else ""))

    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def summary(self) -> str:
        n_pass = sum(1 for r in self.results if r.passed)
        n_total = len(self.results)
        lines = [f"===== SUMMARY: {n_pass}/{n_total} checks passed ====="]
        for r in self.results:
            lines.append(f"  [{'PASS' if r.passed else 'FAIL'}] {r.name}: {r.detail}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Unit mode: verify the NUMA allocator itself (no model / GPU needed)
# ---------------------------------------------------------------------------


def run_unit_mode(args: argparse.Namespace, report: Report) -> None:
    import torch

    from sglang.srt.mem_cache.memory_pool_host import NumaHostTensorAllocator

    nodes = list_numa_nodes()
    if args.numa_node not in nodes:
        report.add(
            "numa-node-exists",
            False,
            f"node {args.numa_node} not found, available={nodes}",
        )
        return
    report.add("numa-node-exists", True, f"available nodes={nodes}")

    alloc_bytes = int(args.unit_alloc_gb * 1024**3)
    before = read_node_memused()
    allocator = NumaHostTensorAllocator(args.numa_node)
    tensor = allocator.allocate((alloc_bytes,), torch.uint8, "cpu")
    after = read_node_memused()

    target_delta = after.get(args.numa_node, 0) - before.get(args.numa_node, 0)
    other_delta = sum(
        after.get(n, 0) - before.get(n, 0) for n in nodes if n != args.numa_node
    )
    report.add(
        "allocator-lands-on-target-node",
        target_delta >= 0.8 * alloc_bytes,
        f"alloc={alloc_bytes / 2**30:.1f}GiB, "
        f"target node {args.numa_node} delta={target_delta / 2**30:.2f}GiB, "
        f"other nodes delta={other_delta / 2**30:.2f}GiB",
    )
    report.add(
        "allocator-minimal-spill",
        other_delta <= 0.2 * alloc_bytes,
        f"other nodes delta={other_delta / 2**30:.2f}GiB "
        f"(limit={0.2 * alloc_bytes / 2**30:.2f}GiB)",
    )

    # Release and verify the memory is returned.
    del tensor
    del allocator
    gc.collect()
    time.sleep(2)
    released = read_node_memused()
    released_delta = released.get(args.numa_node, 0) - before.get(
        args.numa_node, 0
    )
    report.add(
        "allocator-frees-memory",
        released_delta <= 0.3 * alloc_bytes,
        f"node {args.numa_node} still holding "
        f"{released_delta / 2**30:.2f}GiB after free",
    )


# ---------------------------------------------------------------------------
# E2E mode: launch server, verify NUMA placement + HiCache host hits
# ---------------------------------------------------------------------------


def build_server_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--trust-remote-code",
        # HiCache (L2 host pool) + NUMA binding.
        "--enable-hierarchical-cache",
        "--hicache-numa-node",
        str(args.numa_node),
        "--hicache-size",
        str(int(args.hicache_size)),
        # write_through: KV reaches the host pool as soon as the radix node
        # is finalized, no eviction needed for the host copy to exist.
        "--hicache-write-policy",
        "write_through",
        # Exposes usage.cached_tokens on the OpenAI endpoints.
        "--enable-cache-report",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
    ]
    if args.server_max_total_tokens:
        cmd += ["--max-total-tokens", str(args.server_max_total_tokens)]
    if args.server_extra_args:
        cmd += args.server_extra_args
    return cmd


def wait_for_server(
    base_url: str, process: subprocess.Popen, timeout_s: int
) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process.poll() is not None:
            print(f"Server process exited early with code {process.returncode}")
            return False
        try:
            if requests.get(f"{base_url}/health", timeout=5).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(5)
    return False


def get_cached_tokens(
    base_url: str, prompt: str, max_tokens: int = 8
) -> tuple[int, str]:
    """Send a completion request, return (cached_tokens, generated_text)."""
    resp = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": "default",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage", {})
    # With --enable-cache-report, OpenAI completions expose the hit length as
    # usage.prompt_tokens_details.cached_tokens.
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens", 0)
    text = data["choices"][0]["text"] if data.get("choices") else ""
    return cached, text


def get_token_capacity(base_url: str) -> int | None:
    try:
        info = requests.get(f"{base_url}/server_info", timeout=30).json()
        states = info.get("internal_states") or []
        for state in states:
            cap = (state.get("memory_usage") or {}).get("token_capacity")
            if cap:
                return int(cap)
    except (requests.RequestException, ValueError, KeyError):
        pass
    return None


def make_prompt(tag: str, approx_tokens: int) -> str:
    """Deterministic pseudo-random prompt of ~approx_tokens tokens."""
    unit = (
        f"Report {tag}: the quick brown fox {tag} jumps over the lazy dog "
        f"while {tag} engineers measure prefix cache behavior. "
    )
    # ~1 token per 4 chars for english text
    return unit * max(1, approx_tokens * 4 // len(unit))


def run_e2e_mode(args: argparse.Namespace, report: Report) -> None:
    base_url = f"http://{args.host}:{args.port}"

    # -- preflight ---------------------------------------------------------
    nodes = list_numa_nodes()
    if args.numa_node not in nodes:
        report.add(
            "numa-node-exists",
            False,
            f"node {args.numa_node} not found, available={nodes}",
        )
        return
    report.add("numa-node-exists", True, f"available nodes={nodes}")

    memused_before = read_node_memused()
    pool_bytes = int(args.hicache_size * 1e9)

    # -- launch ------------------------------------------------------------
    cmd = build_server_command(args)
    print("Launching server:\n  " + " ".join(cmd) + "\n")
    log_file = open(args.server_log, "w")
    process = subprocess.Popen(
        cmd, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
    )

    try:
        if not wait_for_server(base_url, process, args.launch_timeout):
            report.add(
                "server-launch", False, f"see {args.server_log} for details"
            )
            return
        report.add(
            "server-launch",
            True,
            f"pid={process.pid}, log={args.server_log}",
        )

        # -- NUMA placement verification ----------------------------------
        time.sleep(10)  # let startup allocations settle
        memused_after = read_node_memused()
        deltas = {
            n: memused_after.get(n, 0) - memused_before.get(n, 0) for n in nodes
        }
        target_delta = deltas.get(args.numa_node, 0)
        others = {n: d for n, d in deltas.items() if n != args.numa_node}
        max_other_node, max_other_delta = (
            max(others.items(), key=lambda kv: kv[1]) if others else (None, 0)
        )

        print("\nPer-node MemUsed deltas since before server launch:")
        for n in nodes:
            marker = "  <-- hicache-numa-node" if n == args.numa_node else ""
            print(f"  node {n}: {deltas[n] / 2**30:+8.2f} GiB{marker}")
        print()

        report.add(
            "host-pool-on-target-node",
            target_delta >= args.numa_threshold_frac * pool_bytes,
            f"node {args.numa_node} delta={target_delta / 2**30:.2f}GiB, "
            f"expected >= {args.numa_threshold_frac * args.hicache_size:.1f}GiB "
            f"({args.numa_threshold_frac:.0%} of --hicache-size)",
        )
        report.add(
            "host-pool-not-dominated-elsewhere",
            target_delta >= max_other_delta,
            f"largest other node {max_other_node}: "
            f"{max_other_delta / 2**30:.2f}GiB vs target "
            f"{target_delta / 2**30:.2f}GiB",
        )

        # Precise cross-check via numa_maps (informational).
        per_node = summarize_large_anon_maps(process.pid)
        detail = ", ".join(
            f"node {n}: {b / 2**30:.2f}GiB" for n, b in sorted(per_node.items())
        )
        report.add(
            "numa-maps-large-anon-detail",
            True,
            detail or "no large anon mappings found (informational)",
        )

        # -- HiCache functional tests -------------------------------------
        capacity = get_token_capacity(base_url)
        print(f"Device KV token capacity: {capacity}")

        prompt_a = make_prompt("alpha", args.prompt_tokens)
        prompt_a_suffix = "\n\nSummarize the report above in one word:"

        # T1: first request, nothing cached.
        cached1, text1 = get_cached_tokens(base_url, prompt_a + prompt_a_suffix)
        report.add(
            "t1-cold-request",
            cached1 == 0 and len(text1.strip()) > 0,
            f"cached_tokens={cached1}, generated={text1.strip()[:20]!r}",
        )

        # T2: immediate repeat -> device prefix hit (sanity for cache report).
        cached2, _ = get_cached_tokens(base_url, prompt_a + prompt_a_suffix)
        report.add(
            "t2-warm-device-hit",
            cached2 > 0,
            f"cached_tokens={cached2} of ~{args.prompt_tokens} prompt tokens",
        )

        # T3: eviction pressure with unique prompts, total tokens sized to
        # overflow the device pool so prompt A gets evicted to host.
        evict_target = args.evict_total_tokens or (
            (capacity * 2) if capacity else 150_000
        )
        n_evict = max(4, evict_target // args.prompt_tokens)
        print(
            f"\nApplying eviction pressure: {n_evict} unique prompts x "
            f"~{args.prompt_tokens} tokens (~{n_evict * args.prompt_tokens} tokens, "
            f"device capacity={capacity})"
        )
        for i in range(n_evict):
            get_cached_tokens(
                base_url, make_prompt(f"evict-{i:05d}", args.prompt_tokens)
            )
            if (i + 1) % 10 == 0:
                print(f"  eviction progress {i + 1}/{n_evict}")

        # T4: prompt A again. The device copy has been evicted, so any prefix
        # hit must come from the host pool (NUMA node bound above).
        cached4, text4 = get_cached_tokens(base_url, prompt_a + prompt_a_suffix)
        # SWA tail truncation may reduce the matched length, so only require a
        # page-aligned partial hit.
        report.add(
            "t4-host-hit-after-eviction",
            cached4 > 0,
            f"cached_tokens={cached4} (host hit after device eviction)",
        )
        report.add(
            "t4-still-generates",
            len(text4.strip()) > 0,
            f"generated={text4.strip()[:20]!r}",
        )

        # T5: server still healthy after the host load path was exercised.
        try:
            healthy = (
                requests.get(f"{base_url}/health", timeout=10).status_code == 200
            )
        except requests.RequestException:
            healthy = False
        report.add("t5-server-healthy", healthy)

    finally:
        # -- cleanup ---------------------------------------------------------
        try:
            from sglang.srt.utils import kill_process_tree

            kill_process_tree(process.pid)
        except Exception:
            process.kill()
        log_file.close()

        # Verify the NUMA memory was released with the server.
        time.sleep(10)
        memused_final = read_node_memused()
        final_delta = (
            memused_final.get(args.numa_node, 0)
            - memused_before.get(args.numa_node, 0)
        )
        report.add(
            "host-pool-released-on-exit",
            final_delta <= 0.4 * pool_bytes,
            f"node {args.numa_node} still holding "
            f"{final_delta / 2**30:.2f}GiB after server exit",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepSeek-V4 HiCache + NUMA binding test"
    )
    parser.add_argument(
        "--mode",
        choices=["unit", "e2e"],
        default="e2e",
        help="unit: NUMA allocator only; e2e: full server test",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "DSV4_MODEL", "/models/DeepSeek-V4-flash"
        ),
        help="DeepSeek-V4-flash model path (e2e mode)",
    )
    parser.add_argument(
        "--numa-node",
        type=int,
        default=int(os.environ.get("HICACHE_NUMA_NODE", "4")),
        help="Target NUMA node for the host KV pool (CXL node)",
    )
    parser.add_argument(
        "--hicache-size",
        type=float,
        default=8,
        help="Host KV pool size in GB passed to --hicache-size (e2e)",
    )
    parser.add_argument(
        "--numa-threshold-frac",
        type=float,
        default=0.4,
        help="Min fraction of --hicache-size that must land on the target node",
    )
    parser.add_argument(
        "--unit-alloc-gb", type=float, default=4, help="Allocator size (unit mode)"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument(
        "--mem-fraction-static", type=float, default=0.85, help="Server arg"
    )
    parser.add_argument(
        "--server-max-total-tokens",
        type=int,
        default=0,
        help="Optional --max-total-tokens for the server; a small value makes "
        "the eviction pressure deterministic",
    )
    parser.add_argument(
        "--server-extra-args",
        nargs="*",
        default=None,
        help="Extra args appended to the server launch command",
    )
    parser.add_argument(
        "--server-log", default="dsv4_hicache_numa_server.log"
    )
    parser.add_argument(
        "--launch-timeout", type=int, default=3600, help="Server launch timeout (s)"
    )
    parser.add_argument(
        "--prompt-tokens", type=int, default=2000, help="Approx tokens per test prompt"
    )
    parser.add_argument(
        "--evict-total-tokens",
        type=int,
        default=0,
        help="Total unique tokens used for eviction pressure (0 = 2x capacity)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = Report()

    if not os.path.isdir(NODE_SYSFS):
        print(f"ERROR: {NODE_SYSFS} not found; this test must run on Linux.")
        return 1

    if args.mode == "unit":
        run_unit_mode(args, report)
    else:
        run_e2e_mode(args, report)

    print("\n" + report.summary())
    return 0 if report.all_passed() else 1


if __name__ == "__main__":
    sys.exit(main())
