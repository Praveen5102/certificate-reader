"""CPU threads actually available to this process.

In a container os.cpu_count() reports the HOST's cores, not the container's
share; running more compute threads than the quota allows makes OCR and model
inference much slower (threads wait on each other). Override with CPU_THREADS.
"""
from __future__ import annotations

import math
import os
from pathlib import Path


def _cgroup_quota() -> int | None:
    try:                                             # cgroup v2: "<quota> <period>" or "max <period>"
        q, p = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if q != "max":
            return max(1, math.ceil(int(q) / int(p)))
    except (OSError, ValueError):
        pass
    try:                                             # cgroup v1
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return max(1, math.ceil(q / p))
    except (OSError, ValueError):
        pass
    return None


def cpu_info() -> dict:
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {"os_cpu_count": os.cpu_count(), "affinity": affinity, "cgroup_quota": _cgroup_quota(),
            "threads_used": available_cpus()}


def available_cpus() -> int:
    env = os.environ.get("CPU_THREADS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    q = _cgroup_quota()
    return max(1, min(n, q) if q else n)
