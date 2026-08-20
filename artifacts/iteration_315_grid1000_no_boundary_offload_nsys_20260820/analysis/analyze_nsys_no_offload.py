#!/usr/bin/env python3
"""Analyze a Grid1000 Nsight export without collapsing repeated NVTX ranges.

The analyzer is intentionally conservative: kernel names that Nsight cannot
resolve remain UNKNOWN and are never counted as compute or NCCL. Metrics are
computed per node/device/phase first; a node-level aggregate is only a
distribution over devices, not a union of overlapping GPU timelines.
"""

import csv
import hashlib
import json
import pathlib
import sqlite3
from collections import defaultdict


RUN = pathlib.Path(__file__).resolve().parents[1]
RAW = RUN / "raw"
MACRO_PHASES = ("h3/student_step", "h3/critic_phase")
FALLBACK_PHASE_PREFIXES = ("h3/critic_prepare_5xG", "h3/critic_update_F")


def union(intervals):
    ordered = sorted((int(a), int(b)) for a, b in intervals if b > a)
    merged = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return sum(end - start for start, end in merged), merged


def subtract(intervals, covered):
    _, covered = union(covered)
    remainder = []
    for start, end in intervals:
        cursor = start
        for cover_start, cover_end in covered:
            if cover_end <= cursor:
                continue
            if cover_start >= end:
                break
            if cover_start > cursor:
                remainder.append((cursor, min(cover_start, end)))
            cursor = max(cursor, cover_end)
            if cursor >= end:
                break
        if cursor < end:
            remainder.append((cursor, end))
    return remainder


def sha256(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def load_sqlite(path):
    connection = sqlite3.connect(path)
    strings = dict(connection.execute("SELECT id, value FROM StringIds"))
    phase_ranges = defaultdict(list)
    for start, end, text in connection.execute(
        "SELECT start, end, text FROM NVTX_EVENTS "
        "WHERE text IS NOT NULL AND end IS NOT NULL ORDER BY start"
    ):
        if text and str(text).startswith("h3/"):
            phase_ranges[str(text)].append((int(start), int(end)))

    kernels = []
    for row in connection.execute(
        "SELECT start, end, demangledName, shortName, mangledName, "
        "deviceId, streamId, globalPid, correlationId "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        start, end, demangled, short, mangled, device, stream, gpid, corr = row
        name = strings.get(demangled) or strings.get(short) or strings.get(mangled)
        kernels.append(
            {
                "start": int(start), "end": int(end),
                "name": name or "UNRESOLVED_KERNEL",
                "device": int(device), "stream": int(stream),
                "gpid": gpid, "corr": corr,
            }
        )

    memcpys = []
    for row in connection.execute(
        "SELECT start, end, bytes, copyKind, srcKind, dstKind, deviceId, "
        "contextId, streamId, globalPid, correlationId "
        "FROM CUPTI_ACTIVITY_KIND_MEMCPY"
    ):
        start, end, bytes_, copy_kind, src, dst, device, context, stream, gpid, corr = row
        memcpys.append(
            {
                "start": int(start), "end": int(end), "bytes": int(bytes_),
                "copy_kind": int(copy_kind), "src": src, "dst": dst,
                "device": int(device), "context": context, "stream": int(stream),
                "gpid": gpid, "corr": corr,
            }
        )
    connection.close()
    return dict(phase_ranges), kernels, memcpys


def interval_records(records):
    return [(record["start"], record["end"]) for record in records]


def device_phase_metrics(kernels, memcpys, start, end):
    phase_kernels = [record for record in kernels if record["start"] < end and record["end"] > start]
    phase_memcpys = [record for record in memcpys if record["start"] < end and record["end"] > start]
    per_device = {}
    devices = sorted({record["device"] for record in phase_kernels} | {record["device"] for record in phase_memcpys})
    for device in devices:
        device_kernels = [record for record in phase_kernels if record["device"] == device]
        device_memcpys = [record for record in phase_memcpys if record["device"] == device]
        nccl = [record for record in device_kernels if "nccl" in record["name"].lower()]
        known_compute = [
            record for record in device_kernels
            if record["name"] != "UNRESOLVED_KERNEL" and "nccl" not in record["name"].lower()
        ]
        unresolved = [record for record in device_kernels if record["name"] == "UNRESOLVED_KERNEL"]
        all_intervals = interval_records(device_kernels)
        nccl_intervals = interval_records(nccl)
        compute_intervals = interval_records(known_compute)
        per_device[str(device)] = {
            "phase_wall_s": (end - start) / 1e9,
            "kernel_events": len(device_kernels),
            "kernel_union_s": union(all_intervals)[0] / 1e9,
            "known_compute_union_s": union(compute_intervals)[0] / 1e9,
            "nccl_union_s": union(nccl_intervals)[0] / 1e9,
            "nccl_exposed_vs_known_compute_s": union(subtract(nccl_intervals, compute_intervals))[0] / 1e9,
            "unresolved_kernel_events": len(unresolved),
            "unresolved_kernel_union_s": union(interval_records(unresolved))[0] / 1e9,
            "h2d_bytes": sum(record["bytes"] for record in device_memcpys if record["copy_kind"] == 1),
            "h2d_events": sum(record["copy_kind"] == 1 for record in device_memcpys),
            "d2h_bytes": sum(record["bytes"] for record in device_memcpys if record["copy_kind"] == 2),
            "d2h_events": sum(record["copy_kind"] == 2 for record in device_memcpys),
        }
    return per_device


def summarize_distribution(per_device, field):
    values = sorted(info[field] for info in per_device.values())
    if not values:
        return {"count": 0}
    return {
        "count": len(values), "min": values[0],
        "median": values[len(values) // 2],
        "mean": sum(values) / len(values), "max": values[-1],
    }


rows = []
result = {
    "run_id": RUN.name,
    "profile_kind": "diagnostic_only",
    "formal_timing_used": False,
    "phase_storage": "repeated NVTX ranges preserved as lists",
    "metrics_scope": "node/device/phase; no cross-device union used for utilization",
    "nodes": {}, "world16_usable": False,
}

for sqlite_path in sorted(RAW.glob("node*/*.sqlite")):
    node = sqlite_path.parent.name
    phase_ranges, kernels, memcpys = load_sqlite(sqlite_path)
    macro_ranges = []
    for name in MACRO_PHASES:
        macro_ranges.extend((name, index, start, end) for index, (start, end) in enumerate(phase_ranges.get(name, [])))
    # Keep the single student macro range and add fallback critic ranges when
    # this producer does not emit the newer h3/critic_phase marker.  Repeated
    # h3/critic_update_F entries are intentionally retained as occurrences.
    if "h3/critic_phase" not in phase_ranges:
        for name, ranges in phase_ranges.items():
            if name.startswith(FALLBACK_PHASE_PREFIXES):
                macro_ranges.extend((name, index, start, end) for index, (start, end) in enumerate(ranges))
    workload_ranges = [(start, end) for _, _, start, end in macro_ranges]
    workload_kernels = [record for record in kernels if any(record["start"] < end and record["end"] > start for start, end in workload_ranges)]
    workload_memcpys = [record for record in memcpys if any(record["start"] < end and record["end"] > start for start, end in workload_ranges)]
    device_ids = sorted({record["device"] for record in workload_kernels})
    phase_results = {}
    for name, index, start, end in macro_ranges:
        per_device = device_phase_metrics(kernels, memcpys, start, end)
        phase_id = f"{name}#{index}"
        phase_results[phase_id] = {
            "name": name, "occurrence": index,
            "start_ns": start, "end_ns": end,
            "per_device": per_device,
            "device_distributions": {
                field: summarize_distribution(per_device, field)
                for field in ("kernel_union_s", "known_compute_union_s", "nccl_union_s", "nccl_exposed_vs_known_compute_s", "h2d_bytes", "d2h_bytes")
            },
        }
        for device, info in per_device.items():
            for metric, value in info.items():
                rows.append({"node": node, "device": device, "phase": phase_id, "metric": metric, "value": value})

    errors = []
    log_path = RUN / "logs" / f"{node}_launcher.log"
    if log_path.exists():
        log_text = log_path.read_text(errors="ignore")
        if "Errors occurred while processing the raw events" in log_text:
            errors.append("nsys_report_processing_error")
        if "ProcessEventsError" in log_text:
            errors.append("ProcessEventsError")
    result["nodes"][node] = {
        "sqlite": str(sqlite_path), "sqlite_sha256": sha256(sqlite_path),
        "phase_ranges": {name: [[start, end] for start, end in ranges] for name, ranges in phase_ranges.items()},
        "macro_phase_ranges": [
            {"name": name, "occurrence": index, "start_ns": start, "end_ns": end}
            for name, index, start, end in macro_ranges
        ],
        "phase_results": phase_results,
        "kernel_events_in_workload": len(workload_kernels),
        "kernel_devices_in_workload": device_ids,
        "kernel_device_coverage": len(device_ids) / 8,
        "unresolved_kernel_events": sum(record["name"] == "UNRESOLVED_KERNEL" for record in workload_kernels),
        "unresolved_kernel_duration_s": union(interval_records([record for record in workload_kernels if record["name"] == "UNRESOLVED_KERNEL"]))[0] / 1e9,
        "h2d_bytes_in_workload": sum(record["bytes"] for record in workload_memcpys if record["copy_kind"] == 1),
        "d2h_bytes_in_workload": sum(record["bytes"] for record in workload_memcpys if record["copy_kind"] == 2),
        "d2h_events_in_workload": sum(record["copy_kind"] == 2 for record in workload_memcpys),
        "capture_processing_errors": sorted(set(errors)),
    }

valid = True
reasons = []
for node, info in result["nodes"].items():
    names = {item["name"] for item in info["macro_phase_ranges"]}
    if not {"h3/student_step", "h3/critic_phase"}.issubset(names):
        valid = False; reasons.append(f"{node}:missing_student_or_critic_phase")
    if info["kernel_device_coverage"] < 1:
        valid = False; reasons.append(f"{node}:kernel_devices={info['kernel_devices_in_workload']}")
    if info["capture_processing_errors"]:
        valid = False; reasons.append(f"{node}:processing_error")
    for phase_id, phase in info["phase_results"].items():
        for device, metrics in phase["per_device"].items():
            if metrics["kernel_union_s"] > 0 and metrics["unresolved_kernel_union_s"] / metrics["kernel_union_s"] >= 0.01:
                valid = False; reasons.append(f"{node}:{phase_id}:device{device}:unresolved_ratio>=1%")
result["world16_usable"] = valid
result["validity_reasons"] = reasons

(RUN / "analysis").mkdir(parents=True, exist_ok=True)
(RUN / "analysis" / "nsys_no_offload_summary.json").write_text(json.dumps(result, indent=2))
with (RUN / "analysis" / "nsys_no_offload_summary.csv").open("w", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=["node", "device", "phase", "metric", "value"])
    writer.writeheader(); writer.writerows(rows)
print(json.dumps(result, indent=2))
