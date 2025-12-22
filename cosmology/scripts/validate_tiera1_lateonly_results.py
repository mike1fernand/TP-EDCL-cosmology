#!/usr/bin/env python3
"""
validate_tiera1_lateonly_results.py

Purpose
-------
Tier-A1 (late-only) *referee-safe* validator for the EDCL Phase-1 pipeline.

This script is designed to be:
  - Deterministic (no stochastic thresholds).
  - Defensive (robust chain discovery, robust parsing, strict JSON output).
  - Referee-friendly (explicit, pre-registered acceptance tests).

Inputs
------
Either:
  --workdir <DIR>   : the Tier-A1 suite work directory created by
                      cosmology/scripts/run_tiera1_lateonly_suite.py

Or:
  --bundle <ZIP>    : a bundle zip previously produced by the suite runner.
                      (The bundle is extracted to a temp directory for validation.)

Outputs (written into the workdir root)
---------------------------------------
  results_summary.json : machine-readable summary
  results_report.md    : human-readable report

Exit codes
----------
  0 : PASS
  1 : WARN (quality warnings in non-referee profiles)
  2 : FAIL
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import io
import json
import os
import re
import shutil
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception as e:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from e


# -----------------------------
# Utilities
# -----------------------------
def _utc_stamp() -> str:
    # timezone-aware (avoid deprecation warnings)
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_read_text(path: Path, max_bytes: int = 2_000_000) -> str:
    try:
        data = path.read_bytes()
    except Exception:
        return ""
    if len(data) > max_bytes:
        data = data[:max_bytes] + b"\n[TRUNCATED]\n"
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return data.decode(errors="replace")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_text(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _weighted_quantile(x: List[float], w: List[float], q: float) -> float:
    """Weighted quantile, using cumulative weights (Cobaya/GetDist-friendly)."""
    if not x:
        return float("nan")
    if q <= 0:
        return min(x)
    if q >= 1:
        return max(x)

    idx = sorted(range(len(x)), key=lambda i: x[i])
    xs = [x[i] for i in idx]
    ws = [w[i] for i in idx]

    tot = float(sum(ws))
    if not (tot > 0):
        # fallback: unweighted
        n = len(xs)
        k = int(round(q * (n - 1)))
        return xs[max(0, min(n - 1, k))]

    target = q * tot
    c = 0.0
    for i, (xi, wi) in enumerate(zip(xs, ws)):
        c_next = c + wi
        if c_next >= target:
            return xi
        c = c_next
    return xs[-1]


def _format_float(x: Optional[float], ndp: int = 6) -> str:
    if x is None:
        return "n/a"
    try:
        if x != x:  # NaN
            return "nan"
        return f"{x:.{ndp}g}"
    except Exception:
        return "n/a"


# -----------------------------
# Chain parsing
# -----------------------------
@dataclass
class ChainTable:
    columns: List[str]
    rows: List[List[float]]

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    def col(self, name: str) -> Optional[List[float]]:
        try:
            j = self.columns.index(name)
        except ValueError:
            return None
        return [r[j] for r in self.rows]

    def bestfit_index(self, chi2_col: str) -> Optional[int]:
        c = self.col(chi2_col)
        if c is None:
            return None
        best_i = None
        best = None
        for i, v in enumerate(c):
            if best is None or v < best:
                best = v
                best_i = i
        return best_i


def _open_maybe_gz(path: Path) -> io.TextIOBase:
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _load_chain_file(path: Path) -> ChainTable:
    cols: List[str] = []
    rows: List[List[float]] = []
    with _open_maybe_gz(path) as f:
        for line in f:
            if not line.strip():
                continue
            if line.startswith("#"):
                if not cols:
                    cols = line.lstrip("#").split()
                continue
            if not cols:
                continue
            parts = line.split()
            if len(parts) != len(cols):
                continue
            try:
                rows.append([float(p) for p in parts])
            except Exception:
                continue
    return ChainTable(columns=cols, rows=rows)


def _merge_chains(chain_paths: List[Path]) -> ChainTable:
    if not chain_paths:
        return ChainTable(columns=[], rows=[])
    tables = [_load_chain_file(p) for p in chain_paths]
    common = tables[0].columns[:]
    for t in tables[1:]:
        common = [c for c in common if c in t.columns]
    if not common:
        return ChainTable(columns=[], rows=[])
    merged_rows: List[List[float]] = []
    for t in tables:
        idxs = [t.columns.index(c) for c in common]
        for r in t.rows:
            merged_rows.append([r[j] for j in idxs])
    return ChainTable(columns=common, rows=merged_rows)


# -----------------------------
# Discovery helpers
# -----------------------------
def _split_output_prefix(output_value: str) -> Tuple[Path, str]:
    """
    Interpret Cobaya's `output` field.

    Typical for this repo: output = /abs/path/to/workdir/chains/<prefix>
    Cobaya then uses chain_dir=/abs/path/to/workdir/chains and prefix=<prefix>.
    """
    out = Path(output_value)
    if out.parent == Path("."):
        # bare prefix; assume workdir/chains
        return Path("chains"), out.name
    return out.parent, out.name


def _find_updated_yaml(chain_dir: Path, prefix: str) -> Optional[Path]:
    cand = chain_dir / f"{prefix}.updated.yaml"
    return cand if cand.exists() else None


def _discover_chain_files(chain_dir: Path, prefix: str) -> List[Path]:
    """
    Strictly match Cobaya chain naming to avoid prefix-overlap bugs.
    Typical pattern: <prefix>.<n>.txt or <prefix>.<n>.txt.gz
    """
    if not chain_dir.exists():
        return []
    pat = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.txt(\.gz)?$")
    pat_single = re.compile(rf"^{re.escape(prefix)}\.txt(\.gz)?$")
    out: List[Path] = []
    for p in sorted(chain_dir.iterdir()):
        if not p.is_file():
            continue
        if pat.match(p.name) or pat_single.match(p.name):
            out.append(p)
    return out


# -----------------------------
# Chi2 helpers
# -----------------------------
def _first_matching(columns: List[str], patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        try:
            rx = re.compile(pat)
        except re.error:
            continue
        for c in columns:
            if rx.match(c):
                return c
    return None


def _select_unique_chi2_components(columns: List[str]) -> List[str]:
    """
    Cobaya can output both short aliases (chi2__BAO) and fully-qualified names
    (chi2__bao.desi_dr2.desi_bao_all). Summing all chi2__ columns double-counts.
    """
    chi2_cols = [c for c in columns if c.startswith("chi2__")]

    def prefer(long_pat: str, short: str) -> Optional[str]:
        longs = [c for c in chi2_cols if re.match(long_pat, c)]
        if longs:
            return sorted(longs)[0]
        if short in chi2_cols:
            return short
        return None

    selected: List[str] = []
    for c in [
        prefer(r"^chi2__bao\.", "chi2__BAO"),
        prefer(r"^chi2__sn\.", "chi2__SN"),
        prefer(r"^chi2__H0\.", "chi2__H0"),
    ]:
        if c:
            selected.append(c)

    used = set(selected)
    for c in chi2_cols:
        if c in used:
            continue
        if c in ("chi2__BAO", "chi2__SN", "chi2__H0"):
            continue
        if c.startswith("chi2__bao.") or c.startswith("chi2__sn.") or c.startswith("chi2__H0."):
            continue
        selected.append(c)
    return selected


# -----------------------------
# Log scanning (conservative)
# -----------------------------
def _scan_logs_for_fatal_patterns(log_dir: Path, patterns: List[str]) -> List[Dict[str, Any]]:
    if not log_dir.exists():
        return []
    compiled: List[re.Pattern] = []
    for p in patterns:
        # Enforce word boundaries for 'nan' if someone leaves it unanchored in config.
        if p.lower() in ("nan", "NaN".lower()):
            p = r"\bnan\b"
        try:
            compiled.append(re.compile(p, flags=re.IGNORECASE))
        except re.error:
            continue

    hits: List[Dict[str, Any]] = []
    for lp in sorted(log_dir.glob("*.log")):
        txt = _safe_read_text(lp)
        if not txt:
            continue
        for line in txt.splitlines():
            # Allowlist: benign string from our build recipe.
            if "ERROR: Cannot uninstall" in line:
                continue
            for rx in compiled:
                if rx.search(line):
                    hits.append({"file": lp.name, "pattern": rx.pattern, "line": line.strip()[:300]})
                    break
    return hits


# -----------------------------
# Validation core
# -----------------------------
def validate(workdir: Path, config_path: Path, profile_requested: str) -> Tuple[int, Dict[str, Any], str]:
    cfg = _load_yaml(config_path)

    # Profiles
    profiles = cfg.get("profiles", {}) or {}
    if profile_requested not in profiles:
        raise ValueError(f"Unknown profile '{profile_requested}'. Available: {sorted(profiles.keys())}")

    # Resolve alias profiles (e.g., smoke -> iterate)
    profile_resolved = profile_requested
    alias_of = (profiles.get(profile_requested) or {}).get("alias_of")
    if isinstance(alias_of, str) and alias_of:
        profile_resolved = alias_of

    prof = profiles.get(profile_resolved, {}) or {}
    # Support both keys for backwards compatibility
    quality_policy = prof.get("quality_gate_policy") or prof.get("policy") or "warn"
    quality_policy = "hard" if str(quality_policy).lower().startswith("hard") else "warn"
    min_samples = int(prof.get("min_samples", 0) or 0)
    min_chains = int(prof.get("min_chains", 1) or 1)

    # Acceptance spec (Phase-1 / Tier-A1)
    acc = cfg.get("acceptance", {}) or {}
    acc_act = acc.get("activation", {}) or {}
    acc_col = acc.get("collapse", {}) or {}
    acc_ratio = acc.get("collapse_ratio", {}) or {}

    activation_run = str(acc_act.get("run", "edcl_h0"))
    activation_param = str(acc_act.get("param", "alpha_R"))
    activation_q = float(acc_act.get("q", 0.5))
    activation_min = float(acc_act.get("min", 0.03))

    collapse_run = str(acc_col.get("run", "edcl_noh0"))
    collapse_param = str(acc_col.get("param", "alpha_R"))
    collapse_q = float(acc_col.get("q", 0.95))
    collapse_pass_max = float(acc_col.get("pass_max", 0.03))
    collapse_strong_max = float(acc_col.get("strong_pass_max", 0.02))

    ratio_num_run = str(acc_ratio.get("numerator_run", collapse_run))
    ratio_num_q = float(acc_ratio.get("numerator_q", collapse_q))
    ratio_den_run = str(acc_ratio.get("denominator_run", activation_run))
    ratio_den_q = float(acc_ratio.get("denominator_q", activation_q))
    ratio_max = float(acc_ratio.get("max", 0.5))

    # Checks
    checks = cfg.get("checks", {}) or {}
    updated_yaml_required = bool(checks.get("updated_yaml_required", True))
    chi2_total_patterns = list(checks.get("chi2_total_column_patterns", [r"^chi2$"]))
    log_patterns = list(checks.get("log_fail_patterns", [])) or [
        r"Traceback \(most recent call last\)",
        r"Segmentation fault",
        r"\bFATAL\b",
        r"\bAborting\b",
        r"Class did not read input parameter",
        r"Serious error",
        r"RuntimeError:",
    ]
    required_paths = list(checks.get("required_paths", []))

    # Required paths
    missing_required: List[str] = []
    for rel in required_paths:
        if not (workdir / rel).exists():
            missing_required.append(rel)

    # Log scan
    log_hits = _scan_logs_for_fatal_patterns(workdir / "logs", log_patterns)

    runs_cfg = cfg.get("runs", {}) or {}

    # Determine which (run,param) we need quantiles for, based on acceptance rules
    needed_params_by_run: Dict[str, List[str]] = {}
    for r, p in [
        (activation_run, activation_param),
        (collapse_run, collapse_param),
        (ratio_num_run, collapse_param),
        (ratio_den_run, activation_param),
    ]:
        needed_params_by_run.setdefault(r, [])
        if p not in needed_params_by_run[r]:
            needed_params_by_run[r].append(p)

    run_results: Dict[str, Any] = {}
    quality_warnings: List[str] = []
    hard_fail_reasons: List[str] = []

    def _gate_or_warn(msg: str) -> None:
        if quality_policy == "hard":
            hard_fail_reasons.append(msg)
        else:
            quality_warnings.append(msg)

    for run_key, rcfg in runs_cfg.items():
        yaml_rel = rcfg.get("yaml")
        label = rcfg.get("label", run_key)
        track_params = rcfg.get("track_params") or needed_params_by_run.get(run_key, [])

        if not isinstance(yaml_rel, str):
            hard_fail_reasons.append(f"{run_key}: missing 'yaml' in validation_config.yaml")
            continue


        ypath = workdir / yaml_rel
        if not ypath.exists():
            # Backwards-compatible: config may store bare filenames, expected under workdir/yamls/
            ypath = workdir / "yamls" / yaml_rel
        if not ypath.exists():
            hard_fail_reasons.append(f"{run_key}: YAML not found: {yaml_rel}")
            continue

        ydata = _load_yaml(ypath)
        output_val = ydata.get("output")
        if not isinstance(output_val, str) or not output_val.strip():
            hard_fail_reasons.append(f"{run_key}: YAML missing 'output' path: {yaml_rel}")
            continue

        chain_dir_raw, prefix = _split_output_prefix(output_val.strip())

        # If output is absolute and we're validating an extracted bundle, the absolute path won't exist.
        if chain_dir_raw.is_absolute() and not chain_dir_raw.exists():
            chain_dir = workdir / "chains"
        else:
            chain_dir = chain_dir_raw if chain_dir_raw.is_absolute() else (workdir / chain_dir_raw)

        updated_yaml = _find_updated_yaml(chain_dir, prefix)
        if updated_yaml_required and updated_yaml is None:
            _gate_or_warn(f"{run_key}: missing updated YAML: {chain_dir}/{prefix}.updated.yaml")

        chain_files = _discover_chain_files(chain_dir, prefix)
        if len(chain_files) < 1:
            hard_fail_reasons.append(f"{run_key}: no chain files found for prefix '{prefix}' in {chain_dir}")
            continue

        if len(chain_files) < min_chains:
            _gate_or_warn(f"{run_key}: only {len(chain_files)} chain file(s) found (min_chains={min_chains})")

        merged = _merge_chains(chain_files)
        if merged.n_rows == 0 or not merged.columns:
            hard_fail_reasons.append(f"{run_key}: could not parse any samples from chain files")
            continue

        w = merged.col("weight") or [1.0] * merged.n_rows
        n_weighted = float(sum(w))

        if min_samples > 0 and n_weighted < min_samples:
            _gate_or_warn(f"{run_key}: low sample count (sum weights) = {n_weighted:.0f} < min_samples={min_samples}")

        # Quantiles for tracked params
        param_summary: Dict[str, Any] = {}
        for p in track_params:
            col = merged.col(p)
            if col is None:
                _gate_or_warn(f"{run_key}: missing parameter column '{p}' in chains")
                continue
            param_summary[p] = {
                "q05": _weighted_quantile(col, w, 0.05),
                "q50": _weighted_quantile(col, w, 0.50),
                "q95": _weighted_quantile(col, w, 0.95),
            }

        # Best-fit chi2
        chi2_total_col = _first_matching(merged.columns, chi2_total_patterns)
        bestfit: Dict[str, Any] = {"method": None, "chi2_total_bestfit": None, "components": {}}
        if chi2_total_col and merged.col(chi2_total_col) is not None:
            i_bf = merged.bestfit_index(chi2_total_col)
            if i_bf is not None:
                bestfit["method"] = f"min({chi2_total_col})"
                bestfit["chi2_total_bestfit"] = merged.col(chi2_total_col)[i_bf]
                for cc in _select_unique_chi2_components(merged.columns):
                    cvals = merged.col(cc)
                    if cvals is not None:
                        bestfit["components"][cc] = cvals[i_bf]
        else:
            # Fallback: sum a unique set of chi2__ columns
            comp_cols = _select_unique_chi2_components(merged.columns)
            comp_arrays = [merged.col(c) for c in comp_cols]
            if comp_cols and all(a is not None for a in comp_arrays):
                totals = [sum(a[i] for a in comp_arrays if a is not None) for i in range(merged.n_rows)]
                i_bf = min(range(len(totals)), key=lambda i: totals[i])
                bestfit["method"] = "min(sum(unique chi2__))"
                bestfit["chi2_total_bestfit"] = totals[i_bf]
                for cc, arr in zip(comp_cols, comp_arrays):
                    if arr is not None:
                        bestfit["components"][cc] = arr[i_bf]

        run_results[run_key] = {
            "label": label,
            "yaml": str(yaml_rel),
            "output": {"raw": output_val, "chain_dir": str(chain_dir), "prefix": prefix},
            "updated_yaml": str(updated_yaml) if updated_yaml else None,
            "chain_files": [str(p) for p in chain_files],
            "n_rows": merged.n_rows,
            "n_samples_weighted": n_weighted,
            "tracked_params": param_summary,
            "bestfit": bestfit,
        }

    # -----------------------------
    # Evaluate acceptance criteria
    # -----------------------------
    status = "PASS"
    reasons: List[str] = []

    if missing_required:
        status = "FAIL"
        reasons.append("Missing required paths: " + ", ".join(missing_required))
    if hard_fail_reasons:
        status = "FAIL"
        reasons.extend(hard_fail_reasons)

    # Log hits are FAIL in referee, WARN otherwise
    if log_hits:
        msg = f"Fatal-pattern matches detected in logs ({len(log_hits)} hit(s))."
        if quality_policy == "hard":
            status = "FAIL"
            reasons.append(msg)
        else:
            if status != "FAIL":
                status = "WARN"
            quality_warnings.append(msg)

    # Helper for acceptance extraction
    def _get_q(run_key: str, param: str, qkey: str) -> Optional[float]:
        try:
            return float(run_results[run_key]["tracked_params"][param][qkey])
        except Exception:
            return None

    # Activation
    act_val = _get_q(activation_run, activation_param, "q50" if activation_q == 0.5 else "q95" if activation_q == 0.95 else "q50")
    # If someone ever changes q in config, compute directly would be better; but we keep stable q=0.5.
    activation_pass = (act_val is not None) and (act_val >= activation_min)
    if not activation_pass:
        status = "FAIL"
        reasons.append(
            f"Activation failed: q{int(100*activation_q)}({activation_param}) in {activation_run} = {_format_float(act_val)} < {activation_min}"
        )

    # Collapse
    col_val = _get_q(collapse_run, collapse_param, "q95" if collapse_q == 0.95 else "q50")
    collapse_pass = (col_val is not None) and (col_val <= collapse_pass_max)
    strong_pass = (col_val is not None) and (col_val <= collapse_strong_max)
    if not collapse_pass:
        status = "FAIL"
        reasons.append(
            f"Collapse failed: q{int(100*collapse_q)}({collapse_param}) in {collapse_run} = {_format_float(col_val)} > {collapse_pass_max}"
        )

    # Collapse ratio
    num = _get_q(ratio_num_run, collapse_param, "q95" if ratio_num_q == 0.95 else "q50")
    den = _get_q(ratio_den_run, activation_param, "q50" if ratio_den_q == 0.5 else "q95")
    ratio = None
    ratio_pass = None
    if num is not None and den is not None and den > 0:
        ratio = num / den
        ratio_pass = ratio <= ratio_max
        if not ratio_pass:
            status = "FAIL"
            reasons.append(
                f"Relative collapse failed: q{int(100*ratio_num_q)}({collapse_param})/{int(100*ratio_den_q)}({activation_param}) = {_format_float(ratio)} > {ratio_max}"
            )

    # If we passed physics but have quality warnings, return WARN (iterate)
    if status == "PASS" and quality_warnings and quality_policy != "hard":
        status = "WARN"

    physics: Dict[str, Any] = {
        "activation": {
            "run": activation_run,
            "param": activation_param,
            "q": activation_q,
            "threshold_min": activation_min,
            "value": act_val,
            "pass": bool(activation_pass),
        },
        "collapse": {
            "run": collapse_run,
            "param": collapse_param,
            "q": collapse_q,
            "pass_max": collapse_pass_max,
            "strong_pass_max": collapse_strong_max,
            "value": col_val,
            "pass": bool(collapse_pass),
            "strong_pass": bool(strong_pass),
        },
        "collapse_ratio": {
            "numerator_run": ratio_num_run,
            "numerator_q": ratio_num_q,
            "denominator_run": ratio_den_run,
            "denominator_q": ratio_den_q,
            "max": ratio_max,
            "value": ratio,
            "pass": (bool(ratio_pass) if ratio_pass is not None else None),
        },
        "reasons": reasons,
    }

    summary: Dict[str, Any] = {
        "timestamp_utc": _utc_stamp(),
        "workdir": str(workdir),
        "config_path": str(config_path),
        "profile_requested": profile_requested,
        "profile_resolved": profile_resolved,
        "quality_gate_policy": quality_policy,
        "quality_warnings": quality_warnings,
        "missing_required_paths": missing_required,
        "log_fatal_hits": log_hits,
        "runs": run_results,
        "physics": physics,
        "status": status,
    }

    # Report
    md: List[str] = []
    md.append(f"# Tier-A1 validation report ({summary['timestamp_utc']} UTC)\n")
    md.append(f"**Overall status:** {status}\n")
    md.append(f"**Profile:** requested `{profile_requested}`, resolved `{profile_resolved}` (policy: `{quality_policy}`)\n")

    if missing_required:
        md.append("## Missing required paths\n")
        for p in missing_required:
            md.append(f"- {p}\n")

    if quality_warnings:
        md.append("## Quality warnings\n")
        for wmsg in quality_warnings:
            md.append(f"- {wmsg}\n")

    if log_hits:
        md.append("## Fatal-pattern log hits\n")
        for h in log_hits[:30]:
            md.append(f"- `{h['file']}` matched `{h['pattern']}`: `{h['line']}`\n")
        if len(log_hits) > 30:
            md.append(f"- ... ({len(log_hits)-30} more)\n")

    md.append("## Run summaries\n")
    for rk, rr in run_results.items():
        md.append(f"### {rk}: {rr.get('label','')}\n")
        md.append(f"- YAML: `{rr.get('yaml')}`\n")
        out = rr.get("output", {})
        md.append(f"- Output prefix: `{out.get('prefix')}`\n")
        md.append(f"- Chain dir: `{out.get('chain_dir')}`\n")
        md.append(f"- Chain files ({len(rr.get('chain_files', []))}):\n")
        for cf in rr.get("chain_files", [])[:10]:
            md.append(f"  - `{cf}`\n")
        if len(rr.get("chain_files", [])) > 10:
            md.append(f"  - ... ({len(rr.get('chain_files', [])) - 10} more)\n")
        md.append(f"- Samples: rows={rr.get('n_rows')}, sum(weights)={_format_float(rr.get('n_samples_weighted'))}\n")
        md.append(f"- Updated YAML: `{rr.get('updated_yaml')}`\n" if rr.get("updated_yaml") else "- Updated YAML: **missing**\n")

        tps = rr.get("tracked_params", {})
        if tps:
            md.append("- Tracked parameter quantiles:\n")
            for p, qs in tps.items():
                md.append(
                    f"  - `{p}`: q05={_format_float(qs.get('q05'))}, q50={_format_float(qs.get('q50'))}, q95={_format_float(qs.get('q95'))}\n"
                )

        bf = rr.get("bestfit", {})
        if bf and bf.get("chi2_total_bestfit") is not None:
            md.append(f"- Best-fit chi2_total: {_format_float(bf.get('chi2_total_bestfit'))} ({bf.get('method')})\n")
            comps = bf.get("components", {}) or {}
            if comps:
                md.append("  - Components:\n")
                for k, v in comps.items():
                    md.append(f"    - `{k}`: {_format_float(v)}\n")
        md.append("\n")

    md.append("## Acceptance criteria (Tier-A1)\n")
    md.append(
        textwrap.dedent(
            f"""
            **Activation:** q{int(100*activation_q)}({activation_param}) in `{activation_run}` >= {activation_min}

            **Collapse:** q{int(100*collapse_q)}({collapse_param}) in `{collapse_run}` <= {collapse_pass_max}
            - strong pass: <= {collapse_strong_max}

            **Relative collapse:** q{int(100*ratio_num_q)}({collapse_param}) / q{int(100*ratio_den_q)}({activation_param}) <= {ratio_max}
            """
        ).strip()
        + "\n\n"
    )

    md.append("### Results\n")
    md.append(
        f"- Activation: {_format_float(act_val)} (>= {activation_min}) -> {'PASS' if activation_pass else 'FAIL'}\n"
    )
    md.append(
        f"- Collapse: {_format_float(col_val)} (<= {collapse_pass_max}) -> {'PASS' if collapse_pass else 'FAIL'}\n"
    )
    if strong_pass:
        md.append(f"  - Strong pass achieved (<= {collapse_strong_max}).\n")
    if ratio is not None:
        md.append(f"- Relative collapse: {_format_float(ratio)} (<= {ratio_max}) -> {'PASS' if ratio_pass else 'FAIL'}\n")
    else:
        md.append("- Relative collapse: n/a (missing required quantiles)\n")

    if reasons:
        md.append("\n## Failure reasons\n")
        for r in reasons:
            md.append(f"- {r}\n")

    report = "\n".join(md)

    if status == "PASS":
        return 0, summary, report
    if status == "WARN":
        return 1, summary, report
    return 2, summary, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="", help="Suite work directory (preferred).")
    ap.add_argument("--bundle", default="", help="Bundle zip to validate (will be extracted).")
    ap.add_argument(
        "--config",
        default="cosmology/config/validation_config.yaml",
        help="Validation config yaml (repo-relative unless absolute).",
    )
    ap.add_argument(
        "--profile",
        default="iterate",
        choices=["iterate", "smoke", "referee"],
        help="Validation profile. 'smoke' is an alias of 'iterate'.",
    )
    args = ap.parse_args()

    if not args.workdir and not args.bundle:
        ap.error("Must provide either --workdir or --bundle")

    cleanup_dir: Optional[str] = None
    workdir = Path(args.workdir).resolve() if args.workdir else None
    if args.bundle:
        bundle = Path(args.bundle).resolve()
        if not bundle.exists():
            raise FileNotFoundError(bundle)
        tmp = tempfile.mkdtemp(prefix="tiera1_validate_")
        cleanup_dir = tmp
        shutil.unpack_archive(str(bundle), tmp)
        workdir = Path(tmp).resolve()

    assert workdir is not None

    # Resolve config path
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()
        if not cfg_path.exists():
            cfg_path = (workdir / args.config).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Validation config not found: {cfg_path}")

    code, summary, report = validate(workdir=workdir, config_path=cfg_path, profile_requested=args.profile)

    _write_json(workdir / "results_summary.json", summary)
    _write_text(workdir / "results_report.md", report)

    # Convenience: if validating a bundle (extracted to /tmp), also copy artifacts to /content for easy download in Colab.
    try:
        if os.path.isdir("/content"):
            shutil.copy2(str(workdir / "results_report.md"), "/content/results_report.md")
            shutil.copy2(str(workdir / "results_summary.json"), "/content/results_summary.json")
            print("[INFO] Copied validation artifacts to /content/results_report.md and /content/results_summary.json")
    except Exception as e:
        print(f"[WARN] Could not copy validation artifacts to /content: {e}")

    print(f"[INFO] Wrote: {workdir / 'results_report.md'}")
    print(f"[INFO] Wrote: {workdir / 'results_summary.json'}")

    # Note: if validating a bundle zip, we intentionally do not delete the extracted
    # directory automatically; the caller can delete it if desired.

    return code


if __name__ == "__main__":
    raise SystemExit(main())
