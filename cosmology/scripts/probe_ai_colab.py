#!/usr/bin/env python3
"""probe_ai_colab.py

Purpose
-------
Fast, defensive *availability + capability probe* for using Gemini inside Colab
as part of the Tier-A1 validator/autotuner loop.

This script is intentionally *not* the autotuner. It answers two questions:

  (Q1) "What AI backends are available in this runtime?"
       - Prefer Colab-native `google.colab.ai` (no API key).
       - Fallback to the Google Gen AI SDK (`google-genai`) using an API key
         (read from Colab Secrets via `google.colab.userdata`, or from
         environment variables GEMINI_API_KEY / GOOGLE_API_KEY).

  (Q2) "Can the selected backend reliably return machine-parseable JSON patches?"
       - For `google-genai`, we use JSON schema enforcement via
         response_mime_type='application/json' + response_json_schema.
       - For `google.colab.ai`, schema enforcement is not guaranteed, so we
         prompt for strict JSON and locally validate.

Outputs
-------
Writes a small forensic bundle under:
  cosmology/paper_artifacts/ai_probe/<timestamp>/
    - ai_probe.json  (machine-readable)
    - ai_probe.txt   (human-readable)

Usage
-----
  python cosmology/scripts/probe_ai_colab.py

In Colab, if you want the SDK fallback path, set a secret named GEMINI_API_KEY
or GOOGLE_API_KEY (recommended: GEMINI_API_KEY).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple


def _utc_stamp() -> str:
    return _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _is_colab() -> bool:
    if os.path.isdir("/content"):
        try:
            import google.colab  # noqa: F401

            return True
        except Exception:
            return False
    return False


def _safe_trunc(s: str, n: int = 4000) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else s[:n] + "\n...<truncated>..."


def _rank_model_name(name: str) -> Tuple[int, int, str]:
    """Heuristic ranking: prefer flash > pro > others, and newer versions.

    Rationale: Pro models can have stricter quota/rate limits on some keys. For
    autotuning/validation loops we prefer *availability* over maximum capability.
    You can always override via --prefer-model.
    """
    nl = name.lower()
    # Exclude obvious non-text models
    if any(tok in nl for tok in ["embedding", "imagen", "veo", "audio", "tts"]):
        return (999, 999, name)

    tier = 50
    if "flash" in nl:
        tier = 0
    elif "pro" in nl:
        tier = 10

    # Prefer higher version numbers if present
    # Extract first float-like pattern (e.g., 2.5, 2.0)
    m = re.search(r"(\d+)\.(\d+)", nl)
    ver = 0
    if m:
        ver = int(m.group(1)) * 100 + int(m.group(2))
    return (tier, -ver, name)


def _validate_patch_payload(obj: Any) -> Tuple[bool, str]:
    """Minimal strict validation for a patch-style response."""
    if not isinstance(obj, dict):
        return False, "Top-level JSON must be an object."
    if "proposed_changes" not in obj:
        return False, "Missing key: proposed_changes"
    if not isinstance(obj["proposed_changes"], list):
        return False, "proposed_changes must be a list"
    for i, ch in enumerate(obj["proposed_changes"]):
        if not isinstance(ch, dict):
            return False, f"change[{i}] must be an object"
        for k in ["op", "path", "value"]:
            if k not in ch:
                return False, f"change[{i}] missing key: {k}"
        if ch["op"] not in ["set"]:
            return False, f"change[{i}].op must be 'set'"
        if not isinstance(ch["path"], str) or not ch["path"].strip():
            return False, f"change[{i}].path must be a non-empty string"
    # rationale is optional but recommended
    return True, "ok"


def _get_colab_secret(name: str) -> Optional[str]:
    """Read from Colab Secrets without printing."""
    if not _is_colab():
        return None
    try:
        from google.colab import userdata  # type: ignore

        val = userdata.get(name)
        if val:
            return str(val)
    except Exception:
        return None
    return None




def _ensure_genai_key_in_env() -> str:
    """Best-effort: ensure an API key is available to the google-genai SDK.

    Returns a short string describing the source: 'env', 'colab.userdata', or 'missing'.
    """
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "env"
    k = _get_colab_secret("GEMINI_API_KEY") or _get_colab_secret("GOOGLE_API_KEY")
    if k:
        # Prefer GEMINI_API_KEY if present
        os.environ.setdefault("GEMINI_API_KEY", k)
        return "colab.userdata"
    return "missing"

def _probe_google_colab_ai(
    prompt: str,
    patch_prompt: str,
    prefer_model: str = "",
    max_models_log: int = 30,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "backend": "google.colab.ai",
        "available": False,
        "model_selected": None,
        "models": None,
        "text_ok": False,
        "json_patch_ok": False,
        "errors": [],
    }
    try:
        from google.colab import ai  # type: ignore

        out["available"] = True
        # Version metadata best-effort
        try:
            out["package_version"] = importlib.metadata.version("google-colab-ai")
        except Exception:
            out["package_version"] = None

        models: List[str] = []
        if hasattr(ai, "list_models"):
            try:
                models = list(ai.list_models())  # type: ignore
            except Exception as e:
                out["errors"].append(f"ai.list_models failed: {type(e).__name__}: {e}")
        else:
            out["errors"].append("ai.list_models not found (API drift?)")

        if models:
            out["models"] = models[:max_models_log]
        else:
            out["models"] = []

        chosen = None
        if prefer_model:
            chosen = prefer_model
        elif models:
            chosen = sorted(models, key=_rank_model_name)[0]

        out["model_selected"] = chosen

        # generate_text probe
        if not hasattr(ai, "generate_text"):
            out["errors"].append("ai.generate_text not found (API drift?)")
            return out

        try:
            if chosen:
                resp = ai.generate_text(prompt, model_name=chosen)  # type: ignore
            else:
                resp = ai.generate_text(prompt)  # type: ignore
            out["text_response"] = _safe_trunc(resp)
            out["text_ok"] = True
        except TypeError:
            # Some versions might use model=...
            try:
                if chosen:
                    resp = ai.generate_text(prompt, model=chosen)  # type: ignore
                else:
                    resp = ai.generate_text(prompt)  # type: ignore
                out["text_response"] = _safe_trunc(resp)
                out["text_ok"] = True
            except Exception as e:
                out["errors"].append(f"ai.generate_text failed: {type(e).__name__}: {e}")
                return out
        except Exception as e:
            out["errors"].append(f"ai.generate_text failed: {type(e).__name__}: {e}")
            return out

        # JSON patch probe (prompt-only, local parse)
        try:
            if chosen:
                patch_resp = ai.generate_text(patch_prompt, model_name=chosen)  # type: ignore
            else:
                patch_resp = ai.generate_text(patch_prompt)  # type: ignore
            out["json_patch_raw"] = _safe_trunc(patch_resp, n=8000)
            parsed = json.loads(patch_resp)
            ok, why = _validate_patch_payload(parsed)
            out["json_patch_ok"] = bool(ok)
            out["json_patch_validation"] = why
        except Exception as e:
            out["errors"].append(f"JSON patch test failed: {type(e).__name__}: {e}")

        return out
    except ImportError:
        out["available"] = False
        return out
    except Exception as e:
        out["errors"].append(f"Import/probe failed: {type(e).__name__}: {e}")
        return out


def _probe_google_genai_sdk(
    prompt: str,
    prefer_model: str = "",
    max_models_log: int = 30,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "backend": "google-genai",
        "available": False,
        "model_selected": None,
        "models": None,
        "text_ok": False,
        "json_schema_ok": False,
        "errors": [],
    }
    try:
        from google import genai  # type: ignore

        out["available"] = True
        try:
            out["package_version"] = importlib.metadata.version("google-genai")
        except Exception:
            out["package_version"] = None

        # Ensure the client can find an API key from env/Colab secrets.
        # Per SDK docs, GEMINI_API_KEY or GOOGLE_API_KEY can be used.
        out["api_key_source"] = _ensure_genai_key_in_env()

        client = genai.Client()  # type: ignore

        # Model listing
        names: List[str] = []
        try:
            for i, m in enumerate(client.models.list()):  # type: ignore
                if i >= 200:
                    break
                name = getattr(m, "name", None)
                if isinstance(name, str):
                    names.append(name)
                else:
                    names.append(str(m))
        except Exception as e:
            out["errors"].append(f"client.models.list failed: {type(e).__name__}: {e}")

        if names:
            out["models"] = names[:max_models_log]
        else:
            out["models"] = []

        chosen = None
        if prefer_model:
            chosen = prefer_model
        elif names:
            # Filter to gemini-ish names
            filtered = [n for n in names if "gemini" in n.lower() and "embedding" not in n.lower()]
            if not filtered:
                filtered = names
            chosen = sorted(filtered, key=_rank_model_name)[0]
        else:
            chosen = "gemini-2.5-flash"  # last-resort guess
        out["model_selected"] = chosen

        # Text generation probe
        try:
            resp = client.models.generate_content(model=chosen, contents=prompt)  # type: ignore
            out["text_response"] = _safe_trunc(getattr(resp, "text", str(resp)))
            out["text_ok"] = True
        except Exception as e:
            out["errors"].append(f"generate_content failed: {type(e).__name__}: {e}")
            return out

        # JSON schema enforcement probe (critical for autotuning safety)
        patch_schema = {
            "type": "object",
            "properties": {
                "proposed_changes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {"type": "string", "enum": ["set"]},
                            "path": {"type": "string"},
                            "value": {},
                        },
                        "required": ["op", "path", "value"],
                    },
                    "minItems": 1,
                },
                "rationale": {"type": "string"},
            },
            "required": ["proposed_changes", "rationale"],
        }

        patch_prompt = (
            "You are a strict JSON generator. Return ONLY valid JSON.\n"
            "Propose exactly ONE change that sets tuning.test_param to 0.123 (float).\n"
            "Use op='set' and path='tuning.test_param'."
        )

        try:
            resp2 = client.models.generate_content(
                model=chosen,
                contents=patch_prompt,
                config={
                    "temperature": 0,
                    "response_mime_type": "application/json",
                    "response_json_schema": patch_schema,
                },
            )  # type: ignore

            parsed = getattr(resp2, "parsed", None)
            if parsed is None:
                # If SDK doesn't populate .parsed, fall back to text parse
                parsed = json.loads(getattr(resp2, "text", "{}"))
            out["json_schema_raw_text"] = _safe_trunc(getattr(resp2, "text", ""), n=8000)
            ok, why = _validate_patch_payload(parsed)
            out["json_schema_ok"] = bool(ok)
            out["json_schema_validation"] = why
            out["json_schema_parsed"] = parsed
        except Exception as e:
            out["errors"].append(f"JSON schema probe failed: {type(e).__name__}: {e}")

        try:
            client.close()  # type: ignore
        except Exception:
            pass

        return out
    except ImportError:
        out["available"] = False
        return out
    except Exception as e:
        out["errors"].append(f"Import/probe failed: {type(e).__name__}: {e}")
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--prefer-model",
        default="",
        help="Optional model name override (applies to both backends).",
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: cosmology/paper_artifacts/ai_probe/<stamp>).",
    )
    ap.add_argument(
        "--max-models-log",
        type=int,
        default=30,
        help="Max number of model names to include in the report (per backend).",
    )
    args = ap.parse_args()

    stamp = _utc_stamp()
    pack_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if args.outdir:
        outdir = args.outdir
    else:
        outdir = os.path.join(pack_root, "cosmology", "paper_artifacts", "ai_probe", stamp)
    os.makedirs(outdir, exist_ok=True)

    prompt = "Reply with a single line: OK"
    patch_prompt = (
        "Return ONLY valid JSON (no markdown, no extra text).\n"
        "Schema:\n"
        "{\"proposed_changes\":[{\"op\":\"set\",\"path\":\"tuning.test_param\",\"value\":0.123}],\"rationale\":\"...\"}\n"
        "Now output JSON that matches the schema exactly."
    )

    # Ensure (best-effort) that the GenAI SDK can see an API key, so the
    # top-level report reflects reality even if the key is only stored in Colab Secrets.
    key_source = _ensure_genai_key_in_env()

    report: Dict[str, Any] = {
        "timestamp_utc": stamp,
        "is_colab": _is_colab(),
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "api_key_source": key_source,
        "env_has_GEMINI_API_KEY": bool(os.environ.get("GEMINI_API_KEY")),
        "env_has_GOOGLE_API_KEY": bool(os.environ.get("GOOGLE_API_KEY")),
        "backends": {},
    }

    # Backend 1: google.colab.ai
    report["backends"]["google.colab.ai"] = _probe_google_colab_ai(
        prompt,
        patch_prompt,
        prefer_model=args.prefer_model,
        max_models_log=args.max_models_log,
    )

    # Backend 2: google-genai SDK
    report["backends"]["google-genai"] = _probe_google_genai_sdk(
        prompt,
        prefer_model=args.prefer_model,
        max_models_log=args.max_models_log,
    )

    # Decide recommendation
    rec = "none"
    if report["backends"]["google.colab.ai"].get("text_ok"):
        rec = "google.colab.ai"
    if report["backends"]["google-genai"].get("json_schema_ok"):
        # Prefer genai if schema enforcement works (safer for autotuning)
        rec = "google-genai"
    report["recommended_backend"] = rec

    # Write outputs
    json_path = os.path.join(outdir, "ai_probe.json")
    txt_path = os.path.join(outdir, "ai_probe.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    lines: List[str] = []
    lines.append(f"AI probe @ {stamp} UTC")
    lines.append(f"Colab: {report['is_colab']}")
    lines.append(f"Python: {report['python']}")
    lines.append(f"Platform: {report['platform']}")
    lines.append(f"API key source (for google-genai): {report.get('api_key_source', 'unknown')}")
    lines.append("")
    lines.append("Backends:")
    for k, v in report["backends"].items():
        lines.append(f"- {k}: available={v.get('available')} text_ok={v.get('text_ok')} json_patch_ok={v.get('json_patch_ok', v.get('json_schema_ok'))}")
        if v.get("model_selected"):
            lines.append(f"    model_selected: {v.get('model_selected')}")
        if v.get("errors"):
            for e in v.get("errors")[:5]:
                lines.append(f"    error: {e}")
    lines.append("")
    lines.append(f"Recommended backend for autotuning: {rec}")
    lines.append("")
    lines.append("Wrote:")
    lines.append(f"  - {json_path}")
    lines.append(f"  - {txt_path}")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
