#!/usr/bin/env python3
"""
PDF formula -> external molecule API -> SMILES workflow.

Inputs:
  - local PDFs
  - OpenRouter/OpenAI-compatible API config in markush_config.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import socket
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

DEFAULT_API_TIMEOUT = 45
DEFAULT_API_RETRIES = 6
DEFAULT_API_BASE_DELAY = 2.0
DEFAULT_MODEL = (
    os.getenv("OPENROUTER_MODEL")
    or os.getenv("OPENAI_MODEL")
    or "MiniMax-M2.7"
)
DEFAULT_OUTDIR = "markush_outputs"
KEY_FILE_CANDIDATES = (".env",)

socket.setdefaulttimeout(DEFAULT_API_TIMEOUT + 15)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def default_workflow_config() -> dict[str, Any]:
    return {
        "api": {
            "api_key_env": "OPENROUTER_API_KEY",
            "base_url_env": "OPENROUTER_BASE_URL",
            "api_key": "",
            "base_url": "",
            "key_files": list(KEY_FILE_CANDIDATES),
        },
        "models": {
            "primary": DEFAULT_MODEL,
            "fallback": [],
        },
        "runtime": {
            "transport": "auto",
            "max_output_tokens": 16000,
            "max_pdf_chars": 180000,
            "max_pages_per_pdf": 80,
            "reuse_uploads": False,
            "force": False,
            "no_api": False,
        },
        "paths": {
            "outdir": DEFAULT_OUTDIR,
            "pdfs": [],
            "csvs": [],
        },
    }


def load_workflow_config(config_path: Optional[str]) -> dict[str, Any]:
    config = default_workflow_config()
    if not config_path:
        return config

    path = Path(config_path).expanduser()
    if not path.exists():
        return config

    loaded = read_json(path)
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Config file must contain a JSON object: {path}")
    return deep_merge(config, loaded)


def extract_api_key_from_text(text: str) -> Optional[str]:
    assignment = re.search(
        r"(?:OPENAI_API_KEY|api[_ -]?key)\s*[:=]\s*([A-Za-z0-9_\-]{20,}|sk-[A-Za-z0-9_\-]+)",
        text,
        flags=re.IGNORECASE,
    )
    if assignment:
        return assignment.group(1).strip().strip("'\"")

    sk_token = re.search(r"\bsk-[A-Za-z0-9_\-]{20,}\b", text)
    return sk_token.group(0) if sk_token else None


def extract_base_url_from_text(text: str) -> Optional[str]:
    match = re.search(
        r"(?:OPENAI_BASE_URL|base[_ -]?url)\s*[:=]\s*(https?://\S+)",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip().strip("'\"") if match else None


def load_openai_config(api_config: dict[str, Any]) -> tuple[str, Optional[str]]:
    api_key_env = str(api_config.get("api_key_env") or "OPENAI_API_KEY")
    base_url_env = str(api_config.get("base_url_env") or "OPENAI_BASE_URL")
    api_key = str(api_config.get("api_key") or "") or os.getenv(api_key_env)
    base_url = str(api_config.get("base_url") or "") or os.getenv(base_url_env)

    key_files = [
        Path(path).expanduser().resolve()
        for path in api_config.get("key_files", [])
    ]
    for key_file in key_files:
        if not key_file.exists():
            continue
        text = key_file.read_text(encoding="utf-8", errors="ignore")
        api_key = api_key or extract_api_key_from_text(text)
        base_url = base_url or extract_base_url_from_text(text)

    if not api_key:
        searched = ", ".join(str(path) for path in key_files)
        raise RuntimeError(
            "No OpenAI-compatible API key found. Set the configured API key env "
            f"or put the key in one of: {searched}"
        )
    return api_key, base_url


def require_openai_client(api_config: dict[str, Any]):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is not installed. Run: "
            "python3 -m pip install -r requirements.txt"
        ) from exc

    api_key, base_url = load_openai_config(api_config)
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            content_type = getattr(content, "type", None)
            if content_type is None and isinstance(content, dict):
                content_type = content.get("type")
            if content_type != "output_text":
                continue
            text = getattr(content, "text", None)
            if text:
                chunks.append(text.strip())
            elif isinstance(content, dict) and content.get("text"):
                chunks.append(str(content["text"]).strip())
    if chunks:
        return "\n".join(chunks)
    raise RuntimeError("Response did not contain output_text.")


def parse_json_response_text(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("Structured output was not a JSON object.")
        return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("Structured output was not a JSON object.")
        return parsed

    raise json.JSONDecodeError("No complete JSON object found", text, 0)


def default_formula_workflow_config() -> dict[str, Any]:
    return {
        "pdfs": [],
        "outdir": "formula_workflow_outputs",
        "llm": {
            "model": "",
            "pages_per_chunk": 8,
            "max_chunk_chars": 24000,
            "max_output_tokens": 4000,
        },
        "request": {
            "delay": 1.0,
            "api_timeout": DEFAULT_API_TIMEOUT,
            "api_retries": DEFAULT_API_RETRIES,
            "api_base_delay": DEFAULT_API_BASE_DELAY,
            "include_pubchem_formula": False,
        },
        "cache": {
            "force": False,
            "reuse_pdf_pages": True,
            "reuse_page_chunks": True,
            "reuse_compound_scope": True,
            "reuse_formula_extract": True,
            "reuse_activity_extract": True,
            "reuse_external_api": True,
        },
        "compound_scope": {
            "target_compound_ids": [],
            "excluded_compound_ids": [],
            "context_max_chars": 60000,
            "demote_reference_only_final_compounds": True,
        },
        "referee": {
            "enabled": True,
            "statuses": ["ambiguous", "resolved_bindingdb_sequence"],
            "context_chars_per_row": 4000,
            "max_output_tokens": 2000,
            "reuse_cache": True,
        },
        "activity": {
            "enabled": True,
            "primary_assay_keywords": [
                "KRAS G13D",
                "IC50",
                "HTRF",
                "TR-FRET",
                "biochemical",
            ],
        },
        "output": {
            "pdf_pages_json": "pdf_pages.json",
            "pdf_page_chunks_json": "pdf_page_chunks.json",
            "compound_scope_json": "compound_scope.json",
            "formula_records_json": "formula_records.json",
            "activity_records_json": "activity_records.json",
            "detailed_csv": "formula_to_smiles.csv",
            "final_csv": "",
            "ambiguity_referee_json": "ambiguity_referee.json",
        },
    }


def resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else (base_dir / path)


def resolve_output_path(path_text: str, outdir: Path) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else (outdir / path)


def arg_or_config(value: Any, config: dict[str, Any], *keys: str) -> Any:
    if value is not None:
        return value
    node: Any = config
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


FORMULA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "tables": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "table_ref": {"type": "string"},
                    "page_refs": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "string"},
                },
                "required": ["table_ref", "page_refs", "description"],
            },
        },
        "records": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "compound_id": {"type": "string"},
                    "raw_formula": {"type": "string"},
                    "formula_kind": {
                        "type": "string",
                        "enum": ["neutral", "mh_plus", "unknown"],
                    },
                    "source_type": {
                        "type": "string",
                        "enum": ["table", "characterization", "other"],
                    },
                    "page_refs": {"type": "array", "items": {"type": "string"}},
                    "evidence": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "compound_id",
                    "raw_formula",
                    "formula_kind",
                    "source_type",
                    "page_refs",
                    "evidence",
                    "confidence",
                ],
            },
        },
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["tables", "records", "notes"],
}


ACTIVITY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "tables": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "table_ref": {"type": "string"},
                    "page_refs": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "string"},
                    "activity_columns": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["table_ref", "page_refs", "description", "activity_columns"],
            },
        },
        "records": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "compound_id": {"type": "string"},
                    "assay_name": {"type": "string"},
                    "target": {"type": "string"},
                    "endpoint": {"type": "string"},
                    "value_text": {"type": "string"},
                    "qualifier": {"type": "string"},
                    "value_numeric": {"type": "string"},
                    "unit": {"type": "string"},
                    "selectivity_fold": {"type": "string"},
                    "source_type": {
                        "type": "string",
                        "enum": ["table", "text", "figure", "other"],
                    },
                    "page_refs": {"type": "array", "items": {"type": "string"}},
                    "evidence": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "compound_id",
                    "assay_name",
                    "target",
                    "endpoint",
                    "value_text",
                    "qualifier",
                    "value_numeric",
                    "unit",
                    "selectivity_fold",
                    "source_type",
                    "page_refs",
                    "evidence",
                    "confidence",
                ],
            },
        },
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["tables", "records", "notes"],
}


COMPOUND_SCOPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "paper": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "title": {"type": "string"},
                "abstract_summary": {"type": "string"},
                "primary_target_or_assay": {"type": "string"},
            },
            "required": ["title", "abstract_summary", "primary_target_or_assay"],
        },
        "final_compounds": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "compound_id": {"type": "string"},
                    "role": {"type": "string"},
                    "source_refs": {"type": "array", "items": {"type": "string"}},
                    "evidence": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["compound_id", "role", "source_refs", "evidence", "confidence"],
            },
        },
        "intermediates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "compound_id": {"type": "string"},
                    "role": {"type": "string"},
                    "source_refs": {"type": "array", "items": {"type": "string"}},
                    "evidence": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["compound_id", "role", "source_refs", "evidence", "confidence"],
            },
        },
        "excluded_compounds": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "compound_id": {"type": "string"},
                    "reason": {"type": "string"},
                    "source_refs": {"type": "array", "items": {"type": "string"}},
                    "evidence": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["compound_id", "reason", "source_refs", "evidence", "confidence"],
            },
        },
        "target_id_patterns": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "paper",
        "final_compounds",
        "intermediates",
        "excluded_compounds",
        "target_id_patterns",
        "notes",
    ],
}


CSV_NAME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "basename": {
            "type": "string",
            "description": "Short lowercase snake_case CSV basename without extension.",
        },
    },
    "required": ["basename"],
}


AMBIGUITY_REFEREE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "compound_id": {"type": "string"},
        "decision": {
            "type": "string",
            "enum": ["keep_current", "select_candidate", "unresolved"],
        },
        "selected_chembl_id": {"type": "string"},
        "selected_bindingdb_id": {"type": "string"},
        "confidence": {"type": "number"},
        "distinguishing_features": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": [
        "compound_id",
        "decision",
        "selected_chembl_id",
        "selected_bindingdb_id",
        "confidence",
        "distinguishing_features",
        "rationale",
    ],
}


def read_pdf_pages(pdf_paths: list[Path]) -> list[dict[str, Any]]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("Install PyMuPDF: python3 -m pip install -r requirements.txt") from exc

    pages: list[dict[str, Any]] = []
    for path in pdf_paths:
        doc = fitz.open(path)
        for index in range(doc.page_count):
            page = doc[index]
            text = re.sub(r"[ \t]+", " ", page.get_text("text").strip())
            pages.append(
                {
                    "filename": path.name,
                    "page": index + 1,
                    "image_count": len(page.get_images(full=True)),
                    "text": text,
                }
            )
    return pages


def page_chunks(
    pages: list[dict[str, Any]],
    pages_per_chunk: int,
    max_chunk_chars: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    char_count = 0

    def flush() -> None:
        nonlocal current, char_count
        if not current:
            return
        refs = [f"{p['filename']}:{p['page']}" for p in current]
        text = "\n".join(
            "\n--- PDF {filename} page {page} (images={image_count}) ---\n{text}".format(**p)
            for p in current
        )
        chunks.append({"chunk_id": f"chunk_{len(chunks) + 1:03d}", "page_refs": refs, "text": text})
        current = []
        char_count = 0

    for page in pages:
        page_len = len(page["text"])
        if current and (len(current) >= pages_per_chunk or char_count + page_len > max_chunk_chars):
            flush()
        current.append(page)
        char_count += page_len
    flush()
    return chunks


def load_cached_pdf_pages(path: Path) -> Optional[list[dict[str, Any]]]:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    pages = data.get("pages") if isinstance(data, dict) else data
    if not isinstance(pages, list):
        return None
    required = {"filename", "page", "image_count", "text"}
    if not all(isinstance(page, dict) and required.issubset(page) for page in pages):
        return None
    return pages


def write_pdf_pages_cache(path: Path, pdf_paths: list[Path], pages: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pdfs": [str(path) for path in pdf_paths],
        "pages": pages,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_cached_page_chunks(path: Path) -> Optional[list[dict[str, Any]]]:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = data.get("chunks") if isinstance(data, dict) else data
    if not isinstance(chunks, list):
        return None
    required = {"chunk_id", "page_refs", "text"}
    if not all(isinstance(chunk, dict) and required.issubset(chunk) for chunk in chunks):
        return None
    return chunks


def write_page_chunks_cache(path: Path, pdf_paths: list[Path], chunks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pdfs": [str(path) for path in pdf_paths],
        "chunks": chunks,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_or_load_page_chunks(
    pdf_paths: list[Path],
    pages_per_chunk: int,
    max_chunk_chars: int,
    pdf_pages_cache_path: Path,
    page_chunks_cache_path: Path,
    reuse_pdf_pages: bool,
    reuse_page_chunks: bool,
    force: bool,
) -> list[dict[str, Any]]:
    if reuse_page_chunks and not force:
        chunks = load_cached_page_chunks(page_chunks_cache_path)
        if chunks is not None:
            print(f"Using cached page chunks: {page_chunks_cache_path}")
            return chunks

    pages: Optional[list[dict[str, Any]]] = None
    if reuse_pdf_pages and not force:
        pages = load_cached_pdf_pages(pdf_pages_cache_path)
        if pages is not None:
            print(f"Using cached PDF pages: {pdf_pages_cache_path}")

    if pages is None:
        if not pdf_paths:
            raise RuntimeError("No PDFs found and no reusable PDF text/page-chunk cache exists.")
        pages = read_pdf_pages(pdf_paths)
        write_pdf_pages_cache(pdf_pages_cache_path, pdf_paths, pages)
        print(f"Built PDF page cache: {pdf_pages_cache_path}")

    chunks = page_chunks(pages, pages_per_chunk, max_chunk_chars)
    write_page_chunks_cache(page_chunks_cache_path, pdf_paths, chunks)
    print(f"Built {len(chunks)} chunks from {len(pages)} pages")
    return chunks


def normalize_compound_id(compound_id: str) -> str:
    return re.sub(r"\s+", "", str(compound_id or "").strip())


def canonical_target_compound_id(compound_id: str, target_set: set[str]) -> str:
    compound_id = normalize_compound_id(compound_id)
    if compound_id in target_set:
        return compound_id
    footnote_stripped = re.sub(r"^([A-Za-z]*\d+)[a-z]$", r"\1", compound_id)
    if footnote_stripped in target_set:
        return footnote_stripped
    return compound_id


def compound_sort_key(compound_id: str) -> tuple[Any, ...]:
    text = normalize_compound_id(compound_id)
    parts = re.split(r"(\d+)", text.lower())
    key: list[Any] = []
    for part in parts:
        if not part:
            continue
        key.append((0, int(part)) if part.isdigit() else (1, part))
    return tuple(key or [(1, text.lower())])


def expand_compound_id_text(text: str) -> list[str]:
    text = normalize_compound_id(text)
    if not text:
        return []
    if re.search(r"[,;]", text):
        ids: list[str] = []
        for part in re.split(r"[,;]", text):
            for compound_id in expand_compound_id_text(part):
                if compound_id and compound_id not in ids:
                    ids.append(compound_id)
        return ids
    range_match = re.fullmatch(r"([A-Za-z]*)(\d+)[-–—−]([A-Za-z]*)(\d+)([A-Za-z]?)", text)
    if range_match:
        prefix_a, start, prefix_b, end, suffix = range_match.groups()
        if prefix_a == prefix_b:
            start_i = int(start)
            end_i = int(end)
            if 0 <= end_i - start_i <= 300:
                return [f"{prefix_a}{idx}{suffix}" for idx in range(start_i, end_i + 1)]
    return [text]


def scope_entries_to_ids(entries: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for entry in entries:
        for compound_id in expand_compound_id_text(str(entry.get("compound_id", ""))):
            if compound_id and compound_id not in ids:
                ids.append(compound_id)
    return sorted(ids, key=compound_sort_key)


def compound_id_summary(compound_ids: list[str]) -> str:
    if not compound_ids:
        return "none"
    sorted_ids = sorted(compound_ids, key=compound_sort_key)
    if len(sorted_ids) <= 80:
        return ", ".join(sorted_ids)
    return ", ".join(sorted_ids[:80]) + f", ... ({len(sorted_ids)} total)"


def build_compound_scope_context(chunks: list[dict[str, Any]], max_chars: int) -> str:
    blocks: list[str] = []
    remaining = max_chars

    def append_block(label: str, text: str) -> None:
        nonlocal remaining
        if remaining <= 0 or not text.strip():
            return
        block = f"\n\n### {label}\n{text.strip()}"
        if len(block) > remaining:
            block = block[:remaining]
        blocks.append(block)
        remaining -= len(block)

    first_chunks_text = "\n".join(
        f"\n--- {', '.join(chunk.get('page_refs', []))} ---\n{chunk.get('text', '')}"
        for chunk in chunks[:2]
    )
    append_block("Front matter, title, abstract, introduction excerpts", first_chunks_text)

    caption_keywords = re.compile(
        r"\b("
        r"table|scheme|figure|fig\.|caption|abstract|result|results|sar|activity|"
        r"ic50|ec50|kd|ki|assay|inhibition|inhibitor|compound|compounds|analog|analogue|"
        r"synthesis|synthesized|prepared|intermediate|target|final"
        r")\b",
        flags=re.IGNORECASE,
    )
    relevant_lines: list[str] = []
    for chunk in chunks:
        refs = ", ".join(chunk.get("page_refs", []))
        for raw_line in str(chunk.get("text", "")).splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if len(line) < 8:
                continue
            if caption_keywords.search(line):
                relevant_lines.append(f"[{refs}] {line}")
            if len("\n".join(relevant_lines)) > max_chars:
                break
        if len("\n".join(relevant_lines)) > max_chars:
            break
    append_block("Result-table, Scheme/Table/Figure-caption, and compound-context lines", "\n".join(relevant_lines))
    return "".join(blocks).strip()


def compound_scope_prompt(context: str) -> str:
    return f"""\
You analyze a medicinal chemistry paper and identify which compound IDs should be treated as final target compounds for downstream formula/SMILES extraction.

Use the title, abstract, result tables, biological/SAR tables, Scheme/Table/Figure captions, and nearby compound-context lines.

Classify IDs:
- final_compounds: final target molecules, SAR analogs, assayed compounds, low-potency analogs in focused libraries, lead compounds, or molecules reported as the main inhibitor series.
- intermediates: synthetic precursors, scheme intermediates, protected/deprotected intermediates, building blocks, salts/reagents, or experimental-only precursors.
- excluded_compounds: IDs that should not be used for final SMILES output. Include intermediates, supplemental IDs like S12, and high-numbered compounds if they are synthetic intermediates rather than final assayed compounds.

Rules:
- Return only JSON matching the schema.
- Preserve compound IDs exactly enough to match the PDF text, but remove spaces around IDs.
- Prefer compact ID ranges when evidence supports them, e.g. use one final_compounds entry "7-41" instead of 35 separate entries.
- Use comma-separated compact IDs for small sets when helpful, e.g. "57,60,64".
- Do not include an ID as final merely because it appears in an experimental section.
- Exclude reference/comparator/control compounds and prior-program initial hits used as X-ray or assay baselines unless they are explicitly part of the final analog library output.
- Do include final analog libraries from result/SAR tables even if they are weak, nonselective, or early screening compounds.
- Do include analog ranges when the SI says LCMS/MS data for analogs or compounds in that range and the main paper evaluates that library.
- Prefer result/activity tables and paper captions over synthetic scheme numbering.
- If an ID is ambiguous, put it in excluded_compounds with a reason explaining the uncertainty.
- Keep evidence concise, paraphrased, and under 120 characters. Do not copy text containing quotation marks.
- Keep the whole JSON compact; avoid duplicating identical evidence across many IDs.

PDF-derived context:
{context}
"""


def infer_final_ranges_from_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    patterns = [
        re.compile(
            r"(?:LCMS|MS|mass spectrometric)\s+data\s+(?:is\s+listed\s+)?(?:for\s+)?(?:analogs?|analogues?|compounds?)\s+([A-Za-z]*\d+\s*[-–—−]\s*[A-Za-z]*\d+[A-Za-z]?)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"(?:analogs?|analogues?|compounds?)\s+([A-Za-z]*\d+\s*[-–—−]\s*[A-Za-z]*\d+[A-Za-z]?)\s+(?:were\s+)?(?:synthesi[sz]ed|prepared|evaluated|profiled)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"(?:biochemical\s+evaluation|SAR|focused\s+library|result[s]?)\s+.{0,120}?(?:analogs?|analogues?|compounds?)\s+([A-Za-z]*\d+\s*[-–—−]\s*[A-Za-z]*\d+[A-Za-z]?)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"(?:Table|Figure|Fig\.)\s+\w+\.?\s+.{0,220}?\(([A-Za-z]*\d+\s*[-–—−]\s*[A-Za-z]*\d+[A-Za-z]?)\)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"(?:synthesis|synthetic\s+procedure|experimental\s+procedure)s?\s+(?:of\s+|for\s+)?(?:analogs?|analogues?|compounds?)\s+([A-Za-z]*\d+\s*[-–—−]\s*[A-Za-z]*\d+[A-Za-z]?)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"(?:LCMS|NMR|characterization)\s+(?:spectra|chromatograms|data|records)?.{0,80}?(?:for\s+)?(?:analogs?|analogues?|compounds?)\s+([A-Za-z]*\d+\s*[-–—−]\s*[A-Za-z]*\d+[A-Za-z]?)",
            flags=re.IGNORECASE,
        ),
    ]
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in chunks:
        text = re.sub(r"\s+", " ", str(chunk.get("text", "")))
        for pattern in patterns:
            for match in pattern.finditer(text):
                compound_range = normalize_compound_id(match.group(1))
                if compound_range in seen:
                    continue
                seen.add(compound_range)
                evidence = text[max(0, match.start() - 80) : match.end() + 120].strip()
                entries.append(
                    {
                        "compound_id": compound_range,
                        "role": "final_analog_library_range_inferred_from_text",
                        "source_refs": list(chunk.get("page_refs", [])),
                        "evidence": evidence[:180],
                        "confidence": 0.88,
                    }
                )
    return entries


def augment_scope_from_text_patterns(scope: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    existing_final_ids = set(scope_entries_to_ids(scope.get("final_compounds", [])))
    excluded_ids = set(scope_entries_to_ids(scope.get("excluded_compounds", [])))
    for entry in infer_final_ranges_from_chunks(chunks):
        inferred_ids = set(expand_compound_id_text(str(entry.get("compound_id", ""))))
        if not inferred_ids or inferred_ids.issubset(existing_final_ids):
            continue
        if inferred_ids & excluded_ids:
            continue
        scope.setdefault("final_compounds", []).append(entry)
        existing_final_ids.update(inferred_ids)
        scope.setdefault("notes", []).append(
            f"Added final analog range {entry['compound_id']} from LCMS/MS or analog-library text."
        )
    return scope


def demote_reference_final_compounds(scope: dict[str, Any], manual_target_ids: list[str]) -> dict[str, Any]:
    manual_targets = {
        normalize_compound_id(compound_id)
        for item in manual_target_ids
        for compound_id in expand_compound_id_text(str(item))
    }
    scope["excluded_compounds"] = [
        entry
        for entry in scope.get("excluded_compounds", [])
        if str(entry.get("reason", "")) != "reference_or_comparator_not_final_library_member"
    ]
    scope["notes"] = [
        note
        for note in scope.get("notes", [])
        if not str(note).startswith("Demoted reference/comparator compound ")
    ]
    reference_patterns = (
        r"\bcomparator\b",
        r"\breference\b",
        r"\bcontrol\b",
        r"\binitial\s+hit\b",
        r"\bprior[-\s]+program\b",
        r"\bx[-\s]?ray\b",
        r"\bbaseline\b",
    )
    final_library_terms = (
        "analog library",
        "analogue library",
        "sar analog",
        "focused library",
        "main inhibitor series",
        "final analog",
    )
    retained_final: list[dict[str, Any]] = []
    for entry in scope.get("final_compounds", []):
        expanded_ids = expand_compound_id_text(str(entry.get("compound_id", "")))
        text = " ".join(
            str(entry.get(key, ""))
            for key in ("role", "evidence", "reason")
        ).lower()
        should_demote = (
            len(expanded_ids) == 1
            and expanded_ids[0] not in manual_targets
            and any(re.search(pattern, text) for pattern in reference_patterns)
            and not any(term in text for term in final_library_terms)
        )
        if not should_demote:
            retained_final.append(entry)
            continue
        try:
            confidence = float(entry.get("confidence", 0) or 0)
        except (TypeError, ValueError):
            confidence = 0.0
        scope.setdefault("excluded_compounds", []).append(
            {
                "compound_id": expanded_ids[0],
                "reason": "reference_or_comparator_not_final_library_member",
                "source_refs": list(entry.get("source_refs", [])),
                "evidence": str(entry.get("evidence", ""))[:120],
                "confidence": max(0.85, confidence),
            }
        )
        scope.setdefault("notes", []).append(
            f"Demoted reference/comparator compound {expanded_ids[0]} from final scope."
        )
    scope["final_compounds"] = retained_final
    return scope


def apply_compound_scope_overrides(
    scope: dict[str, Any],
    target_ids: list[str],
    excluded_ids: list[str],
) -> dict[str, Any]:
    final_ids = set(scope_entries_to_ids(scope.get("final_compounds", [])))
    excluded_set = set(scope_entries_to_ids(scope.get("excluded_compounds", [])))
    for compound_id in target_ids:
        for expanded in expand_compound_id_text(compound_id):
            final_ids.add(expanded)
            excluded_set.discard(expanded)
    for compound_id in excluded_ids:
        for expanded in expand_compound_id_text(compound_id):
            excluded_set.add(expanded)
            final_ids.discard(expanded)

    existing_final = {
        normalize_compound_id(entry.get("compound_id", "")): entry
        for entry in scope.get("final_compounds", [])
    }
    existing_excluded = {
        normalize_compound_id(entry.get("compound_id", "")): entry
        for entry in scope.get("excluded_compounds", [])
    }
    scope["final_compounds"] = [
        existing_final.get(
            compound_id,
            {
                "compound_id": compound_id,
                "role": "manual_config_final_target",
                "source_refs": [],
                "evidence": "Added from formula_workflow.compound_scope.target_compound_ids.",
                "confidence": 1.0,
            },
        )
        for compound_id in sorted(final_ids, key=compound_sort_key)
    ]
    scope["excluded_compounds"] = [
        existing_excluded.get(
            compound_id,
            {
                "compound_id": compound_id,
                "reason": "manual_config_excluded",
                "source_refs": [],
                "evidence": "Added from formula_workflow.compound_scope.excluded_compound_ids.",
                "confidence": 1.0,
            },
        )
        for compound_id in sorted(excluded_set, key=compound_sort_key)
    ]
    return scope


def discover_compound_scope(
    client: Any,
    model: str,
    chunks: list[dict[str, Any]],
    cache_path: Path,
    max_output_tokens: int,
    context_max_chars: int,
    target_ids: list[str],
    excluded_ids: list[str],
    reuse_cache: bool,
    force: bool,
) -> dict[str, Any]:
    if cache_path.exists() and reuse_cache and not force:
        scope = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"Using cached {cache_path.name}")
    else:
        context = build_compound_scope_context(chunks, context_max_chars)
        print(f"Calling {model} for compound scope")
        scope = call_json_agent(
            client,
            model,
            compound_scope_prompt(context),
            "compound_scope",
            COMPOUND_SCOPE_SCHEMA,
            min(max_output_tokens, 2500),
        )
    scope = augment_scope_from_text_patterns(scope, chunks)
    scope = demote_reference_final_compounds(scope, target_ids)
    scope = apply_compound_scope_overrides(scope, target_ids, excluded_ids)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(scope, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return scope


def scope_hash(scope: dict[str, Any]) -> str:
    payload = {
        "final_compounds": scope_entries_to_ids(scope.get("final_compounds", [])),
        "excluded_compounds": scope_entries_to_ids(scope.get("excluded_compounds", [])),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def formula_prompt(chunk: dict[str, Any], scope: dict[str, Any]) -> str:
    final_ids = scope_entries_to_ids(scope.get("final_compounds", []))
    intermediate_ids = scope_entries_to_ids(scope.get("intermediates", []))
    excluded_ids = scope_entries_to_ids(scope.get("excluded_compounds", []))
    paper = scope.get("paper", {})
    return f"""\
You extract molecular formula records from PDF text.

Goal:
- Find formulas for final target compounds identified by an earlier paper-level scope analysis.
- First identify table-like regions and captions.
- Then extract formula records from those tables or from characterization lines
  such as "LCMS (ESI): calcd. for C44H51ClF4N9O2+ [M+H]+".

Rules:
- Only include final target compound IDs listed below.
- Do not include intermediate or excluded IDs, even if they have LCMS/HRMS formulas.
- Preserve compound IDs exactly enough to match the ID list.
- If the table uses lowercase footnote suffixes such as 9a or 22a, map them to the base target IDs such as 9 or 22 when those base IDs are listed.
- If the text says "[M+H]+", set formula_kind="mh_plus"; the raw formula is the
  printed protonated formula.
- If the formula is printed as a neutral molecular formula, set formula_kind="neutral".
- Do not infer formulas from SMILES or drawings.
- Evidence should be short but enough to audit the extraction.

Paper title: {paper.get("title", "")}
Primary target/assay: {paper.get("primary_target_or_assay", "")}

Final target compound IDs to include:
{compound_id_summary(final_ids)}

Intermediate IDs to exclude:
{compound_id_summary(intermediate_ids)}

Other excluded IDs:
{compound_id_summary(excluded_ids)}

Chunk pages: {", ".join(chunk["page_refs"])}

PDF text:
{chunk["text"]}
"""


def activity_prompt(
    chunk: dict[str, Any],
    scope: dict[str, Any],
    primary_assay_keywords: Optional[list[str]] = None,
) -> str:
    final_ids = scope_entries_to_ids(scope.get("final_compounds", []))
    intermediate_ids = scope_entries_to_ids(scope.get("intermediates", []))
    excluded_ids = scope_entries_to_ids(scope.get("excluded_compounds", []))
    paper = scope.get("paper", {})
    primary_keywords = [str(item) for item in (primary_assay_keywords or []) if str(item).strip()]
    return f"""\
You extract biological/activity records from medicinal chemistry PDF text.

Goal:
- Find measured activity/selectivity/assay data for final target compounds.
- Focus on SAR/result/activity tables and nearby text.
- Extract one record per compound per endpoint when values are explicit.

Rules:
- Only include final target compound IDs listed below.
- Do not include intermediates or excluded IDs.
- Do not extract LCMS, HRMS, NMR, yield, RT, MW, formula, PK exposure, crystallography, or synthetic procedure values as activity.
- Valid activity endpoints include IC50, EC50, Ki, Kd, inhibition %, selectivity fold, biochemical potency, cellular potency, viability, and similar assay readouts.
- Preserve the raw value text exactly enough to audit units and qualifiers.
- Put numeric value without unit in value_numeric when obvious; otherwise empty string.
- Put unit as nM, uM, mM, %, fold, or empty string.
- If a table gives "0.34 (15x)", value_text should keep the full text, value_numeric="0.34", unit should reflect the table header, and selectivity_fold="15".
- If separate columns give KRAS G13D and WT values, create separate records or include selectivity_fold when directly given.
- Evidence should identify the table row/text line and endpoint.
- When multiple activity columns exist, prioritize records matching these primary assay keywords:
  {", ".join(primary_keywords) if primary_keywords else "(not specified)"}

Paper title: {paper.get("title", "")}
Primary target/assay: {paper.get("primary_target_or_assay", "")}

Final target compound IDs to include:
{compound_id_summary(final_ids)}

Intermediate IDs to exclude:
{compound_id_summary(intermediate_ids)}

Other excluded IDs:
{compound_id_summary(excluded_ids)}

Chunk pages: {", ".join(chunk["page_refs"])}

PDF text:
{chunk["text"]}
"""


def chunk_mentions_compound_id(chunk: dict[str, Any], compound_id: str) -> bool:
    text = str(chunk.get("text", ""))
    cid = re.escape(normalize_compound_id(compound_id))
    patterns = [
        rf"\bcompound\s+{cid}\b",
        rf"\bcompd\.?\s+{cid}\b",
        rf"\bcpd\.?\s+{cid}\b",
        rf"\bafford(?:ed)?\s+(?:compound\s+)?{cid}\b",
        rf"\bobtain(?:ed)?\s+(?:the\s+)?(?:compound\s+)?{cid}\b",
        rf"\b{cid}\s*\(",
        rf"\b{cid}\s*:",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def focused_formula_snippets(missing_ids: list[str], chunks: list[dict[str, Any]]) -> list[str]:
    snippets: list[str] = []
    for compound_id in missing_ids:
        cid = re.escape(normalize_compound_id(compound_id))
        patterns = [
            rf"(?:afford(?:ed)?|obtain(?:ed)?|provide(?:d)?|yield(?:ed)?)\s+(?:the\s+)?(?:compound\s+)?{cid}\b.{{0,1800}}?(?:LCMS|LC/MS|HRMS).{{0,700}}",
            rf"(?:Synthesis\s+of\s+compound\s+{cid}|afford(?:ed)?\s+(?:compound\s+)?{cid}|obtain(?:ed)?\s+(?:the\s+)?compound\s+{cid}|compound\s+{cid})"  # noqa: E501
            rf".{{0,3200}}?(?:LCMS|LC/MS|HRMS).{{0,700}}",
        ]
        for chunk in chunks:
            text = str(chunk.get("text", ""))
            for pattern in patterns:
                for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
                    snippet = re.sub(r"\n{3,}", "\n\n", match.group(0)).strip()
                    if not snippet:
                        continue
                    if "calcd" not in snippet.lower() or not re.search(r"\bC\d+H\d+", snippet):
                        continue
                    block = (
                        f"--- Missing compound {compound_id}; "
                        f"{chunk.get('chunk_id', '')}; {', '.join(chunk.get('page_refs', []))} ---\n"
                        f"{snippet}"
                    )
                    if block not in snippets:
                        snippets.append(block)
                    if sum(f"Missing compound {compound_id};" in s for s in snippets) >= 6:
                        break
                if sum(f"Missing compound {compound_id};" in s for s in snippets) >= 6:
                    break
            if sum(f"Missing compound {compound_id};" in s for s in snippets) >= 6:
                break
    return snippets


def focused_formula_prompt(missing_ids: list[str], snippets: list[str], scope: dict[str, Any]) -> str:
    focused_text = "\n\n".join(snippets)
    return f"""\
You extract missing molecular formula records from focused PDF text.

Only extract these final target compound IDs:
{compound_id_summary(missing_ids)}

Rules:
- Return JSON matching the formula schema.
- Do not include intermediates or reagents.
- If an ID appears as a final product in a synthesis section, extract its LCMS/HRMS formula.
- If the text says "[M+H]+", set formula_kind="mh_plus"; raw_formula is the printed protonated formula.
- If a table has a neutral molecular formula column, set formula_kind="neutral".
- If a table uses footnote suffixes like 31a, map to base target ID 31 when listed above.
- Evidence should mention the short LCMS/HRMS line or table row.

Paper title: {scope.get("paper", {}).get("title", "")}

Focused PDF text:
{focused_text}
"""


def formula_records_from_snippets(missing_ids: list[str], snippets: list[str]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for compound_id in missing_ids:
        matching = [snippet for snippet in snippets if f"Missing compound {compound_id};" in snippet]
        best: Optional[tuple[float, str, list[str], str]] = None
        for snippet in matching:
            matches = list(re.finditer(
                r"(?:LCMS|LC/MS|HRMS)[\s\S]{0,180}?calcd\.?\s*(?:for\s*)?([A-Z][A-Za-z0-9]+)\+?\s*\[M\+H\]\+",
                snippet,
                flags=re.IGNORECASE,
            ))
            if not matches:
                continue
            page_refs_match = re.search(r";\s*([^\n]+?)\s*---", snippet)
            page_refs = []
            if page_refs_match:
                page_refs = [
                    item.strip()
                    for item in page_refs_match.group(1).split(",")
                    if item.strip()
                ]
            for index, match in enumerate(matches):
                start = max(0, match.start() - 900)
                end = min(len(snippet), match.end() + 250)
                context = re.sub(r"\s+", " ", snippet[start:end]).strip()
                context_lower = context.lower()
                score = float(index) * 0.05
                cid = re.escape(normalize_compound_id(compound_id))
                if re.search(
                    rf"(?:afford(?:ed)?|obtain(?:ed)?|provide(?:d)?|yield(?:ed)?)\s+(?:the\s+)?(?:title\s+)?(?:compound\s+)?{cid}\b",
                    context_lower,
                    flags=re.IGNORECASE,
                ):
                    score += 3.0
                if re.search(r"\btitle\s+compound\b", context_lower):
                    score += 2.0
                if re.search(r"\bcompound\s+" + cid + r"\b", context_lower, flags=re.IGNORECASE):
                    score += 0.5
                if re.search(r"\b(?:tert-butyl|boc|protected|intermediate|crude product|compound\s+S\d+)\b", context_lower):
                    score -= 1.5
                candidate = (score, match.group(1), page_refs, context[:300])
                if best is None or candidate[0] > best[0]:
                    best = candidate
        if best is None:
            continue
        _, raw_formula, page_refs, evidence = best
        records.append(
            {
                "compound_id": compound_id,
                "raw_formula": raw_formula,
                "formula_kind": "mh_plus",
                "source_type": "characterization",
                "page_refs": page_refs,
                "evidence": evidence,
                "confidence": 0.9,
            }
        )
    return {"tables": [], "records": records, "notes": ["Deterministic focused LCMS formula fallback."]}


def extract_missing_formula_records(
    client: Any,
    model: str,
    chunks: list[dict[str, Any]],
    scope: dict[str, Any],
    missing_ids: list[str],
    cache_path: Path,
    max_output_tokens: int,
    reuse_cache: bool,
    force: bool,
) -> Optional[dict[str, Any]]:
    focused_chunks = [
        chunk
        for chunk in chunks
        if any(chunk_mentions_compound_id(chunk, compound_id) for compound_id in missing_ids)
    ]
    if not focused_chunks:
        return None
    snippets = focused_formula_snippets(missing_ids, focused_chunks)
    if not snippets:
        return None
    fallback_result = formula_records_from_snippets(missing_ids, snippets)
    if cache_path.exists() and reuse_cache and not force:
        print(f"Using cached {cache_path.name}")
        return json.loads(cache_path.read_text(encoding="utf-8"))

    print(f"Calling {model} for missing formula IDs: {compound_id_summary(missing_ids)}")
    try:
        result = call_json_agent(
            client,
            model,
            focused_formula_prompt(missing_ids, snippets, scope),
            "formula_extract_missing",
            FORMULA_SCHEMA,
            max_output_tokens,
        )
    except Exception:
        result = {"tables": [], "records": [], "notes": ["Focused missing formula LLM call failed."]}
    existing_ids = {
        canonical_target_compound_id(record.get("compound_id", ""), set(missing_ids))
        for record in result.get("records", [])
    }
    for record in fallback_result.get("records", []):
        if record["compound_id"] not in existing_ids:
            result.setdefault("records", []).append(record)
        else:
            result.setdefault("records", []).append(record)
    result.setdefault("notes", []).extend(fallback_result.get("notes", []))
    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def csv_name_prompt(pdf_paths: list[Path], merged: dict[str, Any]) -> str:
    tables = merged.get("tables", [])[:12]
    table_lines = [
        "- {table_ref}: {description}".format(
            table_ref=str(table.get("table_ref", "")).strip(),
            description=str(table.get("description", "")).strip(),
        )
        for table in tables
    ]
    compound_ids = [
        normalize_compound_id(record.get("compound_id", ""))
        for record in merged.get("records", [])
        if normalize_compound_id(record.get("compound_id", ""))
    ]
    compound_range = compound_id_summary(compound_ids) if compound_ids else "unknown"
    return f"""\
Create a concise CSV filename basename for formula-to-SMILES results.

Rules:
- Return only JSON matching the schema.
- basename must be lowercase snake_case.
- basename must be 3 to 8 words.
- Do not include a file extension.
- Prefer the paper target, molecule series, or assay target over generic words.
- Include "smiles" only if it helps clarity.

PDF filenames:
{chr(10).join("- " + path.name for path in pdf_paths)}

Extracted compound range: {compound_range}

Extracted table context:
{chr(10).join(table_lines) if table_lines else "- none"}
"""


def call_json_agent(
    client: Any,
    model: str,
    prompt: str,
    schema_name: str,
    schema: dict[str, Any],
    max_output_tokens: int,
) -> dict[str, Any]:
    response = client.responses.create(
        model=model,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
        max_output_tokens=max_output_tokens,
    )
    text = extract_response_text(response)
    try:
        return parse_json_response_text(text)
    except Exception:
        repair_prompt = f"""\
Repair the following invalid JSON into valid JSON that matches the requested schema.

Rules:
- Return JSON only.
- Preserve all compound IDs and classifications.
- Paraphrase evidence strings if needed.
- Escape all special characters correctly.

Invalid JSON/text:
{text}
"""
        repaired = client.responses.create(
            model=model,
            input=repair_prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": f"{schema_name}_repair",
                    "strict": True,
                    "schema": schema,
                }
            },
            max_output_tokens=max_output_tokens,
        )
        return parse_json_response_text(extract_response_text(repaired))


def safe_csv_basename(text: str, fallback: str = "formula_smiles") -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        return fallback
    parts = text.split("_")
    if len(parts) > 10:
        text = "_".join(parts[:10])
    return text[:120].strip("_") or fallback


def final_csv_path_from_config(
    final_csv_text: Any,
    outdir: Path,
    client: Any,
    model: str,
    pdf_paths: list[Path],
    merged: dict[str, Any],
    max_output_tokens: int,
    force: bool,
) -> Path:
    if final_csv_text:
        return resolve_output_path(str(final_csv_text), outdir)

    cache_path = outdir / "csv_filename.json"
    result: dict[str, Any]
    if cache_path.exists() and not force:
        result = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"Using cached {cache_path.name}")
    else:
        print(f"Calling {model} for CSV filename")
        result = call_json_agent(
            client,
            model,
            csv_name_prompt(pdf_paths, merged),
            "csv_filename",
            CSV_NAME_SCHEMA,
            min(max_output_tokens, 1000),
        )
        cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    basename = safe_csv_basename(str(result.get("basename", "")))
    filename = basename if basename.endswith(".csv") else f"{basename}.csv"
    return resolve_output_path(filename, outdir)


def parse_formula(formula: str) -> Optional[dict[str, int]]:
    counts: dict[str, int] = {}
    formula_text = re.sub(r"\[?M\s*\+\s*H\]?\+", "", str(formula or ""), flags=re.IGNORECASE)
    cleaned = re.sub(r"[^A-Za-z0-9]", "", formula_text)
    text = str(formula or "").strip()
    text = re.sub(r"\[\s*M\s*\+\s*H\s*\]\s*\+?", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"[\s\+\-]+$", "", text)
    cleaned = re.sub(r"[^A-Za-z0-9]", "", cleaned)
    if not cleaned:
        return None
    pos = 0
    for match in re.finditer(r"([A-Z][a-z]?)(\d*)", cleaned):
        if match.start() != pos:
            return None
        element, count_text = match.groups()
        counts[element] = counts.get(element, 0) + int(count_text or "1")
        pos = match.end()
    if pos != len(cleaned) or "C" not in counts:
        return None
    return counts


def formula_text_has_explicit_charge(raw_formula: str) -> bool:
    text = str(raw_formula or "").strip()
    text = re.sub(r"\[\s*M\s*\+\s*H\s*\]\s*\+?", "", text, flags=re.IGNORECASE).strip()
    return bool(re.search(r"\+$", text))


def format_formula(counts: dict[str, int]) -> str:
    order = []
    if "C" in counts:
        order.append("C")
    if "H" in counts:
        order.append("H")
    order.extend(sorted(k for k in counts if k not in {"C", "H"}))
    parts = []
    for element in order:
        count = counts[element]
        if count <= 0:
            continue
        parts.append(element + (str(count) if count != 1 else ""))
    return "".join(parts)


def formula_text_looks_protonated(raw_formula: str) -> bool:
    text = str(raw_formula or "").strip()
    return bool(re.search(r"\+$", text)) and not re.search(r"\[?M\s*\+\s*H\]?\+", text, flags=re.IGNORECASE)


def normalize_formula_with_reason(
    raw_formula: str,
    formula_kind: str,
    source_type: str = "",
) -> tuple[str, str]:
    counts = parse_formula(raw_formula)
    if not counts:
        return "", "invalid_or_missing_formula"
    source_norm = str(source_type or "").strip().lower()
    kind_norm = str(formula_kind or "").strip().lower()
    if source_norm == "table":
        return format_formula(counts), "table_formula_treated_as_neutral"
    if kind_norm == "mh_plus" and counts.get("H", 0) > 0 and formula_text_looks_protonated(raw_formula):
        counts = dict(counts)
        counts["H"] -= 1
        return format_formula(counts), "subtracted_h_from_explicit_mh_plus_formula"
    if kind_norm == "mh_plus":
        return format_formula(counts), "mh_plus_label_without_formula_ion_marker_treated_as_neutral"
    return format_formula(counts), "neutral_formula"


def normalize_formula(raw_formula: str, formula_kind: str) -> str:
    formula, _reason = normalize_formula_with_reason(raw_formula, formula_kind)
    return formula


def formula_atom_signature(raw_formula: str) -> str:
    counts = parse_formula(raw_formula)
    return format_formula(counts) if counts else ""


def formula_record_score(record: dict[str, Any]) -> float:
    confidence = float(record.get("confidence", 0) or 0)
    evidence = str(record.get("evidence", "")).lower()
    source_type = record.get("source_type", "other")
    score = confidence * 10
    score += {"characterization": 0.8, "table": 0.6, "other": 0.0}.get(source_type, 0.0)
    if "lcms" in evidence or "[m+h]" in evidence or "hrms" in evidence:
        score += 0.5
    if "not extracted" in evidence or "likely" in evidence or "no explicit formula" in evidence:
        score -= 5.0
    if "tert-butyl" in evidence or "boc" in evidence or "protected" in evidence:
        score -= 1.5
    if re.search(r"\btitle\s+compound\b", evidence):
        score += 1.5
    if not record.get("page_refs"):
        score -= 1.0
    return score


def parse_activity_numeric(value_text: Any) -> Optional[float]:
    text = str(value_text or "").strip().replace(",", "")
    if not text:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def normalize_activity_to_nM(value: Optional[float], unit: str) -> Optional[float]:
    if value is None:
        return None
    unit_norm = unit.strip().lower().replace("μ", "u").replace("µ", "u")
    if unit_norm in {"nm", "nanomolar", "nanomol/l", "nanomole"}:
        return value
    if unit_norm in {"um", "micromolar", "umol/l", "micromole"}:
        return value * 1000.0
    if unit_norm in {"mm", "millimolar", "mmol/l", "millimole"}:
        return value * 1_000_000.0
    return None


def parse_selectivity_fold(value_text: Any, *, allow_bare_numeric: bool = False) -> Optional[float]:
    text = str(value_text or "").strip()
    if not text:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*[x×]", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    if re.search(r"\bfold\b", text, flags=re.IGNORECASE):
        return parse_activity_numeric(text)
    if re.search(r"\bselectivit(?:y|ies)\b", text, flags=re.IGNORECASE):
        return parse_activity_numeric(text)
    if allow_bare_numeric:
        return parse_activity_numeric(text)
    return None


def activity_record_score(
    record: dict[str, Any],
    primary_assay_keywords: Optional[list[str]] = None,
) -> float:
    score = float(record.get("confidence", 0) or 0) * 10
    text = " ".join(
        str(record.get(key, ""))
        for key in ("assay_name", "target", "endpoint", "evidence", "value_text")
    ).lower()
    for keyword in primary_assay_keywords or []:
        keyword_text = str(keyword).strip().lower()
        if keyword_text and keyword_text in text:
            score += 2.5
    if "ic50" in text:
        score += 2.0
    if "ec50" in text or "ki" in text or "kd" in text:
        score += 1.0
    if "biochemical" in text or "htrf" in text or "tr-fret" in text or "tr fret" in text:
        score += 1.5
    if "selectivity" in text:
        score += 1.0
    primary_hit = any(
        str(keyword).strip().lower() in text
        for keyword in primary_assay_keywords or []
        if str(keyword).strip()
    )
    if "wt" in text and not primary_hit:
        score -= 3.0
    if "cell" in text or "viability" in text:
        score -= 1.0
    if normalize_activity_to_nM(
        parse_activity_numeric(record.get("value_numeric") or record.get("value_text")),
        str(record.get("unit", "")),
    ) is not None:
        score += 1.0
    return score


def compact_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def normalize_activity_record(
    record: dict[str, Any],
    primary_assay_keywords: Optional[list[str]] = None,
) -> dict[str, Any]:
    value = parse_activity_numeric(record.get("value_numeric") or record.get("value_text"))
    unit = str(record.get("unit", ""))
    value_nM = normalize_activity_to_nM(value, unit)
    selectivity = parse_selectivity_fold(record.get("selectivity_fold"), allow_bare_numeric=True)
    if selectivity is None:
        selectivity = parse_selectivity_fold(record.get("value_text"))
    normalized = {
        "compound_id": str(record.get("compound_id", "")).strip(),
        "assay_name": str(record.get("assay_name", "")).strip(),
        "target": str(record.get("target", "")).strip(),
        "endpoint": str(record.get("endpoint", "")).strip(),
        "value_text": str(record.get("value_text", "")).strip(),
        "qualifier": str(record.get("qualifier", "")).strip(),
        "value_numeric": compact_float(value),
        "unit": unit.strip(),
        "value_nM": compact_float(value_nM),
        "selectivity_fold": compact_float(selectivity),
        "source_type": str(record.get("source_type", "")).strip(),
        "page_refs": list(record.get("page_refs", []) or []),
        "evidence": str(record.get("evidence", "")).strip(),
        "confidence": float(record.get("confidence", 0) or 0),
    }
    normalized["score"] = activity_record_score(normalized, primary_assay_keywords)
    return normalized


def merge_activity_records(
    chunk_results: list[dict[str, Any]],
    target_ids: list[str],
    excluded_ids: list[str],
    primary_assay_keywords: Optional[list[str]] = None,
) -> dict[str, Any]:
    target_set = {normalize_compound_id(compound_id) for compound_id in target_ids}
    excluded_set = {normalize_compound_id(compound_id) for compound_id in excluded_ids}
    by_id: dict[str, dict[str, Any]] = {}
    tables: list[dict[str, Any]] = []
    notes: list[str] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for result in chunk_results:
        tables.extend(result.get("tables", []))
        notes.extend(result.get("notes", []))
        for record in result.get("records", []):
            cid = canonical_target_compound_id(record.get("compound_id", ""), target_set)
            if not cid or (target_set and cid not in target_set) or cid in excluded_set:
                continue
            normalized = normalize_activity_record(
                {**record, "compound_id": cid},
                primary_assay_keywords,
            )
            if not normalized.get("value_text") and not normalized.get("selectivity_fold"):
                continue
            dedupe_key = (
                cid,
                normalized.get("assay_name", ""),
                normalized.get("target", ""),
                normalized.get("endpoint", ""),
                normalized.get("value_text", ""),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            dest = by_id.setdefault(
                cid,
                {
                    "compound_id": cid,
                    "activities": [],
                    "activity_source_pages": [],
                    "activity_evidence": [],
                    "activity_warnings": [],
                },
            )
            dest["activities"].append(normalized)
            for page_ref in normalized.get("page_refs", []):
                if page_ref not in dest["activity_source_pages"]:
                    dest["activity_source_pages"].append(page_ref)
            if normalized.get("evidence"):
                dest["activity_evidence"].append(normalized["evidence"])
    for dest in by_id.values():
        activities = sorted(
            dest["activities"],
            key=lambda item: (-float(item.get("score", 0) or 0), item.get("assay_name", "")),
        )
        dest["activities"] = activities
        primary = activities[0] if activities else {}
        dest["primary_activity_assay"] = primary.get("assay_name", "")
        dest["primary_activity_target"] = primary.get("target", "")
        dest["primary_activity_endpoint"] = primary.get("endpoint", "")
        dest["primary_activity_value_text"] = primary.get("value_text", "")
        dest["primary_activity_value_nM"] = primary.get("value_nM", "")
        dest["primary_activity_selectivity_fold"] = primary.get("selectivity_fold", "")
        dest["activity_summary"] = " | ".join(
            " ".join(
                item
                for item in [
                    activity.get("target", ""),
                    activity.get("endpoint", ""),
                    activity.get("value_text", ""),
                    f"sel={activity.get('selectivity_fold')}" if activity.get("selectivity_fold") else "",
                ]
                if item
            )
            for activity in activities[:5]
        )
    return {
        "tables": tables,
        "records": sorted(by_id.values(), key=lambda row: compound_sort_key(row["compound_id"])),
        "notes": sorted(set(notes)),
    }


def attach_activity_records(
    merged_formula: dict[str, Any],
    merged_activity: dict[str, Any],
) -> dict[str, Any]:
    activity_by_id = {
        normalize_compound_id(record.get("compound_id", "")): record
        for record in merged_activity.get("records", [])
    }
    for record in merged_formula.get("records", []):
        activity = activity_by_id.get(normalize_compound_id(record.get("compound_id", "")), {})
        record["activity_summary"] = activity.get("activity_summary", "")
        record["primary_activity_assay"] = activity.get("primary_activity_assay", "")
        record["primary_activity_target"] = activity.get("primary_activity_target", "")
        record["primary_activity_endpoint"] = activity.get("primary_activity_endpoint", "")
        record["primary_activity_value_text"] = activity.get("primary_activity_value_text", "")
        record["primary_activity_value_nM"] = activity.get("primary_activity_value_nM", "")
        record["primary_activity_selectivity_fold"] = activity.get(
            "primary_activity_selectivity_fold",
            "",
        )
        record["activity_source_pages"] = activity.get("activity_source_pages", [])
        record["activity_evidence"] = activity.get("activity_evidence", [])
        record["activities_json"] = json.dumps(
            activity.get("activities", []),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return merged_formula


def merge_formula_records(
    chunk_results: list[dict[str, Any]],
    target_ids: list[str],
    excluded_ids: list[str],
) -> dict[str, Any]:
    by_id: dict[str, dict[str, Any]] = {}
    tables: list[dict[str, Any]] = []
    notes: list[str] = []
    target_set = {normalize_compound_id(compound_id) for compound_id in target_ids}
    excluded_set = {normalize_compound_id(compound_id) for compound_id in excluded_ids}
    for result in chunk_results:
        tables.extend(result.get("tables", []))
        notes.extend(result.get("notes", []))
        for record in result.get("records", []):
            cid = canonical_target_compound_id(record.get("compound_id", ""), target_set)
            if not cid:
                continue
            if target_set and cid not in target_set:
                continue
            if cid in excluded_set:
                continue
            formula_kind = record.get("formula_kind", "unknown")
            source_type = record.get("source_type", "other")
            neutral, normalization_reason = normalize_formula_with_reason(
                record.get("raw_formula", ""),
                formula_kind,
                source_type,
            )
            if not neutral:
                continue
            score = formula_record_score(record)
            dest = by_id.setdefault(
                cid,
                {
                    "compound_id": cid,
                    "neutral_formula": neutral,
                    "formula_score": score,
                    "raw_formulas": [],
                    "formula_kinds": [],
                    "formula_normalization_reasons": [],
                    "source_types": [],
                    "page_refs": [],
                    "evidence": [],
                    "confidence": 0.0,
                    "warnings": [],
                },
            )
            if dest["neutral_formula"] != neutral:
                warning = f"Conflicting neutral formula: {dest['neutral_formula']} vs {neutral}"
                if warning not in dest["warnings"]:
                    dest["warnings"].append(warning)
                current_signature = formula_atom_signature(record.get("raw_formula", ""))
                previous_signatures = {
                    formula_atom_signature(raw_formula)
                    for raw_formula in dest.get("raw_formulas", [])
                }
                previous_kinds = set(dest.get("formula_kinds", []))
                mh_plus_relabel_fix = (
                    formula_kind == "mh_plus"
                    and current_signature
                    and current_signature in previous_signatures
                    and "mh_plus" not in previous_kinds
                )
                if mh_plus_relabel_fix or score > float(dest.get("formula_score", 0)):
                    dest["neutral_formula"] = neutral
                    dest["formula_score"] = score
            else:
                dest["formula_score"] = max(float(dest.get("formula_score", 0)), score)
            for key, source_key in (
                ("raw_formulas", "raw_formula"),
                ("formula_kinds", "formula_kind"),
                ("source_types", "source_type"),
            ):
                value = str(record.get(source_key, "")).strip()
                if value and value not in dest[key]:
                    dest[key].append(value)
            if normalization_reason and normalization_reason not in dest["formula_normalization_reasons"]:
                dest["formula_normalization_reasons"].append(normalization_reason)
            for page_ref in record.get("page_refs", []):
                if page_ref not in dest["page_refs"]:
                    dest["page_refs"].append(page_ref)
            if record.get("evidence"):
                dest["evidence"].append(str(record["evidence"]))
            dest["confidence"] = max(dest["confidence"], float(record.get("confidence", 0)))
    return {
        "tables": tables,
        "records": sorted(by_id.values(), key=lambda row: compound_sort_key(row["compound_id"])),
        "notes": sorted(set(notes)),
    }


def cache_path_for_url(cache_dir: Optional[Path], prefix: str, url: str) -> Optional[Path]:
    if cache_dir is None:
        return None
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    return cache_dir / f"{prefix}_{digest}.json"


def nested_list(data: dict[str, Any], path: tuple[str, ...]) -> list[Any]:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return []
        current = current.get(key, [])
    return current if isinstance(current, list) else []


def fetch_json(
    url: str,
    *,
    timeout: int = DEFAULT_API_TIMEOUT,
    retries: int = DEFAULT_API_RETRIES,
    base_delay: float = DEFAULT_API_BASE_DELAY,
    cache_dir: Optional[Path] = None,
    cache_prefix: str = "api",
    cache_list_path: Optional[tuple[str, ...]] = None,
    cache_empty_lists: bool = True,
) -> Optional[dict[str, Any]]:
    cache_path = cache_path_for_url(cache_dir, cache_prefix, url)
    if cache_path and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cache_list_path and not cache_empty_lists and not nested_list(cached, cache_list_path):
                cache_path.unlink(missing_ok=True)
            else:
                return cached
        except json.JSONDecodeError:
            cache_path.unlink(missing_ok=True)

    request = urllib.request.Request(url, headers={"User-Agent": "MarkushFormulaWorkflow/0.1"})
    last_error = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            should_cache = True
            if cache_list_path and not cache_empty_lists and not nested_list(data, cache_list_path):
                should_cache = False
            if cache_path and should_cache:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return data
        except Exception as exc:
            last_error = exc
            sleep_for = base_delay * (2 ** attempt)
            time.sleep(sleep_for)
    print(f"Request failed after retries: {url} ({last_error})")
    return None


def fetch_text(
    url: str,
    *,
    timeout: int = DEFAULT_API_TIMEOUT,
    retries: int = DEFAULT_API_RETRIES,
    base_delay: float = DEFAULT_API_BASE_DELAY,
    cache_dir: Optional[Path] = None,
    cache_prefix: str = "text",
) -> str:
    cache_path = cache_path_for_url(cache_dir, cache_prefix, url)
    if cache_path and cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="replace")

    request = urllib.request.Request(url, headers={"User-Agent": "MarkushFormulaWorkflow/0.1"})
    last_error = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                text = response.read().decode("utf-8", errors="replace")
            if cache_path and text.strip():
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(text, encoding="utf-8")
            return text
        except Exception as exc:
            last_error = exc
            time.sleep(base_delay * (2 ** attempt))
    print(f"Text request failed after retries: {url} ({last_error})")
    return ""


def chembl_by_formula(
    formula: str,
    *,
    cache_dir: Optional[Path] = None,
    timeout: int = DEFAULT_API_TIMEOUT,
    retries: int = DEFAULT_API_RETRIES,
    base_delay: float = DEFAULT_API_BASE_DELAY,
) -> list[dict[str, Any]]:
    url = (
        "https://www.ebi.ac.uk/chembl/api/data/molecule.json?"
        "molecule_properties__full_molformula="
        + urllib.parse.quote(formula)
        + "&limit=50"
    )
    data = fetch_json(
        url,
        timeout=timeout,
        retries=retries,
        base_delay=base_delay,
        cache_dir=cache_dir,
        cache_prefix="chembl_formula",
        cache_list_path=("molecules",),
        cache_empty_lists=False,
    )
    return (data or {}).get("molecules", [])


def pubchem_fastformula(
    formula: str,
    max_records: int = 10,
    cache_dir: Optional[Path] = None,
    timeout: int = DEFAULT_API_TIMEOUT,
    retries: int = DEFAULT_API_RETRIES,
    base_delay: float = DEFAULT_API_BASE_DELAY,
) -> list[int]:
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastformula/"
        + urllib.parse.quote(formula)
        + f"/cids/JSON?MaxRecords={max_records}&MaxSeconds=20"
    )
    data = fetch_json(
        url,
        timeout=timeout,
        retries=retries,
        base_delay=base_delay,
        cache_dir=cache_dir,
        cache_prefix="pubchem_formula",
        cache_list_path=("IdentifierList", "CID"),
        cache_empty_lists=False,
    )
    if not data:
        return []
    return data.get("IdentifierList", {}).get("CID", [])


def pubchem_properties(
    cids: list[int],
    cache_dir: Optional[Path] = None,
    timeout: int = DEFAULT_API_TIMEOUT,
    retries: int = DEFAULT_API_RETRIES,
    base_delay: float = DEFAULT_API_BASE_DELAY,
) -> dict[int, dict[str, Any]]:
    if not cids:
        return {}
    joined = ",".join(str(cid) for cid in cids)
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
        + joined
        + "/property/MolecularFormula,CanonicalSMILES,IsomericSMILES,MolecularWeight/JSON"
    )
    data = fetch_json(
        url,
        timeout=timeout,
        retries=retries,
        base_delay=base_delay,
        cache_dir=cache_dir,
        cache_prefix="pubchem_props",
        cache_list_path=("PropertyTable", "Properties"),
        cache_empty_lists=False,
    )
    props: dict[int, dict[str, Any]] = {}
    for row in (data or {}).get("PropertyTable", {}).get("Properties", []):
        props[int(row["CID"])] = row
    return props


def pubchem_synonyms_by_name(
    name: str,
    cache_dir: Optional[Path] = None,
    timeout: int = DEFAULT_API_TIMEOUT,
    retries: int = DEFAULT_API_RETRIES,
    base_delay: float = DEFAULT_API_BASE_DELAY,
) -> dict[str, Any]:
    if not name:
        return {"cid": "", "synonyms": []}
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        + urllib.parse.quote(name)
        + "/synonyms/JSON"
    )
    data = fetch_json(
        url,
        timeout=timeout,
        retries=retries,
        base_delay=base_delay,
        cache_dir=cache_dir,
        cache_prefix="pubchem_synonyms",
        cache_list_path=("InformationList", "Information"),
        cache_empty_lists=False,
    )
    info = (data or {}).get("InformationList", {}).get("Information", [])
    if not info:
        return {"cid": "", "synonyms": []}
    return {
        "cid": str(info[0].get("CID", "")),
        "synonyms": [str(item) for item in info[0].get("Synonym", [])],
    }


def bindingdb_source_url(bindingdb_id: str) -> str:
    if not bindingdb_id.startswith("BDBM"):
        return ""
    monomer_id = bindingdb_id.replace("BDBM", "", 1)
    return (
        "https://www.bindingdb.org/rwd/bind/chemsearch/marvin/MolStructure.jsp"
        f"?google={bindingdb_id}&monomerid={monomer_id}"
    )


def bdbm_number(bindingdb_id: str) -> Optional[int]:
    match = re.fullmatch(r"BDBM(\d+)", bindingdb_id or "")
    return int(match.group(1)) if match else None


def append_warning(row: dict[str, Any], note: str) -> None:
    note = note.strip()
    if not note:
        return
    current = [
        item.strip()
        for item in str(row.get("warnings", "")).split("; ")
        if item.strip()
    ]
    if note not in current:
        current.append(note)
    row["warnings"] = "; ".join(current)


def initial_resolution_method(status: str) -> str:
    return {
        "ok_unique": "unique_formula",
        "ambiguous": "formula_ambiguous_first_candidate",
        "ok_pubchem_formula": "pubchem_formula",
        "not_found": "not_found",
    }.get(status, status or "unknown")


def parse_candidate_entries(row: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        candidates = json.loads(str(row.get("chembl_candidates_json", "[]")))
    except json.JSONDecodeError:
        return []
    return candidates if isinstance(candidates, list) else []


def candidate_bindingdb_numbers(candidates: list[dict[str, Any]]) -> str:
    numbers = [
        str(number)
        for number in (bdbm_number(str(candidate.get("bindingdb_id", ""))) for candidate in candidates)
        if number is not None
    ]
    return ";".join(numbers)


def candidate_sequence_deltas(candidates: list[dict[str, Any]], compound_id: str) -> str:
    if not str(compound_id).isdigit():
        return ""
    compound_number = int(str(compound_id))
    deltas: list[str] = []
    for candidate in candidates:
        bindingdb_id = str(candidate.get("bindingdb_id", ""))
        number = bdbm_number(bindingdb_id)
        if number is None:
            deltas.append("")
        else:
            deltas.append(f"{bindingdb_id}:{number - compound_number}")
    return ";".join(deltas)


def candidate_feature_summary(
    candidates: list[dict[str, Any]],
    compound_id: str,
    expected_bindingdb_id: str = "",
) -> str:
    parts: list[str] = []
    compound_number = int(compound_id) if str(compound_id).isdigit() else None
    for candidate in candidates:
        chembl_id = str(candidate.get("chembl_id", ""))
        bindingdb_id = str(candidate.get("bindingdb_id", ""))
        number = bdbm_number(bindingdb_id)
        delta = (
            f",delta={number - compound_number}"
            if number is not None and compound_number is not None
            else ""
        )
        expected = ",expected" if expected_bindingdb_id and bindingdb_id == expected_bindingdb_id else ""
        smiles = str(candidate.get("bindingdb_smiles", "")) or str(candidate.get("smiles", ""))
        parts.append(f"{chembl_id}/{bindingdb_id or '-'}{delta}{expected}: {smiles[:80]}")
    return " | ".join(parts)


def bindingdb_smiles_by_id(
    bindingdb_id: str,
    *,
    cache_dir: Optional[Path] = None,
    timeout: int = DEFAULT_API_TIMEOUT,
    retries: int = DEFAULT_API_RETRIES,
    base_delay: float = DEFAULT_API_BASE_DELAY,
) -> str:
    source_url = bindingdb_source_url(bindingdb_id)
    if not source_url:
        return ""
    # BindingDB pages are an optional source for presentation-identical SMILES.
    # Keep their retries bounded so a slow page cannot block the core lookup.
    page = fetch_text(
        source_url,
        timeout=min(timeout, 20),
        retries=min(retries, 2),
        base_delay=min(base_delay, 2.0),
        cache_dir=cache_dir,
        cache_prefix="bindingdb_html",
    )
    match = re.search(
        r"<b>\s*SMILES\s*</b>\s*<span[^>]*class=\"darkgray\"[^>]*>([^<]+)</span>",
        page,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    return html.unescape(match.group(1)).strip()


def lookup_one_record(
    record: dict[str, Any],
    include_pubchem: bool = True,
    cache_dir: Optional[Path] = None,
    request_timeout: int = DEFAULT_API_TIMEOUT,
    request_retries: int = DEFAULT_API_RETRIES,
    request_base_delay: float = DEFAULT_API_BASE_DELAY,
) -> dict[str, Any]:
    formula = record["neutral_formula"]
    chembl_hits = chembl_by_formula(
        formula,
        cache_dir=cache_dir,
        timeout=request_timeout,
        retries=request_retries,
        base_delay=request_base_delay,
    )
    candidate_entries: list[dict[str, str]] = []
    for hit in chembl_hits:
        chembl_id = hit.get("molecule_chembl_id", "")
        structures = hit.get("molecule_structures") or {}
        synonym_info = pubchem_synonyms_by_name(
            chembl_id,
            cache_dir=cache_dir,
            timeout=request_timeout,
            retries=request_retries,
            base_delay=request_base_delay,
        )
        bindingdb_ids = [syn for syn in synonym_info["synonyms"] if syn.startswith("BDBM")]
        bindingdb_id = bindingdb_ids[0] if bindingdb_ids else ""
        bindingdb_smiles = bindingdb_smiles_by_id(
            bindingdb_id,
            cache_dir=cache_dir,
            timeout=request_timeout,
            retries=request_retries,
            base_delay=request_base_delay,
        )
        candidate_entries.append(
            {
                "chembl_id": chembl_id,
                "smiles": structures.get("canonical_smiles", ""),
                "bindingdb_id": bindingdb_id,
                "bindingdb_source_url": bindingdb_source_url(bindingdb_id),
                "bindingdb_smiles": bindingdb_smiles,
                "pubchem_cid_by_chembl": synonym_info["cid"],
            }
        )
    first_entry = candidate_entries[0] if candidate_entries else {}
    selected_smiles = first_entry.get("bindingdb_smiles", "") or first_entry.get("smiles", "")
    first_chembl_id = first_entry.get("chembl_id", "")
    bindingdb_id = first_entry.get("bindingdb_id", "")
    pubchem_cids = (
        pubchem_fastformula(
            formula,
            max_records=10,
            cache_dir=cache_dir,
            timeout=request_timeout,
            retries=request_retries,
            base_delay=request_base_delay,
        )
        if include_pubchem
        else []
    )
    pubchem_props = (
        pubchem_properties(
            pubchem_cids[:10],
            cache_dir=cache_dir,
            timeout=request_timeout,
            retries=request_retries,
            base_delay=request_base_delay,
        )
        if include_pubchem
        else {}
    )
    exact_pubchem_cids = [
        cid for cid in pubchem_cids if pubchem_props.get(cid, {}).get("MolecularFormula") == formula
    ]
    first_pubchem_cid = exact_pubchem_cids[0] if exact_pubchem_cids else pubchem_cids[0] if pubchem_cids else 0
    first_pubchem_smiles = (
        pubchem_props.get(first_pubchem_cid, {}).get("IsomericSMILES", "") if first_pubchem_cid else ""
    )
    if not selected_smiles and first_pubchem_smiles:
        selected_smiles = first_pubchem_smiles
    if len(chembl_hits) == 1:
        status = "ok_unique"
    elif chembl_hits:
        status = "ambiguous"
    elif first_pubchem_smiles:
        status = "ok_pubchem_formula"
    else:
        status = "not_found"
    resolution_method = initial_resolution_method(status)
    resolution_confidence = "high" if status == "ok_unique" else "medium" if selected_smiles else "low"
    candidate_numbers = candidate_bindingdb_numbers(candidate_entries)
    candidate_deltas = candidate_sequence_deltas(candidate_entries, str(record.get("compound_id", "")))
    return {
        "compound_id": record["compound_id"],
        "neutral_formula": formula,
        "selected_smiles": selected_smiles,
        "lookup_status": status,
        "resolution_method": resolution_method,
        "resolution_confidence": resolution_confidence,
        "chembl_hit_count": len(chembl_hits),
        "first_chembl_id": first_chembl_id,
        "chembl_candidate_ids": ";".join(
            hit.get("molecule_chembl_id", "") for hit in chembl_hits
        ),
        "chembl_candidate_bindingdb_ids": ";".join(
            entry.get("bindingdb_id", "") for entry in candidate_entries
        ),
        "chembl_candidates_json": json.dumps(candidate_entries, ensure_ascii=False, separators=(",", ":")),
        "bindingdb_id": bindingdb_id,
        "bindingdb_source_url": bindingdb_source_url(bindingdb_id),
        "bindingdb_smiles": first_entry.get("bindingdb_smiles", ""),
        "pubchem_cid_by_chembl": first_entry.get("pubchem_cid_by_chembl", ""),
        "pubchem_cid_count_max10": len(pubchem_cids),
        "pubchem_cids": ";".join(str(cid) for cid in pubchem_cids[:10]),
        "first_pubchem_smiles": first_pubchem_smiles,
        "formula_source_types": ";".join(str(item) for item in record.get("source_types", [])),
        "formula_kinds": ";".join(str(item) for item in record.get("formula_kinds", [])),
        "formula_normalization_reason": ";".join(str(item) for item in record.get("formula_normalization_reasons", [])),
        "formula_raw_formulas": ";".join(str(item) for item in record.get("raw_formulas", [])),
        "formula_score": str(record.get("formula_score", "")),
        "formula_confidence": str(record.get("confidence", "")),
        "formula_source_pages": ";".join(record.get("page_refs", [])),
        "formula_evidence": " | ".join(record.get("evidence", [])[:3]),
        "activity_summary": record.get("activity_summary", ""),
        "primary_activity_assay": record.get("primary_activity_assay", ""),
        "primary_activity_target": record.get("primary_activity_target", ""),
        "primary_activity_endpoint": record.get("primary_activity_endpoint", ""),
        "primary_activity_value_text": record.get("primary_activity_value_text", ""),
        "primary_activity_value_nM": record.get("primary_activity_value_nM", ""),
        "primary_activity_selectivity_fold": record.get("primary_activity_selectivity_fold", ""),
        "activity_source_pages": ";".join(record.get("activity_source_pages", [])),
        "activity_evidence": " | ".join(record.get("activity_evidence", [])[:3]),
        "activities_json": record.get("activities_json", ""),
        "candidate_bindingdb_numbers": candidate_numbers,
        "candidate_sequence_deltas": candidate_deltas,
        "candidate_feature_summary": candidate_feature_summary(
            candidate_entries,
            str(record.get("compound_id", "")),
        ),
        "sequence_expected_bindingdb_id": "",
        "sequence_offset_support": "",
        "sequence_expected_pubchem_cid": "",
        "sequence_expected_formula": "",
        "sequence_expected_smiles": "",
        "referee_decision": "",
        "referee_selected_chembl_id": "",
        "referee_selected_bindingdb_id": "",
        "referee_confidence": "",
        "referee_distinguishing_features": "",
        "referee_rationale": "",
        "warnings": "; ".join(record.get("warnings", [])),
    }


def external_lookup(
    records: list[dict[str, Any]],
    delay: float,
    include_pubchem: bool = True,
    cache_dir: Optional[Path] = None,
    request_timeout: int = DEFAULT_API_TIMEOUT,
    request_retries: int = DEFAULT_API_RETRIES,
    request_base_delay: float = DEFAULT_API_BASE_DELAY,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            lookup_one_record(
                record,
                include_pubchem=include_pubchem,
                cache_dir=cache_dir,
                request_timeout=request_timeout,
                request_retries=request_retries,
                request_base_delay=request_base_delay,
            )
        )
        if delay:
            time.sleep(delay)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


LOOKUP_FIELDNAMES = [
    "compound_id",
    "neutral_formula",
    "selected_smiles",
    "lookup_status",
    "resolution_method",
    "resolution_confidence",
    "chembl_hit_count",
    "first_chembl_id",
    "chembl_candidate_ids",
    "chembl_candidate_bindingdb_ids",
    "chembl_candidates_json",
    "bindingdb_id",
    "bindingdb_source_url",
    "bindingdb_smiles",
    "pubchem_cid_by_chembl",
    "pubchem_cid_count_max10",
    "pubchem_cids",
    "first_pubchem_smiles",
    "formula_source_types",
    "formula_kinds",
    "formula_normalization_reason",
    "formula_raw_formulas",
    "formula_score",
    "formula_confidence",
    "formula_source_pages",
    "formula_evidence",
    "activity_summary",
    "primary_activity_assay",
    "primary_activity_target",
    "primary_activity_endpoint",
    "primary_activity_value_text",
    "primary_activity_value_nM",
    "primary_activity_selectivity_fold",
    "activity_source_pages",
    "activity_evidence",
    "activities_json",
    "candidate_bindingdb_numbers",
    "candidate_sequence_deltas",
    "candidate_feature_summary",
    "sequence_expected_bindingdb_id",
    "sequence_offset_support",
    "sequence_expected_pubchem_cid",
    "sequence_expected_formula",
    "sequence_expected_smiles",
    "referee_decision",
    "referee_selected_chembl_id",
    "referee_selected_bindingdb_id",
    "referee_confidence",
    "referee_distinguishing_features",
    "referee_rationale",
    "warnings",
]


def write_lookup_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOOKUP_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_bindingdb_sequence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    offsets: dict[int, int] = {}
    for row in rows:
        number = bdbm_number(row.get("bindingdb_id", ""))
        compound_id = str(row.get("compound_id", ""))
        if row.get("lookup_status") == "ok_unique" and number is not None and compound_id.isdigit():
            offset = number - int(compound_id)
            offsets[offset] = offsets.get(offset, 0) + 1
    if not offsets:
        return rows
    offset, support = max(offsets.items(), key=lambda item: item[1])
    if support < 5:
        return rows

    for row in rows:
        compound_id = str(row.get("compound_id", ""))
        if not compound_id.isdigit():
            continue
        expected_bdbm = f"BDBM{offset + int(compound_id)}"
        candidates = parse_candidate_entries(row)
        row["sequence_expected_bindingdb_id"] = expected_bdbm
        row["sequence_offset_support"] = str(support)
        row["candidate_feature_summary"] = candidate_feature_summary(
            candidates,
            compound_id,
            expected_bdbm,
        )
        if row.get("lookup_status") != "ambiguous":
            continue
        match = next(
            (candidate for candidate in candidates if candidate.get("bindingdb_id") == expected_bdbm),
            None,
        )
        if not match:
            continue
        if row.get("bindingdb_id") != expected_bdbm:
            row["selected_smiles"] = match.get("bindingdb_smiles", "") or match.get("smiles", "")
            row["first_chembl_id"] = match.get("chembl_id", "")
            row["bindingdb_id"] = expected_bdbm
            row["bindingdb_source_url"] = match.get(
                "bindingdb_source_url", bindingdb_source_url(expected_bdbm)
            )
            row["bindingdb_smiles"] = match.get("bindingdb_smiles", "")
            row["pubchem_cid_by_chembl"] = match.get("pubchem_cid_by_chembl", "")
            row["lookup_status"] = "resolved_bindingdb_sequence"
            row["resolution_method"] = "bindingdb_sequence"
            row["resolution_confidence"] = "high"
            note = (
                "Resolved formula ambiguity using inferred BindingDB ID sequence "
                f"(offset={offset}, support={support})"
            )
            append_warning(row, note)
    return rows


def repair_rows_with_expected_bindingdb(
    rows: list[dict[str, Any]],
    *,
    cache_dir: Optional[Path] = None,
    request_timeout: int = DEFAULT_API_TIMEOUT,
    request_retries: int = DEFAULT_API_RETRIES,
    request_base_delay: float = DEFAULT_API_BASE_DELAY,
) -> list[dict[str, Any]]:
    for row in rows:
        lookup_status = str(row.get("lookup_status", ""))
        if lookup_status not in {"not_found", "ambiguous"}:
            continue
        expected_bdbm = str(row.get("sequence_expected_bindingdb_id", "")).strip()
        if not expected_bdbm or expected_bdbm == str(row.get("bindingdb_id", "")).strip():
            continue
        try:
            support = int(row.get("sequence_offset_support") or 0)
        except (TypeError, ValueError):
            support = 0
        if support < 5:
            continue

        synonym_info = pubchem_synonyms_by_name(
            expected_bdbm,
            cache_dir=cache_dir,
            timeout=request_timeout,
            retries=request_retries,
            base_delay=request_base_delay,
        )
        pubchem_cid = str(synonym_info.get("cid", ""))
        pubchem_props: dict[str, Any] = {}
        if pubchem_cid.isdigit():
            props_by_cid = pubchem_properties(
                [int(pubchem_cid)],
                cache_dir=cache_dir,
                timeout=request_timeout,
                retries=request_retries,
                base_delay=request_base_delay,
            )
            pubchem_props = props_by_cid.get(int(pubchem_cid), {})
        expected_formula = str(pubchem_props.get("MolecularFormula", ""))
        expected_smiles = str(
            pubchem_props.get("IsomericSMILES", "")
            or pubchem_props.get("CanonicalSMILES", "")
        )
        expected_bindingdb_smiles = bindingdb_smiles_by_id(
            expected_bdbm,
            cache_dir=cache_dir,
            timeout=request_timeout,
            retries=request_retries,
            base_delay=request_base_delay,
        )
        selected_smiles = expected_bindingdb_smiles or expected_smiles
        if not expected_formula and not selected_smiles:
            continue

        row["sequence_expected_pubchem_cid"] = pubchem_cid
        row["sequence_expected_formula"] = expected_formula
        row["sequence_expected_smiles"] = selected_smiles
        if expected_formula:
            row["neutral_formula"] = expected_formula
        if selected_smiles:
            row["selected_smiles"] = selected_smiles
        chembl_synonyms = [
            str(item)
            for item in synonym_info.get("synonyms", [])
            if str(item).startswith("CHEMBL")
        ]
        row["first_chembl_id"] = chembl_synonyms[0] if chembl_synonyms else row.get("first_chembl_id", "")
        row["bindingdb_id"] = expected_bdbm
        row["bindingdb_source_url"] = bindingdb_source_url(expected_bdbm)
        row["bindingdb_smiles"] = expected_bindingdb_smiles
        row["pubchem_cid_by_chembl"] = pubchem_cid
        row["lookup_status"] = "resolved_bindingdb_sequence_mismatch"
        row["resolution_method"] = "bindingdb_sequence_expected_id"
        row["resolution_confidence"] = "high"
        append_warning(
            row,
            (
                "Corrected formula/SMILES using inferred BindingDB ID sequence "
                f"(expected={expected_bdbm}, support={support})"
            ),
        )
    return rows


def trim_for_prompt(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    half = max(1, limit // 2)
    return value[:half].rstrip() + "\n...[truncated]...\n" + value[-half:].lstrip()


def referee_cache_key(row: dict[str, Any]) -> str:
    payload = {
        "compound_id": row.get("compound_id", ""),
        "neutral_formula": row.get("neutral_formula", ""),
        "lookup_status": row.get("lookup_status", ""),
        "candidates": row.get("chembl_candidates_json", ""),
        "sequence_expected_bindingdb_id": row.get("sequence_expected_bindingdb_id", ""),
        "formula_evidence": row.get("formula_evidence", ""),
        "warnings": row.get("warnings", ""),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"{row.get('compound_id', 'compound')}_{digest}"


def ambiguity_referee_prompt(row: dict[str, Any], context_chars: int) -> str:
    candidates = parse_candidate_entries(row)
    compound_id = str(row.get("compound_id", ""))
    compound_number = int(compound_id) if compound_id.isdigit() else None
    candidate_lines: list[str] = []
    for index, candidate in enumerate(candidates, start=1):
        bindingdb_id = str(candidate.get("bindingdb_id", ""))
        number = bdbm_number(bindingdb_id)
        delta = (
            str(number - compound_number)
            if number is not None and compound_number is not None
            else ""
        )
        smiles = str(candidate.get("bindingdb_smiles", "")) or str(candidate.get("smiles", ""))
        candidate_lines.append(
            "\n".join(
                [
                    f"Candidate {index}",
                    f"- chembl_id: {candidate.get('chembl_id', '')}",
                    f"- bindingdb_id: {bindingdb_id}",
                    f"- bindingdb_number: {number if number is not None else ''}",
                    f"- bindingdb_minus_compound_id: {delta}",
                    f"- pubchem_cid_by_chembl: {candidate.get('pubchem_cid_by_chembl', '')}",
                    f"- smiles: {smiles}",
                ]
            )
        )

    evidence_budget = max(500, context_chars // 2)
    return f"""\
You are a chemistry data-curation referee. Resolve one ambiguous formula-to-SMILES assignment.

Return only JSON matching the schema. Do not invent candidates or call external tools.

Decision rules:
- All candidates already share the same molecular formula, so formula alone is not a distinguishing feature.
- Prefer a candidate whose BindingDB ID matches a well-supported inferred BindingDB sequence when sequence_offset_support is high.
- Use formula provenance, raw formula kind, characterization/table evidence, warnings, and candidate metadata as supporting facts.
- If the current selected candidate is still the best supported choice, return decision "keep_current".
- If a different listed candidate is best supported, return decision "select_candidate".
- If the evidence does not distinguish candidates, return decision "unresolved" with low confidence.
- selected_chembl_id and selected_bindingdb_id must be empty for unresolved, otherwise they must come from one listed candidate or the current selection.

Compound:
- compound_id: {compound_id}
- neutral_formula: {row.get('neutral_formula', '')}
- current_lookup_status: {row.get('lookup_status', '')}
- current_selected_chembl_id: {row.get('first_chembl_id', '')}
- current_selected_bindingdb_id: {row.get('bindingdb_id', '')}
- sequence_expected_bindingdb_id: {row.get('sequence_expected_bindingdb_id', '')}
- sequence_offset_support: {row.get('sequence_offset_support', '')}

Formula extraction provenance:
- formula_source_types: {row.get('formula_source_types', '')}
- formula_kinds: {row.get('formula_kinds', '')}
- formula_normalization_reason: {row.get('formula_normalization_reason', '')}
- formula_raw_formulas: {row.get('formula_raw_formulas', '')}
- formula_score: {row.get('formula_score', '')}
- formula_confidence: {row.get('formula_confidence', '')}
- formula_source_pages: {row.get('formula_source_pages', '')}
- formula_evidence: {trim_for_prompt(row.get('formula_evidence', ''), evidence_budget)}
- warnings: {trim_for_prompt(row.get('warnings', ''), max(500, context_chars - evidence_budget))}

Candidate feature summary:
{trim_for_prompt(row.get('candidate_feature_summary', ''), context_chars)}

Candidate records:
{chr(10).join(candidate_lines) if candidate_lines else "- none"}
"""


def find_candidate_for_decision(
    row: dict[str, Any],
    selected_chembl_id: str,
    selected_bindingdb_id: str,
) -> Optional[dict[str, Any]]:
    candidates = parse_candidate_entries(row)
    selected_chembl_id = selected_chembl_id.strip()
    selected_bindingdb_id = selected_bindingdb_id.strip()
    if selected_bindingdb_id:
        for candidate in candidates:
            if str(candidate.get("bindingdb_id", "")) == selected_bindingdb_id:
                return candidate
    if selected_chembl_id:
        for candidate in candidates:
            if str(candidate.get("chembl_id", "")) == selected_chembl_id:
                return candidate
    return None


def apply_referee_decision(row: dict[str, Any], decision: dict[str, Any]) -> None:
    row["referee_decision"] = str(decision.get("decision", ""))
    row["referee_selected_chembl_id"] = str(decision.get("selected_chembl_id", ""))
    row["referee_selected_bindingdb_id"] = str(decision.get("selected_bindingdb_id", ""))
    confidence_value = decision.get("confidence", "")
    try:
        confidence_float = float(confidence_value)
        row["referee_confidence"] = f"{confidence_float:.2f}"
    except (TypeError, ValueError):
        confidence_float = 0.0
        row["referee_confidence"] = str(confidence_value)
    row["referee_distinguishing_features"] = "; ".join(
        str(item) for item in decision.get("distinguishing_features", []) if str(item).strip()
    )
    row["referee_rationale"] = str(decision.get("rationale", ""))

    referee_decision = row["referee_decision"]
    if referee_decision == "unresolved":
        if row.get("lookup_status") == "ambiguous":
            row["resolution_method"] = "unresolved_ambiguous"
            row["resolution_confidence"] = row.get("referee_confidence", "")
        append_warning(row, "Formula ambiguity left unresolved by LLM referee")
        return

    selected_candidate = find_candidate_for_decision(
        row,
        row["referee_selected_chembl_id"],
        row["referee_selected_bindingdb_id"],
    )
    if selected_candidate is None:
        if referee_decision == "keep_current":
            current_bindingdb_id = str(row.get("bindingdb_id", ""))
            current_chembl_id = str(row.get("first_chembl_id", ""))
            if current_bindingdb_id or current_chembl_id:
                row["referee_selected_chembl_id"] = current_chembl_id
                row["referee_selected_bindingdb_id"] = current_bindingdb_id
        else:
            append_warning(row, "LLM referee returned a candidate ID that was not in the candidate list")
        return

    selected_bindingdb_id = str(selected_candidate.get("bindingdb_id", ""))
    sequence_expected = str(row.get("sequence_expected_bindingdb_id", ""))
    if (
        row.get("lookup_status") == "resolved_bindingdb_sequence"
        and sequence_expected
        and selected_bindingdb_id
        and selected_bindingdb_id != sequence_expected
        and confidence_float < 0.95
    ):
        append_warning(
            row,
            "LLM referee suggested a different candidate, but the BindingDB sequence rule was retained",
        )
        return

    current_bindingdb_id = str(row.get("bindingdb_id", ""))
    current_chembl_id = str(row.get("first_chembl_id", ""))
    selected_chembl_id = str(selected_candidate.get("chembl_id", ""))
    selection_changed = (
        selected_bindingdb_id != current_bindingdb_id
        or selected_chembl_id != current_chembl_id
    )
    if referee_decision == "select_candidate" or selection_changed:
        row["selected_smiles"] = selected_candidate.get("bindingdb_smiles", "") or selected_candidate.get("smiles", "")
        row["first_chembl_id"] = selected_chembl_id
        row["bindingdb_id"] = selected_bindingdb_id
        row["bindingdb_source_url"] = selected_candidate.get(
            "bindingdb_source_url",
            bindingdb_source_url(selected_bindingdb_id),
        )
        row["bindingdb_smiles"] = selected_candidate.get("bindingdb_smiles", "")
        row["pubchem_cid_by_chembl"] = selected_candidate.get("pubchem_cid_by_chembl", "")

    if row.get("lookup_status") == "ambiguous" and confidence_float >= 0.70:
        row["lookup_status"] = "resolved_llm_referee"
    row["resolution_method"] = (
        "bindingdb_sequence+llm_referee"
        if row.get("lookup_status") == "resolved_bindingdb_sequence"
        else "llm_referee"
    )
    row["resolution_confidence"] = row.get("referee_confidence", "")
    append_warning(row, "Formula ambiguity reviewed by LLM referee")


def run_ambiguity_referee(
    rows: list[dict[str, Any]],
    *,
    client: Any,
    model: str,
    cache_path: Path,
    enabled: bool,
    statuses: list[str],
    context_chars_per_row: int,
    max_output_tokens: int,
    reuse_cache: bool,
    force: bool,
) -> list[dict[str, Any]]:
    if not enabled:
        return rows
    status_set = {str(status) for status in statuses}
    cache: dict[str, Any] = {"decisions": {}}
    if cache_path.exists() and reuse_cache and not force:
        try:
            loaded = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                cache = loaded
        except json.JSONDecodeError:
            cache = {"decisions": {}}
    decisions = cache.setdefault("decisions", {})

    for row in rows:
        if str(row.get("lookup_status", "")) not in status_set:
            continue
        if int(row.get("chembl_hit_count") or 0) <= 1:
            continue
        key = referee_cache_key(row)
        if key in decisions and reuse_cache and not force:
            decision = decisions[key]
            print(f"Using cached ambiguity referee for compound {row.get('compound_id')}")
        else:
            print(f"Calling {model} ambiguity referee for compound {row.get('compound_id')}")
            decision = call_json_agent(
                client,
                model,
                ambiguity_referee_prompt(row, context_chars_per_row),
                "ambiguity_referee",
                AMBIGUITY_REFEREE_SCHEMA,
                max_output_tokens,
            )
            decisions[key] = decision
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(cache, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        apply_referee_decision(row, decision)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rows


def lookup_and_write_csv(
    path: Path,
    records: list[dict[str, Any]],
    delay: float,
    include_pubchem: bool = True,
    cache_dir: Optional[Path] = None,
    request_timeout: int = DEFAULT_API_TIMEOUT,
    request_retries: int = DEFAULT_API_RETRIES,
    request_base_delay: float = DEFAULT_API_BASE_DELAY,
) -> list[dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for record in records:
        row = lookup_one_record(
            record,
            include_pubchem=include_pubchem,
            cache_dir=cache_dir,
            request_timeout=request_timeout,
            request_retries=request_retries,
            request_base_delay=request_base_delay,
        )
        rows.append(row)
        print(
            record["compound_id"],
            record["neutral_formula"],
            row["lookup_status"],
            row["first_chembl_id"],
        )
        if delay:
            time.sleep(delay)
    rows = resolve_bindingdb_sequence(rows)
    write_lookup_rows_csv(path, rows)
    return rows


def write_final_smiles_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "Compound",
        "Neutral_formula",
        "SMILES",
        "BindingDB_ID",
        "Source_URL",
        "Activity_Assay",
        "Activity_Target",
        "Activity_Endpoint",
        "Activity_Value",
        "Activity_nM",
        "Selectivity_fold",
        "Activity_Evidence",
        "Note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            notes = []
            if row["lookup_status"] == "ambiguous":
                notes.append(
                    "Ambiguous formula lookup; selected first ChEMBL candidate from "
                    + row.get("chembl_candidate_ids", "")
                )
            elif row["lookup_status"] == "resolved_bindingdb_sequence":
                notes.append("Formula ambiguity resolved using inferred BindingDB ID sequence")
            elif row["lookup_status"] == "resolved_bindingdb_sequence_mismatch":
                notes.append("Formula/SMILES corrected using inferred BindingDB ID sequence")
            elif row["lookup_status"] == "resolved_llm_referee":
                notes.append("Formula ambiguity resolved by LLM referee")
            elif row["lookup_status"] == "ok_pubchem_formula":
                notes.append("No ChEMBL exact-formula hit; PubChem formula fallback used")
            elif row["lookup_status"] == "not_found":
                notes.append("No ChEMBL exact-formula hit")
            if row.get("warnings"):
                warning_items = [
                    item
                    for item in row["warnings"].split("; ")
                    if not (
                        row["lookup_status"] == "resolved_bindingdb_sequence"
                        and item.startswith("Resolved formula ambiguity")
                    )
                ]
                notes.extend(warning_items)
            writer.writerow(
                {
                    "Compound": row["compound_id"],
                    "Neutral_formula": row["neutral_formula"],
                    "SMILES": row["selected_smiles"],
                    "BindingDB_ID": row.get("bindingdb_id", ""),
                    "Source_URL": row.get("bindingdb_source_url", ""),
                    "Activity_Assay": row.get("primary_activity_assay", ""),
                    "Activity_Target": row.get("primary_activity_target", ""),
                    "Activity_Endpoint": row.get("primary_activity_endpoint", ""),
                    "Activity_Value": row.get("primary_activity_value_text", ""),
                    "Activity_nM": row.get("primary_activity_value_nM", ""),
                    "Selectivity_fold": row.get("primary_activity_selectivity_fold", ""),
                    "Activity_Evidence": row.get("activity_evidence", ""),
                    "Note": "; ".join(note for note in notes if note),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract formulas from PDFs and map to SMILES.")
    parser.add_argument("--config", default="markush_config.json")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--pdf", action="append", default=[])
    parser.add_argument("--pages-per-chunk", type=int, default=None)
    parser.add_argument("--max-chunk-chars", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--delay", type=float, default=None)
    parser.add_argument("--api-timeout", type=int, default=None)
    parser.add_argument("--api-retries", type=int, default=None)
    parser.add_argument("--api-base-delay", type=float, default=None)
    parser.add_argument("--include-pubchem-formula", action="store_true", default=None)
    parser.add_argument("--skip-pubchem", action="store_true", default=None)
    parser.add_argument("--force", action="store_true", default=None)
    parser.add_argument("--final-csv", default=None)
    parser.add_argument("--detailed-csv", default=None)
    parser.add_argument("--pdf-pages-cache", default=None)
    parser.add_argument("--page-chunks-cache", default=None)
    parser.add_argument("--compound-scope-cache", default=None)
    parser.add_argument("--target-compound-id", action="append", default=[])
    parser.add_argument("--exclude-compound-id", action="append", default=[])
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    config_base_dir = config_path.resolve().parent if config_path.exists() else Path.cwd()
    config = deep_merge(default_workflow_config(), load_workflow_config(args.config))
    formula_config = deep_merge(
        default_formula_workflow_config(),
        config.get("formula_workflow", {}),
    )

    model = arg_or_config(args.model, formula_config, "llm", "model") or config["models"]["primary"]
    config["models"]["primary"] = model

    outdir = resolve_path(
        arg_or_config(args.outdir, formula_config, "outdir"),
        config_base_dir,
    ).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    configured_pdfs = args.pdf if args.pdf else formula_config.get("pdfs", [])
    pdf_paths = [resolve_path(str(path), config_base_dir).resolve() for path in configured_pdfs]
    if not pdf_paths:
        pdf_paths = sorted(config_base_dir.glob("*.pdf"))

    pages_per_chunk = int(arg_or_config(args.pages_per_chunk, formula_config, "llm", "pages_per_chunk"))
    max_chunk_chars = int(arg_or_config(args.max_chunk_chars, formula_config, "llm", "max_chunk_chars"))
    max_output_tokens = int(arg_or_config(args.max_output_tokens, formula_config, "llm", "max_output_tokens"))
    delay = float(arg_or_config(args.delay, formula_config, "request", "delay"))
    api_timeout = int(arg_or_config(args.api_timeout, formula_config, "request", "api_timeout"))
    api_retries = int(arg_or_config(args.api_retries, formula_config, "request", "api_retries"))
    api_base_delay = float(arg_or_config(args.api_base_delay, formula_config, "request", "api_base_delay"))
    force = bool(arg_or_config(args.force, formula_config, "cache", "force"))

    include_pubchem_formula = bool(
        formula_config.get("request", {}).get("include_pubchem_formula", False)
    )
    if args.include_pubchem_formula:
        include_pubchem_formula = True
    if args.skip_pubchem:
        include_pubchem_formula = False

    output_config = formula_config["output"]
    pdf_pages_cache_path = resolve_output_path(
        args.pdf_pages_cache or output_config["pdf_pages_json"],
        outdir,
    )
    page_chunks_cache_path = resolve_output_path(
        args.page_chunks_cache or output_config["pdf_page_chunks_json"],
        outdir,
    )
    compound_scope_cache_path = resolve_output_path(
        args.compound_scope_cache or output_config["compound_scope_json"],
        outdir,
    )
    formula_records_path = resolve_output_path(output_config["formula_records_json"], outdir)
    activity_records_path = resolve_output_path(
        output_config.get("activity_records_json", "activity_records.json"),
        outdir,
    )
    detailed_csv_path = resolve_output_path(
        args.detailed_csv or output_config["detailed_csv"],
        outdir,
    )
    ambiguity_referee_cache_path = resolve_output_path(
        output_config.get("ambiguity_referee_json", "ambiguity_referee.json"),
        outdir,
    )

    chunks = build_or_load_page_chunks(
        pdf_paths,
        pages_per_chunk,
        max_chunk_chars,
        pdf_pages_cache_path,
        page_chunks_cache_path,
        bool(formula_config.get("cache", {}).get("reuse_pdf_pages", True)),
        bool(formula_config.get("cache", {}).get("reuse_page_chunks", True)),
        force,
    )

    client = require_openai_client(config["api"])
    scope_config = formula_config.get("compound_scope", {})
    manual_target_ids = list(scope_config.get("target_compound_ids", []) or []) + list(args.target_compound_id)
    manual_excluded_ids = list(scope_config.get("excluded_compound_ids", []) or []) + list(args.exclude_compound_id)
    compound_scope = discover_compound_scope(
        client,
        model,
        chunks,
        compound_scope_cache_path,
        max_output_tokens,
        int(scope_config.get("context_max_chars", 60000)),
        [str(item) for item in manual_target_ids],
        [str(item) for item in manual_excluded_ids],
        bool(formula_config.get("cache", {}).get("reuse_compound_scope", True)),
        force,
    )
    final_target_ids = scope_entries_to_ids(compound_scope.get("final_compounds", []))
    intermediate_ids = scope_entries_to_ids(compound_scope.get("intermediates", []))
    excluded_ids = sorted(
        set(intermediate_ids + scope_entries_to_ids(compound_scope.get("excluded_compounds", []))),
        key=compound_sort_key,
    )
    if not final_target_ids:
        raise RuntimeError(
            "Compound scope discovery found no final target compound IDs. "
            "Add formula_workflow.compound_scope.target_compound_ids to the config "
            "or inspect compound_scope.json."
        )
    print(f"Final target compound IDs: {compound_id_summary(final_target_ids)}")
    print(f"Excluded/intermediate IDs: {compound_id_summary(excluded_ids)}")

    scope_digest = scope_hash(compound_scope)
    activity_results: list[dict[str, Any]] = []
    merged_activity: dict[str, Any] = {"tables": [], "records": [], "notes": []}
    activity_config = formula_config.get("activity", {})
    activity_enabled = bool(activity_config.get("enabled", True))
    primary_assay_keywords = [str(item) for item in activity_config.get("primary_assay_keywords", [])]
    reuse_activity_extract = bool(formula_config.get("cache", {}).get("reuse_activity_extract", True))
    if activity_enabled:
        for chunk in chunks:
            cache_path = outdir / f"activity_extract_{scope_digest}_{chunk['chunk_id']}.json"
            if cache_path.exists() and reuse_activity_extract and not force:
                result = json.loads(cache_path.read_text(encoding="utf-8"))
                print(f"Using cached {cache_path.name}")
            else:
                print(f"Calling {model} for activity {chunk['chunk_id']}")
                result = call_json_agent(
                    client,
                    model,
                    activity_prompt(
                        chunk,
                        compound_scope,
                        primary_assay_keywords,
                    ),
                    f"activity_extract_{chunk['chunk_id']}",
                    ACTIVITY_SCHEMA,
                    max_output_tokens,
                )
                cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            activity_results.append(result)
        merged_activity = merge_activity_records(
            activity_results,
            final_target_ids,
            excluded_ids,
            primary_assay_keywords,
        )
    activity_records_path.write_text(
        json.dumps(merged_activity, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    chunk_results: list[dict[str, Any]] = []
    reuse_formula_extract = bool(formula_config.get("cache", {}).get("reuse_formula_extract", True))
    for chunk in chunks:
        cache_path = outdir / f"formula_extract_{scope_digest}_{chunk['chunk_id']}.json"
        if cache_path.exists() and reuse_formula_extract and not force:
            result = json.loads(cache_path.read_text(encoding="utf-8"))
            print(f"Using cached {cache_path.name}")
        else:
            print(f"Calling {model} for {chunk['chunk_id']}")
            result = call_json_agent(
                client,
                model,
                formula_prompt(chunk, compound_scope),
                f"formula_extract_{chunk['chunk_id']}",
                FORMULA_SCHEMA,
                max_output_tokens,
            )
            cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        chunk_results.append(result)

    merged = merge_formula_records(chunk_results, final_target_ids, excluded_ids)
    merged = attach_activity_records(merged, merged_activity)
    found_ids = {normalize_compound_id(record.get("compound_id", "")) for record in merged.get("records", [])}
    missing_ids = [
        compound_id
        for compound_id in final_target_ids
        if normalize_compound_id(compound_id) not in found_ids
    ]
    if missing_ids:
        missing_hash = hashlib.sha256(
            json.dumps(missing_ids, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        missing_result = extract_missing_formula_records(
            client,
            model,
            chunks,
            compound_scope,
            missing_ids,
            outdir / f"formula_extract_{scope_digest}_missing_v6_{missing_hash}.json",
            max_output_tokens,
            reuse_formula_extract,
            force,
        )
        if missing_result:
            chunk_results.append(missing_result)
            merged = merge_formula_records(chunk_results, final_target_ids, excluded_ids)
            merged = attach_activity_records(merged, merged_activity)

    formula_records_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    final_csv_path = final_csv_path_from_config(
        args.final_csv or output_config.get("final_csv"),
        outdir,
        client,
        model,
        pdf_paths,
        merged,
        max_output_tokens,
        force,
    )
    api_cache_dir = outdir / "api_cache" if formula_config.get("cache", {}).get("reuse_external_api", True) else None
    lookup_rows = lookup_and_write_csv(
        detailed_csv_path,
        merged["records"],
        delay,
        include_pubchem=include_pubchem_formula,
        cache_dir=api_cache_dir,
        request_timeout=api_timeout,
        request_retries=api_retries,
        request_base_delay=api_base_delay,
    )
    lookup_rows = repair_rows_with_expected_bindingdb(
        lookup_rows,
        cache_dir=api_cache_dir,
        request_timeout=api_timeout,
        request_retries=api_retries,
        request_base_delay=api_base_delay,
    )
    not_found_ids = [
        str(row.get("compound_id", ""))
        for row in lookup_rows
        if row.get("lookup_status") == "not_found"
    ]
    if not_found_ids:
        fallback_result = formula_records_from_snippets(
            not_found_ids,
            focused_formula_snippets(not_found_ids, chunks),
        )
        if fallback_result.get("records"):
            print(
                "Applying deterministic LCMS fallback for unresolved formula IDs: "
                + compound_id_summary(not_found_ids)
            )
            chunk_results.append(fallback_result)
            merged = merge_formula_records(chunk_results, final_target_ids, excluded_ids)
            merged = attach_activity_records(merged, merged_activity)
            formula_records_path.write_text(
                json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            lookup_rows = lookup_and_write_csv(
                detailed_csv_path,
                merged["records"],
                delay,
                include_pubchem=include_pubchem_formula,
                cache_dir=api_cache_dir,
                request_timeout=api_timeout,
                request_retries=api_retries,
                request_base_delay=api_base_delay,
            )
            lookup_rows = repair_rows_with_expected_bindingdb(
                lookup_rows,
                cache_dir=api_cache_dir,
                request_timeout=api_timeout,
                request_retries=api_retries,
                request_base_delay=api_base_delay,
            )
    referee_config = formula_config.get("referee", {})
    lookup_rows = run_ambiguity_referee(
        lookup_rows,
        client=client,
        model=model,
        cache_path=ambiguity_referee_cache_path,
        enabled=bool(referee_config.get("enabled", True)),
        statuses=[str(item) for item in referee_config.get("statuses", ["ambiguous"])],
        context_chars_per_row=int(referee_config.get("context_chars_per_row", 4000)),
        max_output_tokens=int(referee_config.get("max_output_tokens", 2000)),
        reuse_cache=bool(referee_config.get("reuse_cache", True)),
        force=force,
    )
    write_lookup_rows_csv(detailed_csv_path, lookup_rows)
    write_final_smiles_csv(
        final_csv_path,
        lookup_rows,
    )

    print(f"Done. Outputs written to {outdir}")
    print(f"- {formula_records_path}")
    print(f"- {activity_records_path}")
    print(f"- {detailed_csv_path}")
    if ambiguity_referee_cache_path.exists():
        print(f"- {ambiguity_referee_cache_path}")
    print(f"- {final_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
