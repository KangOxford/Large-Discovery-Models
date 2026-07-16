"""JSON Schema for the six LLM-emitted blocks.

The schema mirrors the design doc (§6.2). It is exposed as a single
string constant ``BLOCKS_SCHEMA_JSON`` and is loaded into a
:class:`jsonschema.Draft202012Validator` lazily by
:func:`get_validator` — no module-level schema compilation, so the
import is cheap.

Note on ``oneOf``:
    ``oneOf`` rejects objects that match more than one branch. Our
    blocks have a ``"type"`` field with a fixed ``const`` value, so
    exactly one branch matches. This gives the right semantics: an
    object with ``"type": "noop"`` matches *only* the NoopBlock branch.

Phase filtering (``PHASE_A_ALLOWED`` / ``PHASE_B_ALLOWED``) is done
in code, not in the schema, because the JSON schema can't express
"this object is acceptable only in this phase".
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict

from jsonschema import Draft202012Validator


# ---------------------------------------------------------------------------
# Schema string
# ---------------------------------------------------------------------------


BLOCKS_SCHEMA_JSON: str = r"""
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "advisor_blocks.v1.json",
  "definitions": {
    "ReviewBOBlock": {
      "type": "object",
      "required": ["type", "rationale", "decisions"],
      "properties": {
        "type": {"const": "review_bo"},
        "rationale": {"type": "string", "maxLength": 600},
        "decisions": {
          "type": "object",
          "minProperties": 0,
          "additionalProperties": {
            "type": "string",
            "pattern": "^(ok|skip|override:.+)$"
          }
        }
      },
      "additionalProperties": false
    },
    "ProposeBlock": {
      "type": "object",
      "required": ["type", "rationale", "smiles"],
      "properties": {
        "type": {"const": "propose"},
        "rationale": {"type": "string", "maxLength": 400},
        "smiles": {
          "type": "array", "minItems": 1, "maxItems": 10,
          "items": {"type": "string", "pattern": "^[A-Za-z0-9@+\\-\\[\\]()=#$%/.]+$"}
        },
        "rationale_per_mol": {
          "type": "object",
          "additionalProperties": {"type": "string"}
        }
      },
      "additionalProperties": false
    },
    "RejectBlock": {
      "type": "object",
      "required": ["type", "rationale", "targets", "reason"],
      "properties": {
        "type": {"const": "reject"},
        "rationale": {"type": "string", "maxLength": 200},
        "targets": {
          "type": "array", "minItems": 1, "maxItems": 50,
          "items": {"type": "string"}
        },
        "reason": {
          "enum": [
            "too_similar_to_history",
            "likely_toxic",
            "synthetically_infeasible",
            "out_of_scope_pharmacophore",
            "no_signal_for_target"
          ]
        }
      },
      "additionalProperties": false
    },
    "AnalogBlock": {
      "type": "object",
      "required": ["type", "rationale", "seeds"],
      "properties": {
        "type": {"const": "analog"},
        "rationale": {"type": "string", "maxLength": 400},
        "seeds": {
          "type": "array", "minItems": 1, "maxItems": 10,
          "items": {"type": "string"}
        },
        "generator_hint": {
          "enum": ["conservative", "aggressive", "scaffold_hop", null]
        },
        "n_per_seed": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
        "reasyn_config_override": {
          "type": ["object", "null"],
          "properties": {
            "search_width": {"type": "integer", "minimum": 1, "maximum": 64},
            "num_cycles": {"type": "integer", "minimum": 1, "maximum": 32},
            "num_editflow_samples": {"type": "integer", "minimum": 1, "maximum": 500},
            "time_limit": {"type": "integer", "minimum": 10, "maximum": 3600}
          },
          "additionalProperties": false
        }
      },
      "additionalProperties": false
    },
    "ReviewAnalogsBlock": {
      "type": "object",
      "required": ["type", "rationale", "decisions"],
      "properties": {
        "type": {"const": "review_analogs"},
        "rationale": {"type": "string", "maxLength": 400},
        "decisions": {
          "type": "object",
          "minProperties": 1,
          "additionalProperties": {
            "type": "string",
            "enum": ["keep", "reject", "rescore_with_different_params"]
          }
        }
      },
      "additionalProperties": false
    },
    "NoopBlock": {
      "type": "object",
      "required": ["type", "rationale"],
      "properties": {
        "type": {"const": "noop"},
        "rationale": {"type": "string", "maxLength": 200}
      },
      "additionalProperties": false
    }
  },
  "oneOf": [
    {"$ref": "#/definitions/ReviewBOBlock"},
    {"$ref": "#/definitions/ProposeBlock"},
    {"$ref": "#/definitions/RejectBlock"},
    {"$ref": "#/definitions/AnalogBlock"},
    {"$ref": "#/definitions/ReviewAnalogsBlock"},
    {"$ref": "#/definitions/NoopBlock"}
  ]
}
""".strip()


# ---------------------------------------------------------------------------
# Validator factory
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_validator() -> Draft202012Validator:
    """Compile the schema into a :class:`Draft202012Validator` (cached).

    The cache means re-imports and repeated calls share the same
    compiled validator, which is non-trivial to build (~10-30ms the
    first time).
    """
    schema: Dict[str, Any] = json.loads(BLOCKS_SCHEMA_JSON)
    return Draft202012Validator(schema)


__all__ = ["BLOCKS_SCHEMA_JSON", "get_validator"]
