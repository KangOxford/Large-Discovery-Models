"""Framework-neutral route adapter for ``bo_api.py`` JSON functions."""

from __future__ import annotations

import json
from typing import Any, Tuple

import bo_api


ROUTES: dict[str, str] = {
    "/score/vina": "score_vina_json",
    "/score/nn": "score_nn_json",
    "/acquisition/evaluate": "evaluate_acquisition_json",
}


def handle_request(
    path: str,
    request_body: str,
    **provider_kwargs: Any,
) -> Tuple[int, str]:
    """Return ``(http_status, json_body)`` for a JSON request."""
    handler_name = ROUTES.get(str(path))
    if handler_name is None:
        return 404, json.dumps({
            "ok": False,
            "error": f"unknown route: {path}",
            "error_type": "NotFound",
        })
    kwargs = _kwargs_for_route(path, provider_kwargs)
    handler = getattr(bo_api, handler_name)
    response = handler(request_body, **kwargs)
    status = 502 if _is_bo_api_error(response) else 200
    return status, response


def _kwargs_for_route(path: str, provider_kwargs: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "/score/vina": {"vina_bin", "vina_cache_dir", "vina_max_workers"},
        "/score/nn": {"nn_model_path", "nn_metadata_path"},
        "/acquisition/evaluate": {"gp_device"},
    }[path]
    return {k: v for k, v in provider_kwargs.items() if k in allowed}


def _is_bo_api_error(response: str) -> bool:
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        return True
    return bool(data.get("error") or data.get("ok") is False)


__all__ = ["ROUTES", "handle_request"]
