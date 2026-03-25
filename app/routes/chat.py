"""POST /v1/chat/completions — proxy to DashScope with retry and streaming."""

import asyncio
from typing import Any

import httpx
from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..auth import AuthManager
from ..config import DEFAULT_MODEL, MAX_RETRIES, RETRY_DELAY_S, log
from ..headers import build_headers
from ..models import (
    clamp_max_tokens,
    is_auth_error,
    is_quota_error,
    is_validation_error,
    make_error_response,
    resolve_model,
)

router = APIRouter()


async def _handle_regular(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> JSONResponse:
    resp = await client.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    return JSONResponse(content=resp.json())


async def _handle_streaming(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> StreamingResponse:
    req = client.build_request("POST", url, json=payload, headers=headers)
    resp = await client.send(req, stream=True)
    resp.raise_for_status()

    async def generate():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    x_api_key: str | None = Header(None),
    authorization: str | None = Header(None),
) -> JSONResponse | StreamingResponse:
    from ..main import validate_api_key

    validate_api_key(x_api_key, authorization)

    auth: AuthManager = request.app.state.auth
    client: httpx.AsyncClient = request.app.state.http_client
    request.app.state.request_count += 1

    body: dict[str, Any] = await request.json()
    is_streaming: bool = body.get("stream", False)
    model = resolve_model(body.get("model", DEFAULT_MODEL))
    max_tokens = clamp_max_tokens(model, body.get("max_tokens", 65536))

    access_token = await auth.get_valid_token(client)
    creds = auth.load_credentials()
    url = f"{auth.get_api_endpoint(creds)}/chat/completions"

    payload: dict[str, Any] = {
        "model": model,
        "messages": body.get("messages", []),
        "stream": is_streaming,
        "temperature": body.get("temperature", 0.7),
        "max_tokens": max_tokens,
    }
    for field in (
        "top_p",
        "top_k",
        "repetition_penalty",
        "tools",
        "tool_choice",
        "reasoning",
    ):
        if field in body:
            payload[field] = body[field]

    if is_streaming:
        payload["stream_options"] = {"include_usage": True}

    headers = build_headers(access_token, streaming=is_streaming)

    last_error: Exception | None = None
    last_status: int | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if is_streaming:
                return await _handle_streaming(client, url, payload, headers)
            else:
                return await _handle_regular(client, url, payload, headers)
        except httpx.HTTPStatusError as exc:
            last_error = exc
            status = exc.response.status_code
            last_status = status

            # Check for validation errors first (return 400, don't retry)
            error_message = str(exc)
            if is_validation_error(error_message):
                log.warning(
                    "Validation error (status %d): %s", status, error_message[:100]
                )
                return JSONResponse(
                    status_code=400,
                    content=make_error_response(
                        error_message,
                        error_type="validation_error",
                        code="invalid_request",
                    ),
                )

            # Retry on server errors and rate limits
            if status in (500, 429) and attempt < MAX_RETRIES:
                log.warning("Retry %d/%d (status %d)", attempt, MAX_RETRIES, status)
                await asyncio.sleep(RETRY_DELAY_S * attempt)
                continue

            # Auth errors trigger token refresh
            if is_auth_error(status, error_message):
                try:
                    log.info("Auth error %d, refreshing token...", status)
                    creds = auth.load_credentials()
                    if creds:
                        new_creds = await auth.refresh_token(creds, client)
                        headers = build_headers(
                            new_creds.access_token, streaming=is_streaming
                        )
                        if is_streaming:
                            return await _handle_streaming(
                                client, url, payload, headers
                            )
                        else:
                            return await _handle_regular(client, url, payload, headers)
                except Exception as refresh_err:
                    log.error("Token refresh failed: %s", str(refresh_err))
                    # Return auth error instead of generic 500
                    return JSONResponse(
                        status_code=401,
                        content=make_error_response(
                            "Authentication failed. Please re-authenticate with Qwen CLI.",
                            error_type="authentication_error",
                            code="invalid_token",
                        ),
                    )
            break

        except Exception as exc:
            last_error = exc
            error_message = str(exc)

            # Check for validation errors
            if is_validation_error(error_message):
                log.warning("Validation error: %s", error_message[:100])
                return JSONResponse(
                    status_code=400,
                    content=make_error_response(
                        error_message,
                        error_type="validation_error",
                        code="invalid_request",
                    ),
                )

            # Retry on generic errors
            if attempt < MAX_RETRIES:
                log.warning(
                    "Retry %d/%d (error: %s)", attempt, MAX_RETRIES, error_message[:50]
                )
                await asyncio.sleep(RETRY_DELAY_S * attempt)
                continue
            break

    # Build appropriate error response based on error type
    error_msg = str(last_error) if last_error else "Unknown error"

    if is_validation_error(error_msg):
        return JSONResponse(
            status_code=400,
            content=make_error_response(
                error_msg, error_type="validation_error", code="invalid_request"
            ),
        )

    if is_quota_error(last_status, error_msg):
        return JSONResponse(
            status_code=429,
            content=make_error_response(
                "Rate limit or quota exceeded. Please try again later.",
                error_type="rate_limit_exceeded",
                code="rate_limit_exceeded",
            ),
        )

    if is_auth_error(last_status, error_msg):
        return JSONResponse(
            status_code=401,
            content=make_error_response(
                "Authentication failed. Please re-authenticate with Qwen CLI.",
                error_type="authentication_error",
                code="invalid_token",
            ),
        )

    # Default: generic API error
    return JSONResponse(
        status_code=500,
        content=make_error_response(error_msg, error_type="api_error"),
    )
