# Qwen Code API (Python) - OpenAI-Compatible Proxy Server

A Python proxy server that exposes Qwen models through an OpenAI-compatible API endpoint. Supports tool calling, streaming, and automatic token refresh.

This is a Python port of the original [qwen-code-oai-proxy](https://github.com/aptdnfapt/qwen-code-oai-proxy) (Node.js/JavaScript).

## Features

- **OpenAI-Compatible API** - Works with opencode, crush, claude code router, roo code, cline, and other OpenAI-compatible clients
- **Tool Calling Support** - Full support for function/tool calling
- **Streaming Responses** - Server-sent events (SSE) streaming support
- **Automatic Token Refresh** - OAuth token management with automatic refresh
- **Retry Logic** - Automatic retry on 500/429 errors with exponential backoff
- **API Key Authentication** - Secure your proxy with API keys
- **coder-model** - Qwen3.5-Plus (free via OAuth)

## Quick Start

### Option 1: Using Docker (Recommended)

1. **Authenticate**:

    You need to authenticate with Qwen to generate the required credentials file. Use the official `qwen-code` CLI:

    ```bash
    qwen login
    ```

    This creates the `~/.qwen/oauth_creds.json` file that will be mounted into the container.

2. **Configure Environment**:

    ```bash
    cp .env.example .env.secret
    # Edit .env.secret file with your desired configuration
    ```

3. **Build and Run with Docker Compose**:

    ```bash
    docker compose --env-file .env.secret up -d
    ```

4. **Use the Proxy**: Point your OpenAI-compatible client to `http://localhost:8080/v1`

### Option 2: Local Development

1. **Install Dependencies** (using uv):

    ```bash
    uv sync
    ```

2. **Authenticate**: You need to authenticate with Qwen to generate the required credentials file.

    ```bash
    qwen login
    ```

    This will create the `~/.qwen/oauth_creds.json` file needed by the proxy server.

3. **Start the Server**:

    ```bash
    uv run python -m qwen_code_api.main
    ```

    Or with uvicorn directly:

    ```bash
    uv run uvicorn qwen_code_api.main:app --host 0.0.0.0 --port 8080
    ```

4. **Use the Proxy**: Point your OpenAI-compatible client to `http://localhost:8080/v1`.

**Note**: API key can be any random string if not configured - it doesn't matter for this proxy.

## API Key Authentication

The proxy can be secured with API keys to prevent unauthorized access.

### Setting up API Keys

1. **Single API Key:**

   ```bash
   QWEN_CODE_API_KEY=your-secret-key-here
   ```

2. **Multiple API Keys:**

   ```bash
   QWEN_CODE_API_KEY=key1,key2,key3
   ```

If no API key is configured, the proxy will not require authentication.

## Configuration

The proxy server can be configured using environment variables. Create a `.env.secret` file in the project root (copy from `.env.example`).

| Variable             | Description                                              | Default            |
| -------------------- | -------------------------------------------------------- | ------------------ |
| `PORT`               | Server port                                              | `8080`             |
| `HOST`               | Server host                                              | `0.0.0.0`          |
| `LOG_LEVEL`          | Logging level (`error`, `debug`)                         | `error`            |
| `MAX_RETRIES`        | Maximum retry attempts for failed requests               | `5`                |
| `RETRY_DELAY_MS`     | Base retry delay in milliseconds                         | `1000`             |
| `QWEN_CODE_AUTH_USE` | Use OAuth authentication from `~/.qwen/oauth_creds.json` | `true`             |
| `QWEN_CODE_API_KEY`  | API key(s) for authentication (comma-separated)          | (none)             |
| `DEFAULT_MODEL`      | Default model to use when not specified                  | `coder-model`      |

## Example Usage

### Health Check

Monitor the proxy status:

```bash
curl http://localhost:8080/health
```

### Using Python

```python
from openai import OpenAI

client = OpenAI(
    api_key="fake-key",
    base_url="http://localhost:8080/v1"
)

response = client.chat.completions.create(
    model="coder-model",
    messages=[
        {"role": "user", "content": "2+2=?"}
    ]
)

print(response.choices[0].message.content)
```

## Supported Models

The proxy supports Qwen models available through your Qwen Code OAuth account:

| Model ID      | Description                   | Max Tokens | Notes                             |
| ------------- | ----------------------------- | ---------- | --------------------------------- |
| `coder-model` | Qwen3.5-Plus (via OAuth free) | 32000      | **Only model available for free** |
  
**Note**: With free OAuth authentication, only `coder-model` is available. The proxy passes through whatever models your Qwen account has access to. Use `/model` in Qwen Code CLI to see available models for your account.

## Supported Endpoints

- `POST /v1/chat/completions` - Chat completions (streaming and non-streaming)
- `GET /v1/models` - List available models
- `GET /health` - Health check and status

## Development

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
# Install dependencies
uv sync

# Run type checking
uv run poe typecheck

# Run linting
uv run poe lint

# Run formatting
uv run poe format

# Run all checks
uv run poe check

# Run tests
uv run pytest

# Compare proxy requests against real Qwen Code CLI (requires mitmproxy + qwen CLI)
uv run poe compare-requests
```

## Important Notes

- **Token Limits**: Users might face errors or 504 Gateway Timeout issues when using contexts with 130,000 to 150,000 tokens or more. This appears to be a practical limit for Qwen models.
- **Authentication**: The proxy uses OAuth credentials from `~/.qwen/oauth_creds.json`. Make sure to authenticate with the Qwen CLI before starting the proxy.
- **Token Refresh**: The proxy automatically refreshes OAuth tokens when they expire.

## Updating for Newer Qwen Code Versions

When a new version of Qwen Code is released, you need to update the request headers in [headers.py](src/qwen_code_api/headers.py) to match the new client. Otherwise the API may reject requests from an outdated client fingerprint.

### Manual: intercept with mitmproxy

1. **Install the latest Qwen Code CLI**:

    ```bash
    pnpm add -g @qwen-code/qwen-code
    ```

2. **Intercept the headers** using mitmproxy:

    ```bash
    # Terminal 1: start the proxy
    mitmweb --listen-port 8080

    # Terminal 2: run qwen through the proxy
    HTTPS_PROXY=http://127.0.0.1:8080 NODE_TLS_REJECT_UNAUTHORIZED=0 qwen
    ```

3. **Send a chat message** in the Qwen Code CLI so it makes an API request.

4. **Copy the headers** from the mitmweb UI (`http://127.0.0.1:8081`) — look for requests to `portal.qwen.ai`. The relevant headers to update are:

    - `user-agent` — contains the Qwen Code version (e.g. `QwenCode/0.14.0`)
    - `x-dashscope-useragent` — same version string
    - `x-stainless-package-version` — OpenAI SDK version used internally
    - `x-stainless-runtime-version` — Node.js version

5. **Update** [headers.py](src/qwen_code_api/headers.py) with the new values.

### With a coding agent

Tell your agent:

> Install the latest `@qwen-code/qwen-code` globally with pnpm, then extract the header values from the installed bundle and update `src/qwen_code_api/headers.py`:
>
> 1. Find the pnpm global modules dir (`pnpm root -g`), then read `@qwen-code/qwen-code/package.json` for the CLI version.
> 2. Grep `cli.js` in that package for `VERSION4 =` to get the OpenAI SDK version (`x-stainless-package-version`).
> 3. Update `headers.py`: set the CLI version in `user-agent` and `x-dashscope-useragent` (format: `QwenCode/<version> (linux; x64)`), set the SDK version in `x-stainless-package-version`, and set `x-stainless-runtime-version` to the output of `node --version`.

### Verify after updating

After updating headers, run the comparison script to confirm the proxy's requests match the real client:

```bash
uv run poe compare-requests
```

This starts mitmdump, sends a request through both the proxy and the real Qwen Code CLI, and diffs the outgoing requests (URL, headers, body structure). Exit code 0 means no differences.

## Differences from Node.js Version

This Python implementation maintains feature parity with the original Node.js version while leveraging Python's async capabilities:

- Built with **FastAPI** and **uvicorn**
- Uses **httpx** for async HTTP requests
- OAuth token management with automatic refresh
- Same Docker deployment workflow
- Compatible with all the same AI agents and tools

## License

MIT
