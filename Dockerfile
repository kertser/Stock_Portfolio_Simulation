# ── Stage 1: build deps with uv ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy dependency files first (layer cache)
COPY pyproject.toml uv.lock ./

# Install production dependencies into /app/.venv (no project, no dev deps)
RUN uv sync --frozen --no-install-project --no-dev

# ── Stage 2: runtime ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Install uv (needed to run scripts via `uv run`)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Create non-root user
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy venv from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY --chown=appuser:appgroup . .

# Create cache directory (for yfinance data cache)
RUN mkdir -p /app/.cache && chown appuser:appgroup /app/.cache

USER appuser

# Environment defaults (override via -e or .env)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PORT=8050 \
    HOST=0.0.0.0 \
    DEBUG=false

EXPOSE 8050

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8050/_dash-layout')" || exit 1

CMD ["python", "-m", "portfolio_monte_carlo.dash_app"]
