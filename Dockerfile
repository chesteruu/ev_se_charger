# EASEE Smart Charger Controller
# Multi-stage build for smaller image size

# Build stage
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# Production stage
FROM python:3.11-slim-bookworm AS production

# Labels
LABEL maintainer="your.email@example.com"
LABEL description="EASEE Smart Charger Controller with price optimization and load balancing"
LABEL version="1.0.0"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Application defaults
    WEB_HOST=0.0.0.0 \
    WEB_PORT=8080 \
    LOG_LEVEL=INFO \
    DATABASE_PATH=/data/easee_controller.db

WORKDIR /app

# Create non-root user for security
RUN groupadd --gid 1000 easee && \
    useradd --uid 1000 --gid easee --shell /bin/bash --create-home easee

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=easee:easee src/ ./src/
COPY --chown=easee:easee pyproject.toml ./

# Create data directory for database
RUN mkdir -p /data && chown easee:easee /data

# Switch to non-root user
USER easee

# Expose web dashboard port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/api/status', timeout=5)" || exit 1

# Default command - use explicit PYTHONPATH for NAS compatibility
ENV PYTHONPATH="/app"
CMD ["python", "-m", "src.main"]
