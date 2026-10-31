# Bike-slot camera relay: receives snapshots + health from the camera, serves the latest.
# Build:  docker build -t bikeslot-relay .
# Run:    docker run -p 8000:8000 -e UPLOAD_TOKEN=... -e VIEW_KEY=... bikeslot-relay
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py viewer.html ./

# no root inside the container
RUN useradd --system --uid 10001 relay
USER relay

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/healthz', timeout=2)"

# One process, one worker: the latest snapshot lives in this process' memory.
# $PORT is set by hosts like Render; --proxy-headers because they terminate HTTPS in front of us.
CMD ["sh", "-c", "exec uvicorn app:create_app --factory --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips='*' --no-access-log"]
