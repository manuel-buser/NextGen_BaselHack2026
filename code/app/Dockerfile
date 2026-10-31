FROM python:3.12-slim
# time zone data, so TZ (compose: Europe/Zurich) sets the clock of the history and the predictions
RUN apt-get update && apt-get install -y --no-install-recommends tzdata && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY bikeslot ./bikeslot
COPY racks.yaml racks.demo.yaml ./
ENV PATH="/app/.venv/bin:$PATH" DETECTOR_URL=fake BIKESLOT_RACKS=/app/racks.yaml
EXPOSE 8000
CMD ["uvicorn", "bikeslot.main:app", "--host", "0.0.0.0", "--port", "8000"]
