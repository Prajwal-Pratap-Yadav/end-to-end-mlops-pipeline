# One image runs every Python service in docker-compose.yml (MLflow server,
# trainer, API, monitor, simulator). Sharing it guarantees the MLflow client and
# server versions match and keeps local builds to a single dependency layer.

ARG PYTHON_VERSION=3.11

# ---------------------------------------------------------------- builder stage
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Locked, cross-platform requirements: this layer is cached until they change.
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------------------------------------------------------------- runtime stage
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:${PATH}" \
    LOG_FORMAT=json \
    MLFLOW_DISABLE_TELEMETRY=true \
    DO_NOT_TRACK=true \
    PROMETHEUS_DISABLE_CREATED_SERIES=True \
    GIT_PYTHON_REFRESH=quiet

# Unprivileged runtime user with a fixed high UID (no clash with host users) and its
# own writable home, where libraries such as matplotlib keep their caches. No login shell.
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --home-dir /home/app \
       --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app configs ./configs
COPY --chown=app:app src ./src
COPY --chown=app:app app ./app
COPY --chown=app:app monitoring ./monitoring

# Writable locations are created up front so named volumes inherit the ownership.
RUN mkdir -p /app/data /app/reports /mlflow \
    && chown -R app:app /app/data /app/reports /mlflow

USER app
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
