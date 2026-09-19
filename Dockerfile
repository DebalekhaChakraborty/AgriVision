# AgriVision inspection service — serving image.
#
# Multi-stage: wheels are built in a stage that may carry a compiler, and the
# runtime stage receives only the installed packages. The build tooling never
# reaches the final image.
#
# The base is pinned by digest. A tag is not a pin: python:3.11-slim-bookworm
# moves, and an image that rebuilds differently next month is not reproducible.

FROM python:3.11-slim-bookworm@sha256:a36c24f9cbdf4fd0f52d67f0823eeac19c2028c637cecc392d97f980d4fec56b AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY requirements-serving.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
 && /opt/venv/bin/pip install -r requirements-serving.txt


FROM python:3.11-slim-bookworm@sha256:a36c24f9cbdf4fd0f52d67f0823eeac19c2028c637cecc392d97f980d4fec56b AS runtime

# The opencv-python wheel links against X and GL even when no window is ever
# created. Determined by running `ldd` on cv2.abi3.so inside the image rather
# than guessed: the wheel reported libxcb.so.1 and libGL.so.1 missing, and the
# container failed at `import cv2` while the host was fine. That is precisely
# why container verification is a separate step from host verification.
#
# The alternative is opencv-python-headless, which drops these. It is NOT used
# here: every phase from 1 onward verified opencv-python 5.0.0.93 specifically,
# and swapping the wheel in the serving image would mean the deployed binary is
# not the one the calibration was measured against. Two small libraries are a
# cheaper price than that discrepancy.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libglib2.0-0 \
      libgomp1 \
      libxcb1 \
      libgl1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_DIR=/opt/agrivision/model

WORKDIR /app
# Only what the serving path imports. The research trees, the evaluation
# harnesses, the tests and the corpus manifests are not copied.
COPY competition/__init__.py            competition/__init__.py
COPY competition/agent/                 competition/agent/
COPY competition/models/                competition/models/
COPY competition/service/               competition/service/
COPY competition/vision/                competition/vision/
COPY competition/data/__init__.py       competition/data/__init__.py
COPY competition/evaluation/__init__.py competition/evaluation/__init__.py
# The locked policies the orchestrator loads at startup. Small JSON records,
# not imagery.
COPY competition/evaluation/results/phase2c_locked_policy.json \
     competition/evaluation/results/phase2c_locked_policy.json
COPY competition/evaluation/results/phase2d/locked_artifact_policy.json \
     competition/evaluation/results/phase2d/locked_artifact_policy.json

# The ONNX artifact is NOT baked in. It is fetched from S3 at startup and
# verified against a SHA-256 recorded in configuration, because its source
# checkpoint carries no redistribution grant.
# The login shell is /bin/sh and NOT /usr/sbin/nologin. That looks like a
# hardening detail and is load-bearing: App Runner failed to start this exact
# image with a nologin shell, reporting only "Failed to deploy your application
# image" with no application logs at all. Bisected against a minimal image, the
# same build with only this changed deploys and runs. Non-root is kept; the
# nologin shell is not.
RUN mkdir -p /opt/agrivision/model \
 && useradd --create-home --shell /bin/sh --uid 10001 agrivision \
 && chown -R agrivision:agrivision /opt/agrivision /app
USER agrivision

EXPOSE 8080
# Useful for `docker run` locally. App Runner ignores it and applies its own
# configured health check; bisection confirmed it was not the cause of the
# deployment failure described in PHASE4_AWS_DEPLOYMENT.md.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=4).status==200 else 1)"

# One worker. The perception path is CPU-bound and already uses OpenCV's
# internal parallelism; a second worker would contend for the same cores and
# double the memory for the loaded model.
CMD ["uvicorn", "competition.service.api:app", \
     "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
