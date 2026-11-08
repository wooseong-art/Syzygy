# ==============================================================
# Solar Wind Prediction Competition — submission template
#
# The platform runs:  <ENTRYPOINT> <INFERENCE_TIMESTAMP>
# e.g.                /app/run.sh 20261111T060000Z
#
# Replace everything between the EDIT markers, including the
# language and runtime. Keep the ENTRYPOINT contract.
# ==============================================================
FROM public.ecr.aws/docker/library/python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# ---- PLATFORM LAYER — needed for the upload helper -----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl jq \
    && rm -rf /var/lib/apt/lists/*
COPY upload.sh /app/upload.sh
# -------------------------------------------------------------

# ================= EDIT BELOW =================================
# Python 3.11 comes from the base image (ECR mirror of python:3.11-slim).
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY syzygy_live/ ./syzygy_live/
COPY predict.py model_stage.py emergency.sh ./
RUN chmod +x /app/emergency.sh
# Model code and weights go here once wired in (stage C):
# COPY model/ ./model/
# COPY weights/ ./weights/

# ================= EDIT ABOVE =================================

COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh /app/upload.sh \
    && useradd --create-home --uid 1000 runner \
    && chown -R runner /app
USER runner

ENTRYPOINT ["/app/run.sh"]
