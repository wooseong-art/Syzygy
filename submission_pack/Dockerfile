# ==============================================================
# Solar Wind Prediction Competition — submission template
#
# The platform runs:  <ENTRYPOINT> <INFERENCE_TIMESTAMP>
# e.g.                /app/run.sh 20261111T060000Z
#
# Replace everything between the EDIT markers, including the
# language and runtime. Keep the ENTRYPOINT contract.
# ==============================================================
FROM public.ecr.aws/docker/library/python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# ---- PLATFORM LAYER — needed for the upload helper -----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl jq \
    && rm -rf /var/lib/apt/lists/*
COPY upload.sh /app/upload.sh
# -------------------------------------------------------------

# ================= EDIT BELOW =================================
# Python 3.12 from the base image (ECR mirror of python:3.12-slim) — the model was developed on 3.12.3.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=4 MPLCONFIGDIR=/tmp/mpl

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Team model: code + weights copied unchanged from git_repo by tools/sync_model.py (see model/MANIFEST.json)
COPY model/ ./model/
COPY syzygy_live/ ./syzygy_live/
COPY syzygy_model/ ./syzygy_model/
COPY predict.py model_stage.py emergency.sh ./
RUN chmod +x /app/emergency.sh

# ================= EDIT ABOVE =================================

COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh /app/upload.sh \
    && useradd --create-home --uid 1000 runner \
    && chown -R runner /app
USER runner

ENTRYPOINT ["/app/run.sh"]
