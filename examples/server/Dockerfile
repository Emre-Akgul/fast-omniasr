# Build from the repository root:
#   docker build -f examples/server/Dockerfile -t fast-omniasr-server .
#   docker run --rm -p 8000:8000 fast-omniasr-server
# GPU: --build-arg EXTRAS=cuda,hub,resample, then run with --gpus all -e DEVICE=cuda.
FROM python:3.12-slim

ARG EXTRAS=onnx,hub,resample
ARG MODEL=EmreAkgul/omniASR-CTC-300M-v2-ONNX
ARG PRECISION=fp32

# Install this checkout. To deploy a release instead, replace the next two lines with
#   RUN pip install --no-cache-dir "fast-omniasr[${EXTRAS}]"
COPY pyproject.toml README.md LICENSE /src/
COPY src /src/src
RUN pip install --no-cache-dir "/src[${EXTRAS}]" && rm -rf /src

ENV FAST_OMNIASR_MODEL=${MODEL} FAST_OMNIASR_PRECISION=${PRECISION} \
    DEVICE=cpu WORKERS=1 MAX_QUEUE=8

# Download the model at build time into a fixed, world-readable cache, so the container
# starts with no network access under any user id. Loading it here runs on the CPU, so
# preload fp32 or fp16 (bf16 has no CPU kernels).
RUN python -c "import os; from fast_omniasr import OmniASR; \
OmniASR.from_pretrained(os.environ['FAST_OMNIASR_MODEL'], cache_dir='/models', \
precision=os.environ['FAST_OMNIASR_PRECISION']).close()" \
    && chmod -R a+rX /models

COPY examples/server/server.py /app/server.py
RUN chmod a+r /app/server.py && useradd --create-home --uid 1000 app
WORKDIR /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready')"
# `exec` makes the server PID 1, so `docker stop` sends its SIGTERM straight to it.
CMD ["sh", "-c", "exec python server.py --model \"$FAST_OMNIASR_MODEL\" --precision \"$FAST_OMNIASR_PRECISION\" --device \"$DEVICE\" --workers \"$WORKERS\" --max-queue \"$MAX_QUEUE\" --cache-dir /models --local-files-only"]
