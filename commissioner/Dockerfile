# Gnomic's league commissioner: the current config-driven ruleset-strategy
# commissioner with Gnomic's scheduling policy baked into the image.
#
# Build from a pinned coworld-tools source revision rather than the older
# commissioners-default registry image: current qualifier transitions are
# required for hosted submissions to leave the qualifying state.
FROM python:3.13-slim

ARG COWORLD_TOOLS_REF=8f842e7e5fbf8eb00839bd75d8356b382b63b1cb

ENV COMMISSIONER_KEY=config_driven \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --filter=blob:none --no-checkout https://github.com/Metta-AI/coworld-tools /tmp/coworld-tools \
    && git -C /tmp/coworld-tools sparse-checkout set commissioners \
    && git -C /tmp/coworld-tools checkout "${COWORLD_TOOLS_REF}" \
    && cp -r /tmp/coworld-tools/commissioners /app \
    && rm -rf /tmp/coworld-tools

WORKDIR /app
RUN pip install --no-cache-dir .

COPY gnomic.yaml /gnomic.yaml
COPY gnomic_commissioner.py /app/gnomic_commissioner.py
ENV RULESET_STRATEGY_CONFIG_PATH=/gnomic.yaml

EXPOSE 8080
CMD ["uvicorn", "gnomic_commissioner:app", "--host", "0.0.0.0", "--port", "8080"]
