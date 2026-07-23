FROM python:3.11-slim

WORKDIR /app

# opencv-python-headless still needs a couple of shared libs at runtime even
# without any GUI/X11 support.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copies whatever is in the build context (minus .dockerignore entries).
# frontend/ is optional — if it's absent (api-only deliverable), app/main.py
# simply skips mounting it. No Dockerfile or code change needed between variants.
COPY . .

ENV PORT=8000
EXPOSE 8000

# --forwarded-allow-ips='*' is required for the per-IP rate limiter to see
# real end-user IPs: Cloud Run's front-end proxy is the only thing that can
# ever reach this container directly, but uvicorn's forwarded_allow_ips
# defaults to trusting only 127.0.0.1, so X-Forwarded-For was being ignored
# and request.client.host resolved to the proxy's own address instead of the
# caller's. --proxy-headers is uvicorn's default already; set explicitly here
# so the intent isn't silently dependent on that default staying unchanged.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
