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

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
