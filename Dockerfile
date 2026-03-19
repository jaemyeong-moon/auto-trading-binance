FROM python:3.12-slim

WORKDIR /app

# System deps + timezone
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc && \
    rm -rf /var/lib/apt/lists/*
ENV TZ=Asia/Seoul

# Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[dashboard]"

# App code
COPY . .
RUN pip install --no-cache-dir -e .

# Data & logs directories
RUN mkdir -p /app/data /app/logs /app/models

EXPOSE 8502

CMD ["python", "-m", "src.main"]
