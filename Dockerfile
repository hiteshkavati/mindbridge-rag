FROM python:3.12-slim

ARG WITH_SBERT=false
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir . && \
    if [ "$WITH_SBERT" = "true" ]; then pip install --no-cache-dir ".[sbert]"; fi

RUN useradd --create-home app && mkdir /data && chown app /data
USER app
ENV MB_DB_PATH=/data/mindbridge.db MB_ENV=prod

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["mindbridge", "serve", "--host", "0.0.0.0", "--port", "8000"]
