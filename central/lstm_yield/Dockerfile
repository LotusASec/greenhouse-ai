FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
ENV SERVICE_PORT=9002

EXPOSE 9002

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9002/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "9002"]
