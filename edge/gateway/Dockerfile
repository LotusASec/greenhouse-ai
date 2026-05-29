FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
ENV SERVICE_PORT=8100

EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8100/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100"]
