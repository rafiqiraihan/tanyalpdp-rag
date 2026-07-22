FROM python:3.10-slim

WORKDIR /app

COPY requirements-docker.txt .

RUN pip install --no-cache-dir \
    torch==2.6.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements-docker.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]