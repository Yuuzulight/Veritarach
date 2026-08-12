FROM python:3.11-slim

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src/ src/

RUN uv sync --no-dev

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "veritarach.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
