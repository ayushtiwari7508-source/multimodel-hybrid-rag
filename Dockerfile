FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# System deps for doc parsing + OCR (tesseract, ghostscript for camelot, poppler)
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
        ghostscript \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Web app deps need pyproject/requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY ui ./ui
COPY conftest.py .
COPY pyproject.toml .

EXPOSE 8000 8501

CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]