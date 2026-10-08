# Multi-purpose Dockerfile for Fraud Ring & Coordinated Intelligence Platform
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source tree and model artifacts
COPY configs/ ./configs/
COPY src/ ./src/
COPY api/ ./api/
COPY streamlit_app/ ./streamlit_app/
COPY artifacts/ ./artifacts/
COPY data/processed/ ./data/processed/

# Expose ports: 8000 for FastAPI, 8501 for Streamlit
EXPOSE 8000 8501

# Default command starts FastAPI service
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
