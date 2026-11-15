FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for XGBoost (OpenMP) and health checking
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application artifacts and web service
COPY model.bin predict.py ./

# Expose API port
EXPOSE 9696

# Health check
HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
  CMD curl -f http://localhost:9696/health || exit 1

# Start FastAPI application with uvicorn
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
