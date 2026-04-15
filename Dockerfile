# Backend Dockerfile - Senior Pro Production Edition
FROM python:3.12-slim as builder

WORKDIR /app
COPY requirements.txt .

# Install build dependencies if needed (e.g., for compiled packages)
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Final Production Image ---
FROM python:3.12-slim

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV LOCAL_DEV=false

# Expose API port
EXPOSE 8000

# Start Backend using uvicorn
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
