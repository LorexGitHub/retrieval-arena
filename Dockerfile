# Stage 1: Build frontend
FROM node:20-alpine AS frontend
WORKDIR /app
COPY rag-ui/package.json rag-ui/package-lock.json ./
RUN npm ci
COPY rag-ui/ .
RUN npm run build

# Stage 2: Python backend
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git libpq-dev && rm -rf /var/lib/apt/lists/*

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Copy built frontend
COPY --from=frontend /app/dist /app/rag-ui/dist

EXPOSE 8002

CMD ["uvicorn", "src.combined_app:app", "--host", "0.0.0.0", "--port", "8000"]
