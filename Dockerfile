# CiteLume - container for Hugging Face Spaces (Docker SDK) or any Docker host.
# Stage 1: build the React UI
FROM node:20-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# Stage 2: Python API that also serves the built UI
FROM python:3.12-slim
# DejaVu fonts let the PDF export print Greek letters, symbols and accented names
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
# Hugging Face Spaces run containers as user 1000
RUN useradd -m -u 1000 user
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY --chown=user backend/ backend/
COPY --chown=user --from=ui /ui/dist frontend/dist
RUN mkdir -p backend/.cache && chown -R user /app
USER user
ENV PYTHONUNBUFFERED=1 PORT=7860
WORKDIR /app/backend
EXPOSE 7860
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
