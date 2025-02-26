# Stage 1: Build frontend
FROM node:18 AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Python backend with frontend static files
FROM python:3.12-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install them
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ backend/
COPY backend/male_data.csv male_data.csv
COPY backend/female_data.csv female_data.csv


# Copy built frontend from previous stage
COPY --from=frontend-builder /app/frontend/dist frontend/dist

# Set environment variables
ENV PORT=80

# Expose the port
EXPOSE 80

# go into the backend directory
WORKDIR /app/backend

# Run the backend server
CMD ["python3", "main.py"] 