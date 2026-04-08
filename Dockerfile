FROM python:3.11-slim

WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir \
    openenv-core \
    networkx \
    numpy \
    pydantic \
    fastapi \
    uvicorn \
    openai \
    python-dotenv \
    requests

# Expose port
EXPOSE 8000

# Set workers via environment
ENV WORKERS=4

# Run with uvicorn
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
