FROM python:3.10-slim

# Install uv for fast dependency resolution
RUN pip install uv

# Set working directory
WORKDIR /app

# Expose the standard HF Spaces port
EXPOSE 8080

# Copy dependency files first
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen

# Copy the rest of the application
COPY . .

# Environment setup
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

# Run the FastAPI server
CMD ["uv", "run", "python", "-m", "server.app"]