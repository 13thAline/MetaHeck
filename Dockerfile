FROM python:3.10-slim

# Install uv for fast dependency resolution
RUN pip install uv

# Set working directory
WORKDIR /app

# Expose the standard HF Spaces port
EXPOSE 7860

# Copy dependency files first
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen

# Copy the rest of the application
COPY . .

# Environment setup
ENV PYTHONUNBUFFERED=1

# Run the FastAPI server using uv to automatically handle the venv
CMD ["uv", "run", "python", "-m", "server.app"]