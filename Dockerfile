FROM python:3.10-slim

# Install uv for fast dependency resolution
RUN pip install uv

# Set working directory
WORKDIR /app

# Expose the standard HF Spaces port
EXPOSE 7860

# Copy EVERYTHING into the container first
COPY . /app/

# Environment setup
ENV PYTHONUNBUFFERED=1

# Now that the 'env' and 'server' folders are actually inside /app, run the sync
RUN uv sync --frozen

# Run the FastAPI server using uv to automatically handle the venv
CMD ["uv", "run", "python", "-m", "server.app"]