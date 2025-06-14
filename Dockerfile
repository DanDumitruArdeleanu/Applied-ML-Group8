FROM python:3.10-slim

# Set working directory
WORKDIR /app


# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Set environment variable
ENV PYTHONUNBUFFERED=1



# Run the API
CMD ["python", "scripts/main.py"]