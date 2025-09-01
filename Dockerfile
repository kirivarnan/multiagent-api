# Use an official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --default-timeout=120 -r requirements.txt

# Copy the FastAPI app code
# Copy application code and .env file
COPY . .
#COPY simple_main.py .

# Expose FastAPI's default port
EXPOSE 8060

# Run the application with uvicorn
#CMD ["uvicorn", "simple_main:app", "--host", "0.0.0.0", "--port", "8060"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8060"]
