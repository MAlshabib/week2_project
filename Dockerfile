# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install basic dependencies
RUN apt update && apt install -y curl

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
