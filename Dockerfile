# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit default port
EXPOSE 8080

# Streamlit runs on port 8080 and host 0.0.0.0 for IBM
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
