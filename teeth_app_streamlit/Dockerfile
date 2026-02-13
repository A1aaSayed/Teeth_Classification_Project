# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose streamlit port
EXPOSE 8501

# Run streamlit app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "server.port=8501"]