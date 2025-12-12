# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.12-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy local code to the container image.
COPY requirements.txt .

# Install production dependencies.
# We use standard pip here.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Run the web service on container startup.
# We use gunicorn with 1 worker and 8 threads to handle concurrent requests.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app