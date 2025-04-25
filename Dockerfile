# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app


COPY requirements.txt .
# Copy code


# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Expose the port Flask runs on
EXPOSE 5000

# Run the app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
