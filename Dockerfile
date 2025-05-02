# Use official TensorFlow image as the base image
FROM tensorflow/tensorflow:latest

# Set working directory
WORKDIR /app

# Install required packages for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies with --ignore-installed flag to avoid errors with pre-installed packages
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Set up Kaggle API
RUN mkdir -p /root/.kaggle
COPY kaggle.json /root/.kaggle/
RUN chmod 600 /root/.kaggle/kaggle.json

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p app/static/uploads

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app/app.py
ENV FLASK_ENV=production

# Run the application
CMD ["python", "app/app.py"]