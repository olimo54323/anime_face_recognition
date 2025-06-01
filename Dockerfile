# Użyj nowszego obrazu TensorFlow
FROM tensorflow/tensorflow:2.18.0

# Ustaw katalog roboczy
WORKDIR /app

# Zainstaluj wymagane pakiety systemowe
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Kopiuj requirements i zainstaluj zależności
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --ignore-installed -r requirements.txt

# Kopiuj pliki aplikacji
COPY . .

# Ustaw zmienne środowiskowe
ENV FLASK_APP=app/app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5000

# Uruchom aplikację
CMD ["python", "app/app.py"]