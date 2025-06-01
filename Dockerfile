# Użyj nowszego obrazu TensorFlow zgodnego z wersją z Colab
FROM tensorflow/tensorflow:2.18.0

# Ustaw katalog roboczy
WORKDIR /app

# Zainstaluj wymagane pakiety systemowe dla OpenCV i innych bibliotek
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    libgtk-3-0 \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Kopiuj plik requirements
COPY requirements.txt .

# Zainstaluj dependencies Python z dokładnymi wersjami
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Ustaw API Kaggle (opcjonalne - jeśli potrzebujesz pobierać dane)
RUN mkdir -p /root/.kaggle
COPY kaggle.json /root/.kaggle/ 2>/dev/null || echo "Brak pliku kaggle.json - pomijam"
RUN chmod 600 /root/.kaggle/kaggle.json 2>/dev/null || echo "Brak pliku kaggle.json"

# Kopiuj aplikację
COPY app.py .
COPY preprocessing.py .

# Kopiuj szablony HTML
COPY templates/ ./templates/

# Utwórz niezbędne katalogi
RUN mkdir -p static/uploads models

# Kopiuj modele jeśli istnieją
COPY models/ ./models/ 2>/dev/null || echo "Brak katalogu models - aplikacja będzie działać w trybie demo"

# Ustaw zmienne środowiskowe
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5000

# Utwórz skrypt startowy
RUN echo '#!/bin/bash\n\
echo "=== Anime Face Recognition App ==="\n\
echo "Sprawdzanie struktury plików..."\n\
ls -la\n\
echo "Sprawdzanie katalogu models..."\n\
ls -la models/ 2>/dev/null || echo "Katalog models nie istnieje"\n\
echo "Sprawdzanie katalogu templates..."\n\
ls -la templates/\n\
echo "Uruchamianie aplikacji..."\n\
python app.py' > /app/start.sh && chmod +x /app/start.sh

# Uruchom aplikację
CMD ["/app/start.sh"]