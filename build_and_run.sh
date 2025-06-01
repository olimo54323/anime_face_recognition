#!/bin/bash

echo "=== Anime Face Recognition - Build & Run Script ==="

# Kolory dla lepszej czytelności
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Funkcja logowania
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Sprawdź czy Docker jest zainstalowany
if ! command -v docker &> /dev/null; then
    error "Docker nie jest zainstalowany!"
    exit 1
fi

log "Docker znaleziony: $(docker --version)"

# Utwórz strukture katalogów
log "Tworzenie struktury katalogów..."
mkdir -p templates static/uploads models

# Sprawdź czy pliki istnieją
log "Sprawdzanie plików..."
required_files=("app.py" "requirements.txt" "Dockerfile" "templates/index.html" "templates/results.html")
missing_files=()

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        missing_files+=("$file")
    fi
done

if [[ ${#missing_files[@]} -gt 0 ]]; then
    error "Brakujące pliki:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    warning "Utwórz brakujące pliki przed uruchomieniem!"
    exit 1
fi

log "Wszystkie wymagane pliki obecne ✅"

# Sprawdź czy model istnieje
if [[ -d "models" ]] && [[ -n "$(ls -A models/)" ]]; then
    log "Znaleziono model w katalogu models/ ✅"
    ls -la models/
else
    warning "Brak modelu - aplikacja będzie działać w trybie demo"
fi

# Zbuduj obraz Docker
log "Budowanie obrazu Docker..."
if docker build -t anime-face-recognition . ; then
    log "Obraz Docker zbudowany pomyślnie ✅"
else
    error "Błąd podczas budowania obrazu Docker"
    exit 1
fi

# Zatrzymaj istniejący kontener (jeśli istnieje)
log "Sprawdzanie istniejących kontenerów..."
if docker ps -a --format "table {{.Names}}" | grep -q "anime-app"; then
    log "Zatrzymywanie istniejącego kontenera..."
    docker stop anime-app 2>/dev/null || true
    docker rm anime-app 2>/dev/null || true
fi

# Uruchom kontener
log "Uruchamianie kontenera..."
if docker run -d \
    --name anime-app \
    -p 5000:5000 \
    -v "$(pwd)/models:/app/models" \
    -v "$(pwd)/static/uploads:/app/static/uploads" \
    anime-face-recognition; then
    
    log "Kontener uruchomiony pomyślnie ✅"
    log "Aplikacja dostępna na: http://localhost:5000"
    
    # Pokaż logi przez kilka sekund
    log "Logi aplikacji:"
    sleep 3
    docker logs anime-app
    
else
    error "Błąd podczas uruchamiania kontenera"
    exit 1
fi

echo ""
log "=== GOTOWE! ==="
log "Aplikacja działa na: http://localhost:5000"
log ""
log "Przydatne komendy:"
echo "  - Sprawdź status: docker ps"
echo "  - Zobacz logi: docker logs anime-app"
echo "  - Zatrzymaj: docker stop anime-app"
echo "  - Usuń kontener: docker rm anime-app"
echo "  - Wejdź do kontenera: docker exec -it anime-app bash"
echo ""

# Opcjonalnie otwórz przeglądarkę
read -p "Otworzyć aplikację w przeglądarce? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v xdg-open &> /dev/null; then
        xdg-open http://localhost:5000
    elif command -v open &> /dev/null; then
        open http://localhost:5000
    else
        log "Otwórz ręcznie: http://localhost:5000"
    fi
fi