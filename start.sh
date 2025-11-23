#!/bin/bash
# Script para iniciar la aplicación web (Backend Python + Frontend React)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Iniciando Gestor de Tablas Web..."
echo ""

# Colores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Función para limpiar procesos al salir
cleanup() {
    echo ""
    echo -e "${YELLOW}Deteniendo servidores...${NC}"
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit
}
trap cleanup INT TERM

# ========== BACKEND ==========
echo -e "${BLUE}📦 Configurando Backend...${NC}"
cd backend

# Crear venv si no existe
if [ ! -d "venv" ]; then
    echo "Creando entorno virtual..."
    python3 -m venv venv || {
        echo "Error: No se pudo crear el entorno virtual. Instalando python3-venv..."
        sudo apt-get update && sudo apt-get install -y python3-venv
        python3 -m venv venv
    }
fi

# Activar venv
source venv/bin/activate

# Instalar dependencias
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Instalando dependencias de Python..."
    pip install --upgrade pip > /dev/null 2>&1
    pip install -r requirements.txt
fi

# Crear carpeta uploads si no existe
mkdir -p uploads

# Iniciar backend
echo -e "${GREEN}✅ Iniciando Backend en http://localhost:8000${NC}"
python app.py &
BACKEND_PID=$!

# Esperar a que el backend inicie
sleep 3

# Verificar que el backend esté corriendo
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${YELLOW}Error: El backend no pudo iniciar${NC}"
    exit 1
fi

# ========== FRONTEND ==========
echo -e "${BLUE}⚛️  Configurando Frontend...${NC}"
cd ../frontend

# Verificar si Node.js está instalado
if ! command -v node &> /dev/null; then
    echo -e "${YELLOW}Node.js no está instalado. Instalando...${NC}"
    curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi

# Instalar dependencias si no existen
if [ ! -d "node_modules" ]; then
    echo "Instalando dependencias de Node.js..."
    npm install
fi

# Iniciar frontend
echo -e "${GREEN}✅ Iniciando Frontend en http://localhost:3000${NC}"
npm run dev &
FRONTEND_PID=$!

# Esperar un poco
sleep 2

echo ""
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Aplicación iniciada correctamente!${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo ""
echo -e "   ${BLUE}Backend:${NC}  http://localhost:8000"
echo -e "   ${BLUE}Frontend:${NC} http://localhost:3000"
echo -e "   ${BLUE}API Docs:${NC} http://localhost:8000/docs"
echo ""
echo -e "${YELLOW}Presiona Ctrl+C para detener ambos servidores${NC}"
echo ""

# Esperar a que el usuario presione Ctrl+C
wait
