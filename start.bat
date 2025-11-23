@echo off
REM Script para iniciar la aplicación web en Windows (Backend Python + Frontend React)

cd /d "%~dp0"

echo 🚀 Iniciando Gestor de Tablas Web...
echo.

REM ========== BACKEND ==========
echo 📦 Configurando Backend...
cd backend

REM Crear venv si no existe
if not exist "venv" (
    echo Creando entorno virtual...
    python -m venv venv
    if errorlevel 1 (
        echo Error: No se pudo crear el entorno virtual.
        echo Asegúrate de tener Python instalado.
        pause
        exit /b 1
    )
)

REM Activar venv
call venv\Scripts\activate.bat

REM Instalar dependencias
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo Instalando dependencias de Python...
    python -m pip install --upgrade pip >nul 2>&1
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Error al instalar dependencias.
        pause
        exit /b 1
    )
)

REM Crear carpeta uploads si no existe
if not exist "uploads" mkdir uploads

REM Iniciar backend
echo ✅ Iniciando Backend en http://localhost:8000
start "Backend - Gestor de Tablas" cmd /k "python app.py"

REM Esperar a que el backend inicie
timeout /t 3 /nobreak >nul

REM ========== FRONTEND ==========
echo.
echo ⚛️  Configurando Frontend...
cd ..\frontend

REM Verificar si Node.js está instalado
where node >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Node.js no está instalado.
    echo Por favor instala Node.js desde https://nodejs.org/
    pause
    exit /b 1
)

REM Instalar dependencias si no existen
if not exist "node_modules" (
    echo Instalando dependencias de Node.js...
    call npm install
    if errorlevel 1 (
        echo Error al instalar dependencias de Node.js.
        pause
        exit /b 1
    )
)

REM Iniciar frontend
echo ✅ Iniciando Frontend en http://localhost:3000
start "Frontend - Gestor de Tablas" cmd /k "npm run dev"

REM Esperar un poco
timeout /t 2 /nobreak >nul

echo.
echo ════════════════════════════════════
echo ✅ Aplicación iniciada correctamente!
echo ════════════════════════════════════
echo.
echo    Backend:  http://localhost:8000
echo    Frontend: http://localhost:3000
echo    API Docs: http://localhost:8000/docs
echo.
echo Presiona cualquier tecla para cerrar esta ventana...
echo (Los servidores seguirán corriendo en sus propias ventanas)
pause >nul

