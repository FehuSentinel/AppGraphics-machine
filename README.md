# 📊 Gestor de Tablas - Machine Learning

Aplicación web moderna para gestión de datos y machine learning con backend en FastAPI (Python) y frontend en React.

## 📦 Instalación

### Requisitos Previos

Antes de instalar, asegúrate de tener instalado:

- **Python 3.8 o superior** ([Descargar Python](https://www.python.org/downloads/))
  - En Windows: Marca la opción "Add Python to PATH" durante la instalación
- **Node.js 16 o superior** ([Descargar Node.js](https://nodejs.org/))
- **Git** (opcional, solo si clonas el repositorio)

### Pasos de Instalación

1. **Clonar o descargar el repositorio**
   ```bash
   git clone https://github.com/FehuSentinel/AppGraphics-machine.git
   cd AppGraphics-machine
   ```
   O descarga el ZIP y descomprímelo.

2. **Instalar dependencias del Backend**
   
   **Linux/Mac:**
   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
   
   **Windows:**
   ```cmd
   cd backend
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Instalar dependencias del Frontend**
   ```bash
   cd ../frontend
   npm install
   ```

## 🚀 Inicio Rápido

### Script Automático (Recomendado) ⭐

Los scripts **automatizan completamente** la instalación y configuración:

**Linux / Mac:**
```bash
./start.sh
```

**Windows:**
```cmd
start.bat
```
O hacer doble clic en `start.bat`

#### ¿Qué hace el script automáticamente?

✅ **Verifica requisitos previos** (Python, Node.js)  
✅ **Crea entorno virtual** si no existe  
✅ **Instala/actualiza dependencias** de Python automáticamente  
✅ **Instala/actualiza dependencias** de Node.js automáticamente  
✅ **Crea carpetas necesarias** (uploads)  
✅ **Inicia backend y frontend** automáticamente  

**Nota**: En Linux/Mac, si Node.js no está instalado, el script intentará instalarlo automáticamente. En Windows, debes instalarlo manualmente desde [nodejs.org](https://nodejs.org/).

### Opción 2: Inicio Manual

Si prefieres control manual:

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate  # Windows: venv\Scripts\activate
python app.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

### Acceso a la Aplicación

Una vez iniciada, la aplicación estará disponible en:
- **Frontend**: http://localhost:5173 (Vite)
- **Backend API**: http://localhost:8000
- **Documentación API**: http://localhost:8000/docs

## 📁 Estructura del Proyecto

```
AppTablas/
├── backend/              # API FastAPI
│   ├── app.py                 # Servidor principal
│   ├── modelos.db            # Base de datos SQLite
│   ├── requirements.txt       # Dependencias Python
│   └── uploads/              # Archivos subidos
│
├── frontend/                 # Aplicación React
│   ├── src/
│   │   ├── components/       # Componentes React
│   │   │   ├── DataTable.jsx      # Tabla editable
│   │   │   ├── MLControls.jsx     # Controles ML
│   │   │   ├── GraphPanel.jsx     # Panel de gráfico
│   │   │   ├── GraphView.jsx       # Visualización
│   │   │   ├── PredictionPanel.jsx # Panel de predicción
│   │   │   ├── ResizablePanels.jsx  # Paneles redimensionables
│   │   │   └── StatisticsPanel.jsx # Panel de estadísticas
│   │   ├── services/         # Servicios API
│   │   └── App.jsx
│   └── package.json
│
├── start.sh                   # Script de inicio (Linux/Mac)
└── start.bat                  # Script de inicio (Windows)
```

## ✨ Características

### 📥 Carga de Datos
- ✅ **CSV** con detección automática de encoding
- ✅ **Excel** (.xlsx, .xls) con soporte para múltiples hojas
- ✅ **Preprocesamiento automático** al cargar:
  - Eliminación de duplicados
  - Tratamiento de valores faltantes (mediana/moda)
  - Ajuste de outliers (método IQR, sin eliminar datos)
  - Eliminación de datos erróneos (inf, -inf)
  - Codificación de variables categóricas (LabelEncoder)
  - Creación de variables derivadas (multiplicaciones, divisiones, cuadrados, ratios)
  - Transformaciones logarítmicas (para variables altamente sesgadas)

### ✏️ Edición Interactiva (CRUD)
- ✅ **Edición de celdas** en tiempo real
- ✅ **Agregar/Eliminar filas** dinámicamente
- ✅ **Agregar/Eliminar columnas** dinámicamente
- ✅ **Actualización automática** de gráficos

### 📈 Visualización Inteligente
- ✅ **Gráficos interactivos** que se actualizan automáticamente
- ✅ **3 gráficos principales**:
  - **Datos antes del entrenamiento**: Visualización de la relación entre características y variable objetivo
  - **Real vs Predicho**: Comparación de valores reales vs predicciones del modelo
  - **Learning Curves**: Curvas de aprendizaje para detectar overfitting/underfitting
- ✅ **8 tipos de visualización**: Dispersión, Línea, Barras, Área, Pastel, Combinado, Radar, Treemap
- ✅ **Paneles redimensionables**: Ajusta el tamaño de los paneles según tus necesidades
- ✅ **Estadísticas descriptivas**: Panel con estadísticas post-procesamiento

### 🤖 Machine Learning
- ✅ **2 algoritmos principales**:
  - **Regresión Lineal Simple**: Selecciona automáticamente la mejor característica y entrena un modelo simple (y = a + b*x)
  - **Regresión Lineal Múltiple**: Usa múltiples características seleccionadas (y = a + b₁*x₁ + b₂*x₂ + ...)
- ✅ **Entrenamiento con métricas completas**:
  - R² Score (Train y Test)
  - RMSE (Train y Test)
  - MAE (Train y Test)
  - MAPE (Mean Absolute Percentage Error)
  - Análisis de overfitting (diferencia Train vs Test)
  - Learning Curves automáticas
- ✅ **Mejoras automáticas aplicadas**:
  - Eliminación de multicolinealidad (correlación > 0.95)
  - Selección automática de características (SelectKBest)
    - **Regresión Lineal Simple**: Selecciona solo 1 característica (la mejor)
    - **Regresión Lineal Múltiple**: Selecciona las mejores características
  - Normalización opcional (StandardScaler)
- ✅ **Predicción individual**: Panel interactivo para hacer predicciones con modelos entrenados
  - Selección de campos a modificar
  - Valores por defecto inteligentes basados en correlaciones
  - Sugerencias automáticas basadas en correlaciones entre variables
- ✅ **Guardado en SQLite** para persistencia
- ✅ **Validación robusta**: Verificación de variabilidad de datos, escalado correcto y coeficientes no nulos

## 🎨 Interfaz

### Layout con Paneles Redimensionables
- **Izquierda (2 paneles verticales redimensionables)**:
  - **Panel Superior**: Tabla de datos editable con estadísticas descriptivas
  - **Panel Inferior**: Gráficos interactivos (3 gráficos principales)
- **Derecha (Panel fijo)**:
  - **Panel de Controles ML**:
    1. Selección de Variables (Y objetivo, X características)
    2. Selección de Algoritmo (Regresión Lineal Simple o Múltiple)
    3. Configuración de entrenamiento (tamaño de test, normalización)
    4. Botón de entrenamiento
    5. Métricas del modelo entrenado
    6. Panel de predicción individual

### Flujo de Trabajo
1. **Cargar datos** → Preprocesamiento automático (duplicados, valores faltantes, outliers)
2. **Seleccionar variables** → Variable objetivo y características (las características aparecen desmarcadas por defecto)
3. **Visualizar relación** → Gráfico de datos antes del entrenamiento se actualiza automáticamente
4. **Configurar modelo** → Elegir algoritmo (Simple o Múltiple), tamaño de test, normalización
5. **Entrenar** → Ver métricas, learning curves automáticas
6. **Visualizar resultados** → 
   - Gráfico "Real vs Predicho" muestra la calidad de las predicciones
   - Gráfico "Learning Curves" muestra el aprendizaje del modelo
7. **Hacer predicciones** → 
   - Seleccionar campos a modificar
   - Ingresar valores (con sugerencias inteligentes basadas en correlaciones)
   - Ver predicción del modelo

## 🔧 Comandos Útiles

### Desarrollo

**Reinstalar dependencias del Backend:**
```bash
cd backend
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install --upgrade -r requirements.txt
```

**Reinstalar dependencias del Frontend:**
```bash
cd frontend
rm -rf node_modules  # Windows: rmdir /s node_modules
npm install
```

**Limpiar y reinstalar todo:**
```bash
# Backend
cd backend
rm -rf venv  # Windows: rmdir /s venv
python3 -m venv venv  # Windows: python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend
cd ../frontend
rm -rf node_modules  # Windows: rmdir /s node_modules
npm install
```

> **Nota**: En Windows, puedes usar `cmd` o PowerShell. Los comandos son los mismos.

## 🔧 Tecnologías

- **Backend**: FastAPI, pandas, scikit-learn, SQLite
- **Frontend**: React, Vite, Recharts, Axios
- **Base de Datos**: SQLite3 para persistencia de modelos
- **ML Libraries**: scikit-learn (LinearRegression, SelectKBest, StandardScaler)
- **UI/UX**: Paneles redimensionables, gráficos interactivos, diseño responsive

## 📝 API Endpoints

- `POST /api/upload/csv` - Cargar archivo CSV
- `POST /api/upload/excel` - Cargar archivo Excel
- `GET /api/data/{session_id}` - Obtener datos
- `PUT /api/data/{session_id}` - Actualizar celda
- `POST /api/data/{session_id}/row` - Agregar fila
- `DELETE /api/data/{session_id}/row/{index}` - Eliminar fila
- `POST /api/data/{session_id}/column` - Agregar columna
- `DELETE /api/data/{session_id}/column/{name}` - Eliminar columna
- `POST /api/model/train` - Entrenar modelo
- `POST /api/model/predict` - Hacer predicción con modelo entrenado
- `GET /api/models` - Listar modelos guardados (con comparación de métricas)
- `GET /api/data/{session_id}/statistics` - Estadísticas descriptivas
- `GET /api/data/{session_id}/correlations` - Matriz de correlaciones

Documentación completa en: http://localhost:8000/docs

## 💻 Compatibilidad

✅ **Soportado en:**
- ✅ Windows 10/11
- ✅ Linux (Ubuntu, Debian, etc.)
- ✅ macOS

**Requisitos:**
- Python 3.8+ (con pip)
- Node.js 16+ (con npm)
- Navegador web moderno (Chrome, Firefox, Edge, Safari)

**Scripts de inicio:**
- `start.sh` - Para Linux/Mac
- `start.bat` - Para Windows

## 📄 Licencia

Open Source
