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

Una vez instalado, puedes iniciar la aplicación de dos formas:

### Opción 1: Script Automático (Recomendado)

**Linux / Mac:**
```bash
./start.sh
```

**Windows:**
```cmd
start.bat
```
O hacer doble clic en `start.bat`

### Opción 2: Inicio Manual

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
- **Frontend**: http://localhost:3000
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
│   │   │   ├── DataTable.jsx  # Tabla editable
│   │   │   ├── MLControls.jsx # Controles ML
│   │   │   ├── GraphPanel.jsx # Panel de gráfico
│   │   │   └── GraphView.jsx  # Visualización
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
- ✅ **8 tipos de gráficos**: Dispersión, Línea, Barras, Área, Pastel, Combinado, Radar, Treemap
- ✅ **Visualización automática** de la relación entre características seleccionadas y variable objetivo
- ✅ **Visualización de resultados** del modelo entrenado (Real vs Predicho)
- ✅ **Learning Curves**: Gráfico automático de curvas de aprendizaje para detectar overfitting/underfitting
- ✅ **Gráfico de residuos**: Validación de supuestos de regresión (normalidad, homocedasticidad)
- ✅ **Descarga de gráficos** como imagen PNG

### 🤖 Machine Learning
- ✅ **8 algoritmos disponibles**:
  - Regresión Lineal Simple
  - Regresión Lineal Múltiple
  - Ridge Regression
  - Lasso Regression
  - Random Forest
  - Gradient Boosting
  - **XGBoost** ⭐ (Recomendado - mejor rendimiento)
  - Decision Tree
- ✅ **Entrenamiento con métricas completas**:
  - R² Score (Train y Test)
  - RMSE (Train y Test)
  - MAE (Train y Test)
  - MAPE (Mean Absolute Percentage Error)
  - Análisis de overfitting (diferencia Train vs Test)
  - Cross-Validation (opcional)
  - Feature Importance (para modelos basados en árboles)
- ✅ **Mejoras automáticas aplicadas**:
  - Eliminación de multicolinealidad (VIF)
  - Selección automática de características (SelectKBest)
  - Variables derivadas (multiplicaciones, divisiones, cuadrados, ratios)
  - Transformaciones logarítmicas (para variables sesgadas)
  - Características polinomiales (opcional, para modelos lineales)
- ✅ **Learning Curves**: Visualización automática de curvas de aprendizaje
- ✅ **Predicción de nuevos valores**: Panel interactivo para hacer predicciones con modelos entrenados
- ✅ **Guardado en SQLite** para persistencia
- ✅ **Visualización automática** de datos antes del entrenamiento

## 🎨 Interfaz

### Layout (70% / 30%)
- **Izquierda (70%)**:
  - **Arriba**: Tabla de datos editable
  - **Abajo**: Gráfico interactivo
- **Derecha (30%)**:
  - **Panel continuo** con todos los ajustes:
    1. Selección de Variables (Y objetivo, X características, algoritmo, entrenar)
    2. Configuración de Visualización (tipo de gráfico)

### Flujo de Trabajo
1. **Cargar datos** → Preprocesamiento automático (duplicados, valores faltantes, outliers, variables derivadas)
2. **Seleccionar variables** → Variable objetivo y características
3. **Visualizar relación** → Gráfico se actualiza automáticamente
4. **Configurar modelo** → Elegir algoritmo, división train/test, opciones avanzadas
5. **Entrenar** → Ver métricas, learning curves, feature importance
6. **Visualizar resultados** → Gráfico cambia a "Real vs Predicho" o "Learning Curves"
7. **Hacer predicciones** → Usar el panel de predicción con nuevos valores

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

- **Backend**: FastAPI, pandas, scikit-learn, XGBoost, statsmodels, SQLite
- **Frontend**: React, Vite, Recharts, Axios
- **Base de Datos**: SQLite3 para persistencia de modelos
- **ML Libraries**: scikit-learn, XGBoost, statsmodels (VIF, Shapiro-Wilk)

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
