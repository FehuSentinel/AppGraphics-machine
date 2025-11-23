#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backend API para Gestión de Tablas con Machine Learning
FastAPI + SQLite
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import sqlite3
import json
import os
import pickle
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost no disponible. Instala con: pip install xgboost")
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sqlalchemy import create_engine, inspect
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Gestor de Tablas API", version="1.0.0")

# CORS para permitir conexiones desde React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directorio para archivos subidos
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Base de datos
DB_PATH = os.path.join(os.path.dirname(__file__), "modelos.db")

# Almacenamiento en memoria de datos actuales
current_data_store: Dict[str, pd.DataFrame] = {}
processed_data_store: Dict[str, pd.DataFrame] = {}
original_data_store: Dict[str, pd.DataFrame] = {}


# ========== MODELOS PYDANTIC ==========

class DataUpdate(BaseModel):
    row: int
    col: int
    value: Any

class DataRow(BaseModel):
    data: Dict[str, Any]

class ModelTrainRequest(BaseModel):
    session_id: str
    algorithm: str
    target_column: str
    features: List[str]
    test_size: float = 0.2
    random_state: int = 42
    normalize: bool = False
    auto_feature_selection: bool = True  # Selección automática de features
    remove_multicollinearity: bool = True  # Eliminar multicolinealidad
    use_polynomial_features: bool = False  # Features polinomiales (solo regresión lineal)


# ========== INICIALIZACIÓN DE BASE DE DATOS ==========

def init_database():
    """Inicializa la base de datos SQLite y migra la tabla si es necesario"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Verificar si la tabla existe
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='modelos_ml'")
    table_exists = cursor.fetchone() is not None
    
    # Columnas esenciales que DEBEN existir
    essential_columns = ['nombre_modelo', 'algoritmo', 'fecha_creacion', 'caracteristicas', 
                        'variable_objetivo', 'metricas', 'modelo_serializado']
    
    if not table_exists:
        # Crear tabla nueva con estructura completa
        cursor.execute("""
            CREATE TABLE modelos_ml (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre_modelo TEXT NOT NULL,
                algoritmo TEXT NOT NULL,
                fecha_creacion TEXT NOT NULL,
                caracteristicas TEXT NOT NULL,
                variable_objetivo TEXT NOT NULL,
                metricas TEXT NOT NULL,
                modelo_serializado BLOB NOT NULL,
                scaler_serializado BLOB,
                r2_train REAL,
                r2_test REAL,
                rmse_train REAL,
                rmse_test REAL,
                mae_train REAL,
                mae_test REAL
            )
        """)
        print("✅ Tabla modelos_ml creada con estructura completa")
    else:
        # Tabla existe, verificar estructura
        cursor.execute("PRAGMA table_info(modelos_ml)")
        existing_columns = [row[1] for row in cursor.fetchall()]
        
        # Verificar si faltan columnas esenciales
        missing_essential = [col for col in essential_columns if col not in existing_columns]
        
        if missing_essential:
            # Si faltan columnas esenciales, recrear la tabla
            print(f"⚠️ Faltan columnas esenciales: {missing_essential}")
            print("🔄 Recreando tabla modelos_ml con estructura completa...")
            
            # Backup de datos existentes (solo si hay datos)
            cursor.execute("SELECT COUNT(*) FROM modelos_ml")
            count = cursor.fetchone()[0]
            
            if count > 0:
                print(f"⚠️ La tabla tiene {count} registros. Se perderán al recrear la tabla.")
                print("💡 Si necesitas conservar los modelos, haz un backup manual de modelos.db")
            
            # Eliminar tabla antigua
            cursor.execute("DROP TABLE IF EXISTS modelos_ml")
            
            # Crear tabla nueva con estructura completa
            cursor.execute("""
                CREATE TABLE modelos_ml (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nombre_modelo TEXT NOT NULL,
                    algoritmo TEXT NOT NULL,
                    fecha_creacion TEXT NOT NULL,
                    caracteristicas TEXT NOT NULL,
                    variable_objetivo TEXT NOT NULL,
                    metricas TEXT NOT NULL,
                    modelo_serializado BLOB NOT NULL,
                    scaler_serializado BLOB,
                    r2_train REAL,
                    r2_test REAL,
                    rmse_train REAL,
                    rmse_test REAL,
                    mae_train REAL,
                    mae_test REAL
                )
            """)
            print("✅ Tabla modelos_ml recreada con estructura completa")
        else:
            # Solo faltan columnas opcionales, agregarlas
            optional_columns = {
                'scaler_serializado': 'BLOB',
                'r2_train': 'REAL',
                'r2_test': 'REAL',
                'rmse_train': 'REAL',
                'rmse_test': 'REAL',
                'mae_train': 'REAL',
                'mae_test': 'REAL'
            }
            
            columns_added = 0
            for col_name, col_type in optional_columns.items():
                if col_name not in existing_columns:
                    try:
                        cursor.execute(f"ALTER TABLE modelos_ml ADD COLUMN {col_name} {col_type}")
                        print(f"✅ Columna '{col_name}' agregada")
                        columns_added += 1
                    except sqlite3.OperationalError as e:
                        print(f"⚠️ No se pudo agregar columna '{col_name}': {e}")
            
            if columns_added > 0:
                print(f"✅ Migración completada: {columns_added} columnas agregadas")
            else:
                print("✅ Tabla modelos_ml tiene todas las columnas necesarias")
    
    conn.commit()
    conn.close()

init_database()


# ========== FUNCIONES DE PREPROCESAMIENTO ==========

def auto_preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesamiento automático de datos sin romper los datos
    - Elimina duplicados
    - Trata valores faltantes
    - Maneja outliers
    - Elimina datos erróneos
    - Codifica variables categóricas
    """
    data = df.copy()
    
    # 0. Eliminar duplicados (primero, antes de cualquier otro procesamiento)
    n_before = len(data)
    data = data.drop_duplicates()
    n_duplicates = n_before - len(data)
    if n_duplicates > 0:
        print(f"✅ Eliminados {n_duplicates} registros duplicados ({n_duplicates/n_before*100:.1f}% del total)")
    
    # 1. Tratar valores faltantes
    for col in data.columns:
        if data[col].isnull().any():
            if pd.api.types.is_numeric_dtype(data[col]):
                # Numéricas: usar mediana
                data[col].fillna(data[col].median(), inplace=True)
            else:
                # Categóricas: usar moda o "Desconocido"
                mode_val = data[col].mode()
                if len(mode_val) > 0:
                    data[col].fillna(mode_val[0], inplace=True)
                else:
                    data[col].fillna("Desconocido", inplace=True)
    
    # 2. Tratar outliers (solo numéricas, sin eliminar, solo ajustar)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if data[col].dtype in [np.float64, np.int64]:
            # Método IQR: ajustar valores extremos sin eliminar
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers en lugar de eliminarlos
            data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
    
    # 3. Eliminar datos erróneos (inf, -inf)
    data = data.replace([np.inf, -np.inf], np.nan)
    # Rellenar los NaN que quedaron después de reemplazar inf
    for col in data.columns:
        if data[col].isnull().any():
            if pd.api.types.is_numeric_dtype(data[col]):
                data[col].fillna(data[col].median(), inplace=True)
            else:
                mode_val = data[col].mode()
                if len(mode_val) > 0:
                    data[col].fillna(mode_val[0], inplace=True)
                else:
                    data[col].fillna("Desconocido", inplace=True)
    
    # 4. Codificar variables categóricas (solo si es necesario para ML)
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le
    
    # 5. Crear variables derivadas automáticamente (feature engineering)
    data = create_derived_variables(data)
    
    # 6. Aplicar transformaciones logarítmicas a variables sesgadas
    data = apply_log_transformations(data)
    
    return data


def create_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables derivadas automáticamente basadas en patrones comunes
    - Multiplicaciones: precio × cantidad → ventas_total
    - Divisiones: precio / cantidad → precio_unitario
    - Cuadrados: variable²
    - Sumas: variable1 + variable2
    """
    data = df.copy()
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return data  # No hay suficientes columnas numéricas
    
    new_vars_created = []
    
    # Patrones comunes de nombres de columnas para detectar relaciones
    price_keywords = ['precio', 'price', 'cost', 'costo', 'valor', 'value']
    quantity_keywords = ['cantidad', 'quantity', 'qty', 'volumen', 'volume', 'unidades', 'units']
    sales_keywords = ['ventas', 'sales', 'total', 'ingresos', 'revenue']
    time_keywords = ['tiempo', 'time', 'duracion', 'duration', 'horas', 'hours']
    
    # 1. Detectar y crear multiplicaciones (precio × cantidad → ventas)
    price_cols = [col for col in numeric_cols if any(kw in col.lower() for kw in price_keywords)]
    qty_cols = [col for col in numeric_cols if any(kw in col.lower() for kw in quantity_keywords)]
    
    for price_col in price_cols:
        for qty_col in qty_cols:
            if price_col != qty_col:
                new_var_name = f"{price_col}_x_{qty_col}"
                if new_var_name not in data.columns:
                    # Evitar división por cero
                    data[new_var_name] = data[price_col] * data[qty_col]
                    new_vars_created.append(new_var_name)
                    print(f"✅ Variable derivada creada: {new_var_name} = {price_col} × {qty_col}")
    
    # 2. Crear divisiones (precio / cantidad → precio_unitario)
    for price_col in price_cols:
        for qty_col in qty_cols:
            if price_col != qty_col:
                new_var_name = f"{price_col}_div_{qty_col}"
                if new_var_name not in data.columns:
                    # Evitar división por cero
                    mask = data[qty_col] != 0
                    data[new_var_name] = np.where(mask, data[price_col] / data[qty_col], np.nan)
                    # Rellenar NaN con mediana
                    if data[new_var_name].isna().any():
                        data[new_var_name].fillna(data[new_var_name].median(), inplace=True)
                    new_vars_created.append(new_var_name)
                    print(f"✅ Variable derivada creada: {new_var_name} = {price_col} / {qty_col}")
    
    # 3. Crear cuadrados de variables importantes (solo para las primeras 5 numéricas para evitar demasiadas)
    for col in numeric_cols[:5]:
        if col not in new_vars_created:  # No crear cuadrados de variables ya derivadas
            new_var_name = f"{col}_squared"
            if new_var_name not in data.columns:
                data[new_var_name] = data[col] ** 2
                new_vars_created.append(new_var_name)
                print(f"✅ Variable derivada creada: {new_var_name} = {col}²")
    
    # 4. Crear sumas de variables relacionadas (solo si hay al menos 2 numéricas)
    if len(numeric_cols) >= 2:
        # Sumar las primeras 2-3 variables numéricas si tienen nombres relacionados
        for i in range(min(2, len(numeric_cols))):
            for j in range(i+1, min(i+2, len(numeric_cols))):
                col1 = numeric_cols[i]
                col2 = numeric_cols[j]
                if col1 != col2:
                    new_var_name = f"{col1}_plus_{col2}"
                    if new_var_name not in data.columns and new_var_name not in new_vars_created:
                        data[new_var_name] = data[col1] + data[col2]
                        new_vars_created.append(new_var_name)
                        print(f"✅ Variable derivada creada: {new_var_name} = {col1} + {col2}")
    
    # 5. Crear ratios generales (solo para variables con alta varianza)
    if len(numeric_cols) >= 2:
        for i in range(min(3, len(numeric_cols))):
            for j in range(i+1, min(i+3, len(numeric_cols))):
                col1 = numeric_cols[i]
                col2 = numeric_cols[j]
                if col1 != col2:
                    # Solo crear ratio si ambas variables tienen varianza significativa
                    if data[col1].std() > 0 and data[col2].std() > 0:
                        new_var_name = f"{col1}_ratio_{col2}"
                        if new_var_name not in data.columns and new_var_name not in new_vars_created:
                            mask = data[col2] != 0
                            data[new_var_name] = np.where(mask, data[col1] / (data[col2] + 1e-10), np.nan)
                            if data[new_var_name].isna().any():
                                data[new_var_name].fillna(data[new_var_name].median(), inplace=True)
                            new_vars_created.append(new_var_name)
                            print(f"✅ Variable derivada creada: {new_var_name} = {col1} / {col2}")
    
    if new_vars_created:
        print(f"📊 Total de variables derivadas creadas: {len(new_vars_created)}")
    else:
        print("ℹ️ No se crearon variables derivadas (no se detectaron patrones comunes)")
    
    return data


def apply_log_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica transformaciones logarítmicas a variables con alta asimetría (skewness)
    Esto mejora el rendimiento de modelos lineales con datos sesgados
    """
    data = df.copy()
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        return data
    
    transformed_vars = []
    
    for col in numeric_cols:
        # Solo transformar si todos los valores son positivos
        if (data[col] > 0).all() and data[col].nunique() > 10:  # Al menos 10 valores únicos
            # Calcular asimetría (skewness)
            skewness = stats.skew(data[col].dropna())
            
            # Si la asimetría es alta (> 1.5 o < -1.5), aplicar log
            if abs(skewness) > 1.5:
                new_var_name = f"{col}_log"
                if new_var_name not in data.columns:
                    # Log(x + 1) para evitar problemas con valores muy pequeños
                    data[new_var_name] = np.log1p(data[col])
                    transformed_vars.append(new_var_name)
                    print(f"✅ Transformación log aplicada: {new_var_name} (skewness={skewness:.2f})")
    
    if transformed_vars:
        print(f"📊 Total de transformaciones log aplicadas: {len(transformed_vars)}")
    
    return data


def remove_multicollinearity(X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """
    Elimina features altamente correlacionadas (multicolinealidad)
    Mantiene la feature con mayor correlación con el target si se proporciona
    """
    if X.empty or len(X.columns) < 2:
        return X
    
    # Calcular matriz de correlación
    corr_matrix = X.corr().abs()
    
    # Encontrar pares de features altamente correlacionadas
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Identificar features a eliminar
    to_remove = [column for column in upper_triangle.columns 
                 if any(upper_triangle[column] > threshold)]
    
    if to_remove:
        X_cleaned = X.drop(columns=to_remove)
        print(f"✅ Eliminadas {len(to_remove)} features por multicolinealidad: {to_remove}")
        return X_cleaned
    
    return X


def select_best_features(X: pd.DataFrame, y: pd.Series, k: Optional[int] = None, method: str = 'f_regression') -> tuple:
    """
    Selecciona las mejores features usando SelectKBest
    Si k no se especifica, selecciona automáticamente el 80% de las features
    """
    if X.empty or len(X.columns) < 2:
        return X, []
    
    # Determinar k automáticamente si no se especifica
    if k is None:
        k = max(1, int(len(X.columns) * 0.8))  # 80% de las features
    else:
        k = min(k, len(X.columns))
    
    # Seleccionar método de selección
    if method == 'f_regression':
        selector = SelectKBest(score_func=f_regression, k=k)
    elif method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
    else:
        return X, []
    
    try:
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        feature_scores = selector.scores_
        
        if len(selected_features) < len(X.columns):
            print(f"✅ Feature selection: {len(selected_features)}/{len(X.columns)} features seleccionadas")
            # Mostrar top features
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'score': feature_scores
            }).sort_values('score', ascending=False)
            print(f"   Top 5 features: {feature_importance.head(5)['feature'].tolist()}")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features
    except Exception as e:
        print(f"⚠️ Error en feature selection: {e}")
        return X, []


# ========== ENDPOINTS ==========

@app.get("/")
def root():
    return {"message": "Gestor de Tablas API", "version": "1.0.0"}


@app.post("/api/upload/csv")
async def upload_csv(file: UploadFile = File(...)):
    """Carga un archivo CSV"""
    try:
        session_id = f"session_{datetime.now().timestamp()}"
        file_path = os.path.join(UPLOAD_DIR, f"{session_id}_{file.filename}")
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Leer CSV
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # Guardar original
        original_data_store[session_id] = df.copy()
        
        # Preprocesamiento automático
        processed_df = auto_preprocess_data(df)
        
        # Guardar procesado y actual
        processed_data_store[session_id] = processed_df.copy()
        current_data_store[session_id] = processed_df.copy()
        
        return {
            "session_id": session_id,
            "rows": len(df),
            "columns": len(df.columns),
            "columns_list": df.columns.tolist(),
            "data": processed_df.head(100).to_dict(orient='records'),
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/upload/excel")
async def upload_excel(file: UploadFile = File(...), sheet_name: Optional[str] = None):
    """Carga un archivo Excel"""
    try:
        session_id = f"session_{datetime.now().timestamp()}"
        file_path = os.path.join(UPLOAD_DIR, f"{session_id}_{file.filename}")
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Leer Excel
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(file_path)
        
        # Guardar original
        original_data_store[session_id] = df.copy()
        
        # Preprocesamiento automático
        processed_df = auto_preprocess_data(df)
        
        # Guardar procesado y actual
        processed_data_store[session_id] = processed_df.copy()
        current_data_store[session_id] = processed_df.copy()
        
        return {
            "session_id": session_id,
            "rows": len(df),
            "columns": len(df.columns),
            "columns_list": df.columns.tolist(),
            "data": processed_df.head(100).to_dict(orient='records'),
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/data/{session_id}")
def get_data(session_id: str, limit: int = 1000):
    """Obtiene los datos de una sesión"""
    if session_id not in current_data_store:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    
    df = current_data_store[session_id]
    
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "columns_list": df.columns.tolist(),
        "data": df.head(limit).to_dict(orient='records')
    }


@app.put("/api/data/{session_id}")
def update_data(session_id: str, update: DataUpdate):
    """Actualiza un valor en los datos"""
    if session_id not in current_data_store:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    
    df = current_data_store[session_id]
    col_name = df.columns[update.col]
    
    try:
        # Intentar convertir al tipo original
        original_dtype = df[col_name].dtype
        if pd.api.types.is_numeric_dtype(original_dtype):
            df.iloc[update.row, update.col] = pd.to_numeric(update.value, errors='coerce')
        else:
            df.iloc[update.row, update.col] = str(update.value)
        
        # Re-aplicar preprocesamiento automático
        processed_df = auto_preprocess_data(df)
        current_data_store[session_id] = processed_df
        
        return {"success": True, "value": processed_df.iloc[update.row, update.col]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/data/{session_id}/row")
def add_row(session_id: str, row: DataRow):
    """Agrega una nueva fila"""
    if session_id not in current_data_store:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    
    df = current_data_store[session_id]
    new_row = pd.DataFrame([row.data])
    df = pd.concat([df, new_row], ignore_index=True)
    
    # Re-aplicar preprocesamiento
    processed_df = auto_preprocess_data(df)
    current_data_store[session_id] = processed_df
    
    return {"success": True, "rows": len(processed_df)}


@app.delete("/api/data/{session_id}/row/{row_index}")
def delete_row(session_id: str, row_index: int):
    """Elimina una fila"""
    if session_id not in current_data_store:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    
    df = current_data_store[session_id]
    df = df.drop(df.index[row_index]).reset_index(drop=True)
    current_data_store[session_id] = df
    
    return {"success": True, "rows": len(df)}


@app.post("/api/data/{session_id}/column")
def add_column(session_id: str, request: Dict[str, str]):
    """Agrega una nueva columna"""
    if session_id not in current_data_store:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    
    column_name = request.get("column_name")
    if not column_name:
        raise HTTPException(status_code=400, detail="Nombre de columna requerido")
    
    df = current_data_store[session_id]
    df[column_name] = 0  # Valor por defecto
    current_data_store[session_id] = df
    
    return {"success": True, "columns": df.columns.tolist()}


@app.delete("/api/data/{session_id}/column/{column_name}")
def delete_column(session_id: str, column_name: str):
    """Elimina una columna"""
    if session_id not in current_data_store:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    
    df = current_data_store[session_id]
    df = df.drop(columns=[column_name])
    current_data_store[session_id] = df
    
    return {"success": True, "columns": df.columns.tolist()}


@app.get("/api/data/{session_id}/statistics")
def get_statistics(session_id: str):
    """Obtiene estadísticas descriptivas de los datos (similar a pandas.describe())"""
    if session_id not in current_data_store:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    
    df = current_data_store[session_id]
    
    # Calcular estadísticas solo para columnas numéricas
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return {"statistics": {}}
    
    # Calcular estadísticas descriptivas
    stats = numeric_df.describe().to_dict()
    
    # Agregar información adicional
    result = {}
    for col in numeric_df.columns:
        result[col] = {
            "count": float(stats[col].get("count", 0)),
            "mean": float(stats[col].get("mean", 0)),
            "std": float(stats[col].get("std", 0)),
            "min": float(stats[col].get("min", 0)),
            "25%": float(stats[col].get("25%", 0)),
            "50%": float(stats[col].get("50%", 0)),  # mediana
            "75%": float(stats[col].get("75%", 0)),
            "max": float(stats[col].get("max", 0))
        }
    
    return {"statistics": result}


@app.get("/api/data/{session_id}/correlations")
def get_correlations(session_id: str, target_column: Optional[str] = None):
    """Obtiene correlaciones entre variables numéricas"""
    if session_id not in current_data_store:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    
    df = current_data_store[session_id]
    
    # Calcular correlaciones solo para columnas numéricas
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty or len(numeric_df.columns) < 2:
        return {"correlations": {}, "target_correlations": {}}
    
    # Matriz de correlación completa
    corr_matrix = numeric_df.corr()
    
    # Convertir a diccionario
    correlations = {}
    for col1 in corr_matrix.columns:
        correlations[col1] = {}
        for col2 in corr_matrix.columns:
            correlations[col1][col2] = float(corr_matrix.loc[col1, col2]) if not np.isnan(corr_matrix.loc[col1, col2]) else 0.0
    
    # Si se especifica target_column, devolver correlaciones con esa columna
    target_correlations = {}
    if target_column and target_column in numeric_df.columns:
        target_correlations = {
            col: float(corr_matrix.loc[target_column, col]) 
            for col in corr_matrix.columns 
            if col != target_column and not np.isnan(corr_matrix.loc[target_column, col])
        }
        # Ordenar por valor absoluto de correlación (mayor a menor)
        target_correlations = dict(sorted(
            target_correlations.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        ))
    
    return {
        "correlations": correlations,
        "target_correlations": target_correlations
    }


@app.post("/api/model/train")
def train_model(request: ModelTrainRequest):
    """Entrena un modelo de machine learning"""
    if request.session_id not in current_data_store:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    
    df = current_data_store[request.session_id].copy()
    
    try:
        # Validar que las columnas existan
        if not request.features or len(request.features) == 0:
            raise ValueError("Debe seleccionar al menos una característica")
        
        missing_features = [f for f in request.features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Columnas no encontradas: {missing_features}")
        
        if not request.target_column:
            raise ValueError("Debe seleccionar una variable objetivo")
        
        if request.target_column not in df.columns:
            raise ValueError(f"Columna objetivo no encontrada: {request.target_column}")
        
        # Verificar que target no esté en features
        if request.target_column in request.features:
            raise ValueError(f"La variable objetivo '{request.target_column}' no puede estar en las características")
        
        # Preparar datos - asegurar que sean numéricos
        X = df[request.features].copy()
        y = df[request.target_column].copy()
        
        # Convertir a numérico si es necesario
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        if not pd.api.types.is_numeric_dtype(y):
            y = pd.to_numeric(y, errors='coerce')
        
        # Eliminar filas con NaN
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            raise ValueError("No hay datos válidos después de la limpieza. Verifique que las columnas seleccionadas contengan valores numéricos válidos.")
        
        # MEJORA 1: Eliminar multicolinealidad (features altamente correlacionadas)
        selected_features_list = []
        if request.remove_multicollinearity and len(X.columns) > 1:
            X = remove_multicollinearity(X, threshold=0.95)
        
        # MEJORA 2: Feature selection automática (seleccionar mejores features)
        # Nota: Se hace después de eliminar multicolinealidad pero antes del split
        if request.auto_feature_selection and len(X.columns) > 3:
            X_selected, selected_features_list = select_best_features(X, y, method='f_regression')
            if selected_features_list and len(selected_features_list) < len(X.columns):
                # Aplicar selección a todo X
                X = X_selected
                print(f"✅ Feature selection aplicada: {len(selected_features_list)}/{len(X.columns)} features seleccionadas")
            else:
                selected_features_list = X.columns.tolist()
        
        # Validar que haya suficientes datos para train/test split
        min_samples = max(2, int(1 / request.test_size) + 1)  # Mínimo para tener al menos 1 muestra en test
        if len(X) < min_samples:
            raise ValueError(f"Se necesitan al menos {min_samples} muestras válidas para entrenar. Actualmente hay {len(X)} muestras válidas después de filtrar NaN.")
        
        # Validar que test_size sea válido
        if request.test_size <= 0 or request.test_size >= 1:
            raise ValueError(f"test_size debe estar entre 0 y 1. Valor recibido: {request.test_size}")
        
        # MEJORA 3: Polynomial features para regresión lineal (opcional)
        polynomial_transformer = None
        if request.use_polynomial_features and request.algorithm in ["Regresión Lineal Simple", "Regresión Lineal Múltiple", "Ridge Regression", "Lasso Regression"]:
            if len(X.columns) <= 5:  # Solo si hay pocas features (evitar explosión combinatoria)
                polynomial_transformer = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
                X_poly = polynomial_transformer.fit_transform(X)
                feature_names = polynomial_transformer.get_feature_names_out(X.columns)
                X = pd.DataFrame(X_poly, columns=feature_names, index=X.index)
                print(f"✅ Polynomial features creadas: {X.shape[1]} features (incluye interacciones)")
        
        # Normalización si se solicita
        scaler = None
        if request.normalize:
            scaler = StandardScaler()
            X_train_temp, X_test_temp, y_train, y_test = train_test_split(
                X, y, test_size=request.test_size, random_state=request.random_state
            )
            X_train = pd.DataFrame(
                scaler.fit_transform(X_train_temp),
                columns=X_train_temp.columns,
                index=X_train_temp.index
            )
            X_test = pd.DataFrame(
                scaler.transform(X_test_temp),
                columns=X_test_temp.columns,
                index=X_test_temp.index
            )
        else:
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=request.test_size, random_state=request.random_state
            )
        
        # Crear modelo con hiperparámetros mejorados
        if request.algorithm == "Regresión Lineal Simple":
            model = LinearRegression()
        elif request.algorithm == "Regresión Lineal Múltiple":
            model = LinearRegression()
        elif request.algorithm == "Ridge Regression":
            # Ajustar alpha según el tamaño de los datos
            alpha = max(0.1, min(10.0, len(X) / 100))
            model = Ridge(alpha=alpha, solver='auto')
        elif request.algorithm == "Lasso Regression":
            # Ajustar alpha según el tamaño de los datos
            alpha = max(0.01, min(1.0, len(X) / 1000))
            model = Lasso(alpha=alpha, max_iter=2000)
        elif request.algorithm == "Random Forest":
            # Hiperparámetros mejorados
            n_estimators = min(200, max(50, len(X) // 10))
            max_depth = min(20, max(5, int(np.log2(len(X)))))
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif request.algorithm == "Gradient Boosting":
            # Hiperparámetros mejorados con early stopping
            n_estimators = min(200, max(50, len(X) // 10))
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                subsample=0.8
            )
        elif request.algorithm == "Decision Tree":
            max_depth = min(20, max(5, int(np.log2(len(X)))))
            model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif request.algorithm == "XGBoost":
            if not XGBOOST_AVAILABLE:
                raise ValueError("XGBoost no está instalado. Instala con: pip install xgboost")
            # Hiperparámetros optimizados para XGBoost
            n_estimators = min(200, max(50, len(X) // 10))
            max_depth = min(10, max(3, int(np.log2(len(X)))))
            model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Algoritmo no soportado: {request.algorithm}")
        
        # Validar dimensiones antes de entrenar
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError(f"No hay suficientes datos para entrenar. Train: {len(X_train)} muestras")
        
        if len(X_train) != len(y_train):
            raise ValueError(f"Dimensiones inconsistentes: X_train tiene {len(X_train)} filas pero y_train tiene {len(y_train)}")
        
        # MEJORA: Learning curves (calcular error en diferentes tamaños de entrenamiento)
        learning_curve_data = []
        if len(X_train) >= 20:  # Solo si hay suficientes datos
            try:
                train_sizes = np.linspace(0.1, 1.0, min(10, len(X_train)//10))
                for size in train_sizes:
                    n_samples = max(1, int(len(X_train) * size))
                    X_train_subset = X_train.iloc[:n_samples]
                    y_train_subset = y_train.iloc[:n_samples]
                    
                    # Crear y entrenar modelo temporal
                    temp_model = type(model)(**model.get_params())
                    temp_model.fit(X_train_subset, y_train_subset)
                    
                    # Predecir en train y test
                    train_pred = temp_model.predict(X_train_subset)
                    test_pred = temp_model.predict(X_test)
                    
                    train_error = mean_squared_error(y_train_subset, train_pred)
                    test_error = mean_squared_error(y_test, test_pred)
                    
                    learning_curve_data.append({
                        "train_size": n_samples,
                        "train_error": float(np.sqrt(train_error)),
                        "test_error": float(np.sqrt(test_error))
                    })
            except Exception as e:
                print(f"⚠️ Error calculando learning curves: {e}")
                learning_curve_data = []
        
        # Entrenar modelo final
        model.fit(X_train, y_train)
        
        # Predecir
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Validar que las predicciones tengan el mismo tamaño
        if len(y_test_pred) != len(y_test):
            raise ValueError(f"Error en predicciones: y_test_pred tiene {len(y_test_pred)} valores pero y_test tiene {len(y_test)}")
        
        # Métricas (siguiendo el ejemplo de predicciónvalorauto_metricas.py)
        # MSE (Mean Squared Error)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        # RMSE (Root Mean Squared Error) - raíz de MSE
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        
        # R² Score (coeficiente de determinación)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # MAE (Mean Absolute Error)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Métricas adicionales para mejor evaluación
        # MAPE (Mean Absolute Percentage Error) - si todos los valores son positivos
        if (y_test > 0).all():
            mape_test = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
        else:
            mape_test = None
        
        # Error porcentual promedio
        mean_y_test = np.mean(y_test)
        if mean_y_test != 0:
            error_percentual = (test_rmse / abs(mean_y_test)) * 100
        else:
            error_percentual = None
        
        # Diferencia entre train y test (para detectar overfitting)
        r2_diff = train_r2 - test_r2
        rmse_diff = test_rmse - train_rmse
        
        # Calcular residuos para validación de supuestos (solo para regresión lineal)
        residuals_test = y_test - y_test_pred
        residuals_train = y_train - y_train_pred
        
        # Estadísticas de residuos
        residuals_stats = {
            'mean': float(np.mean(residuals_test)),
            'std': float(np.std(residuals_test)),
            'shapiro_stat': None,
            'shapiro_pvalue': None,
            'normality_test': 'No aplicable'
        }
        
        # Test de normalidad de Shapiro-Wilk (solo si hay suficientes datos)
        if len(residuals_test) >= 3 and len(residuals_test) <= 5000:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(residuals_test)
                residuals_stats['shapiro_stat'] = float(shapiro_stat)
                residuals_stats['shapiro_pvalue'] = float(shapiro_p)
                residuals_stats['normality_test'] = 'Normal' if shapiro_p > 0.05 else 'No normal'
            except:
                pass
        
        # Guardar en SQLite
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        model_bytes = pickle.dumps(model)
        nombre_modelo = f"{request.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        cursor.execute("""
            INSERT INTO modelos_ml 
            (nombre_modelo, algoritmo, fecha_creacion, caracteristicas, variable_objetivo,
             metricas, modelo_serializado, r2_train, r2_test, rmse_train, rmse_test,
             mae_train, mae_test)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            nombre_modelo,
            request.algorithm,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            json.dumps(request.features),
            request.target_column,
            json.dumps({
                'train_r2': float(train_r2),
                'test_r2': float(test_r2),
                'train_mse': float(train_mse),
                'test_mse': float(test_mse),
                'train_rmse': float(train_rmse),
                'test_rmse': float(test_rmse),
                'train_mae': float(train_mae),
                'test_mae': float(test_mae),
                'mape_test': float(mape_test) if mape_test is not None else None,
                'error_percentual': float(error_percentual) if error_percentual is not None else None,
                'r2_diff': float(r2_diff),
                'rmse_diff': float(rmse_diff)
            }),
            model_bytes,
            float(train_r2),
            float(test_r2),
            float(train_rmse),
            float(test_rmse),
            float(train_mae),
            float(test_mae)
        ))
        
        conn.commit()
        model_id = cursor.lastrowid
        conn.close()
        
        # Preparar respuesta con todas las mejoras
        response_metrics = {
            "train_r2": float(train_r2),
            "test_r2": float(test_r2),
            "train_mse": float(train_mse),
            "test_mse": float(test_mse),
            "train_rmse": float(train_rmse),
            "test_rmse": float(test_rmse),
            "train_mae": float(train_mae),
            "test_mae": float(test_mae),
            "mape_test": float(mape_test) if mape_test is not None else None,
            "error_percentual": float(error_percentual) if error_percentual is not None else None,
            "r2_diff": float(r2_diff),
            "rmse_diff": float(rmse_diff),
            "mean_y_test": float(mean_y_test)
        }
        
        # Agregar cross-validation scores si están disponibles
        if cv_scores is not None:
            response_metrics["cv_r2_mean"] = float(np.mean(cv_scores))
            response_metrics["cv_r2_std"] = float(np.std(cv_scores))
        
        # Agregar feature importance si está disponible
        response_data = {
            "success": True,
            "model_id": model_id,
            "metrics": response_metrics,
            "feature_importance": feature_importance if feature_importance else {},
            "selected_features": selected_features_list if selected_features_list else request.features,
            "improvements_applied": {
                "multicollinearity_removed": request.remove_multicollinearity,
                "feature_selection": request.auto_feature_selection,
                "polynomial_features": request.use_polynomial_features,
                "log_transformations": True,  # Siempre aplicado en preprocesamiento
                "derived_variables": True  # Siempre aplicado en preprocesamiento
            },
            "predictions": {
                "y_test": y_test.tolist(),
                "y_test_pred": y_test_pred.tolist(),
                "residuals": residuals_test.tolist()
            },
            "residuals_stats": residuals_stats,
            "data_split": {
                "train_size": len(X_train),
                "test_size": len(X_test),
                "train_percentage": float(len(X_train) / (len(X_train) + len(X_test))),
                "test_percentage": float(len(X_test) / (len(X_train) + len(X_test)))
            },
            "learning_curve": learning_curve_data if learning_curve_data else []
        }
    except HTTPException as he:
        raise he
    except KeyError as e:
        error_msg = f"Columna no encontrada: {str(e)}"
        print(f"KeyError en train_model: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except ValueError as e:
        error_msg = str(e)
        print(f"ValueError en train_model: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        error_msg = str(e)
        print(f"Error completo en train_model:\n{error_detail}")
        print(f"Request recibido: session_id={request.session_id}, target={request.target_column}, features={request.features}, algorithm={request.algorithm}")
        raise HTTPException(status_code=400, detail=f"Error al entrenar modelo: {error_msg}")


@app.get("/api/models")
def get_models():
    """Obtiene todos los modelos guardados"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, nombre_modelo, algoritmo, fecha_creacion, caracteristicas,
               variable_objetivo, metricas, r2_train, r2_test, rmse_train, rmse_test
        FROM modelos_ml
        ORDER BY fecha_creacion DESC
    """)
    
    models = []
    for row in cursor.fetchall():
        models.append({
            "id": row[0],
            "nombre_modelo": row[1],
            "algoritmo": row[2],
            "fecha_creacion": row[3],
            "caracteristicas": json.loads(row[4]),
            "variable_objetivo": row[5],
            "metricas": json.loads(row[6]),
            "r2_train": row[7],
            "r2_test": row[8],
            "rmse_train": row[9],
            "rmse_test": row[10]
        })
    
    conn.close()
    return {"models": models}


class PredictionRequest(BaseModel):
    model_id: int
    features: Dict[str, float]  # Diccionario con valores de features


@app.post("/api/model/predict")
def predict_with_model(request: PredictionRequest):
    """Hace una predicción usando un modelo entrenado guardado"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Cargar modelo
    cursor.execute("""
        SELECT modelo_serializado, caracteristicas, variable_objetivo, algoritmo
        FROM modelos_ml 
        WHERE id = ?
    """, (request.model_id,))
    
    result = cursor.fetchone()
    if not result:
        conn.close()
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    
    model_bytes, features_json, target_column, algorithm = result
    
    try:
        features_list = json.loads(features_json)
    except:
        features_list = []
    
    # Deserializar modelo
    model = pickle.loads(model_bytes)
    
    # Preparar datos de entrada
    if len(request.features) != len(features_list):
        conn.close()
        raise HTTPException(
            status_code=400, 
            detail=f"Se esperaban {len(features_list)} features, se recibieron {len(request.features)}"
        )
    
    # Crear DataFrame con los valores
    input_data = {}
    for feature in features_list:
        if feature not in request.features:
            conn.close()
            raise HTTPException(
                status_code=400, 
                detail=f"Feature faltante: {feature}"
            )
        input_data[feature] = [request.features[feature]]
    
    X_input = pd.DataFrame(input_data)
    
    # Hacer predicción
    try:
        prediction = model.predict(X_input)[0]
        conn.close()
        
        return {
            "success": True,
            "prediction": float(prediction),
            "target_column": target_column,
            "features_used": features_list,
            "model_algorithm": algorithm
        }
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=400, detail=f"Error al hacer predicción: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

