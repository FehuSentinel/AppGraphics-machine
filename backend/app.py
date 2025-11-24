#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backend API para Gestión de Tablas con Machine Learning
FastAPI + SQLite
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
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
    # IMPORTANTE: No aplicar cap de outliers si eliminaría toda la variabilidad
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if data[col].dtype in [np.float64, np.int64]:
            # Guardar valores originales para comparar
            original_unique = data[col].nunique()
            original_std = data[col].std()
            
            # Método IQR: ajustar valores extremos sin eliminar
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Si IQR es 0 o muy pequeño, no aplicar cap (todos los valores son iguales o casi iguales)
            if IQR < 1e-10:
                print(f"⚠️ Columna '{col}': IQR muy pequeño ({IQR:.10f}), no se aplica cap de outliers")
                continue
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Verificar cuántos valores están fuera del rango
            values_below = (data[col] < lower_bound).sum()
            values_above = (data[col] > upper_bound).sum()
            
            # Si TODOS los valores están fuera del rango, no aplicar cap (eliminaría toda la variabilidad)
            if values_below + values_above == len(data[col]):
                print(f"⚠️ Columna '{col}': Todos los valores están fuera del rango IQR, no se aplica cap para preservar variabilidad")
                continue
            
            # Aplicar cap solo si no eliminará toda la variabilidad
            data_before_cap = data[col].copy()
            data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
            
            # Verificar si después del cap todos los valores quedan iguales
            unique_after = data[col].nunique()
            std_after = data[col].std()
            
            if unique_after == 1:
                # Revertir el cambio si eliminó toda la variabilidad
                print(f"⚠️ Columna '{col}': El cap de outliers eliminó toda la variabilidad, se revierte el cambio")
                data[col] = data_before_cap
            elif std_after < original_std * 0.1:  # Si la desviación estándar se redujo en más del 90%
                # Revertir si la variabilidad se redujo demasiado
                print(f"⚠️ Columna '{col}': El cap de outliers redujo demasiado la variabilidad (std: {original_std:.2f} -> {std_after:.2f}), se revierte el cambio")
                data[col] = data_before_cap
            elif values_below > 0 or values_above > 0:
                print(f"✅ Columna '{col}': Ajustados {values_below + values_above} outliers (debajo: {values_below}, arriba: {values_above})")
    
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
    
    # NOTA: Variables derivadas y transformaciones log se aplican SOLO si se solicitan explícitamente
    # durante el entrenamiento, no automáticamente durante la carga de datos
    # Esto simplifica el flujo y evita problemas con predicciones
    
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
def get_data(session_id: str, limit: int = 10000):
    """Obtiene los datos de una sesión (por defecto hasta 10000 registros para asegurar que los gráficos vean todos los datos)"""
    if session_id not in current_data_store:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    
    df = current_data_store[session_id]
    
    # IMPORTANTE: Devolver TODOS los datos para que los gráficos muestren lo mismo que se usa para entrenar
    # Si hay más de 10000 registros, devolver todos pero con advertencia
    if len(df) > limit:
        print(f"⚠️ Advertencia: Hay {len(df)} registros, devolviendo primeros {limit}")
        data_to_return = df.head(limit)
    else:
        data_to_return = df
    
    print(f"📊 Devolviendo {len(data_to_return)} registros de {len(df)} totales para visualización")
    
    return {
        "rows": len(df),  # Total de filas en el backend
        "columns": len(df.columns),
        "columns_list": df.columns.tolist(),
        "data": data_to_return.to_dict(orient='records'),  # Todos los datos (hasta el límite)
        "total_rows": len(df),  # Total real
        "returned_rows": len(data_to_return)  # Filas devueltas
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
    """Entrena un modelo de machine learning usando EXACTAMENTE los mismos datos que se muestran en los gráficos"""
    print(f"📥 Request recibido: session_id={request.session_id}, target={request.target_column}, features={request.features}, algorithm={request.algorithm}")
    
    if request.session_id not in current_data_store:
        print(f"❌ Sesión no encontrada: {request.session_id}")
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    
    df = current_data_store[request.session_id].copy()
    print(f"✅ Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
    
    # IMPORTANTE: Usar TODOS los datos disponibles, no solo una muestra
    print(f"📊 Entrenando modelo con {len(df)} registros completos")
    print(f"📋 Columnas disponibles: {df.columns.tolist()}")
    print(f"🎯 Variable objetivo: {request.target_column}")
    print(f"🔧 Características seleccionadas: {request.features}")
    print(f"📋 Columnas disponibles después del preprocesamiento: {df.columns.tolist()}")
    
    try:
        # Validar que las columnas existan
        if not request.features or len(request.features) == 0:
            raise ValueError("Debe seleccionar al menos una característica")
        
        # Filtrar solo las columnas que realmente existen después del preprocesamiento
        available_features = [f for f in request.features if f in df.columns]
        missing_features = [f for f in request.features if f not in df.columns]
        
        if missing_features:
            print(f"⚠️ Advertencia: Algunas columnas no existen después del preprocesamiento: {missing_features}")
            print(f"📋 Columnas disponibles en el dataset: {df.columns.tolist()}")
            print(f"✅ Usando solo las columnas disponibles: {available_features}")
        
        if len(available_features) == 0:
            available_cols_str = ", ".join(df.columns.tolist()[:10])  # Mostrar primeras 10
            if len(df.columns) > 10:
                available_cols_str += f", ... (total: {len(df.columns)} columnas)"
            raise ValueError(
                f"Ninguna de las columnas seleccionadas existe después del preprocesamiento.\n"
                f"Columnas seleccionadas: {request.features}\n"
                f"Columnas disponibles: {available_cols_str}\n"
                f"Nota: El preprocesamiento puede haber creado nuevas columnas derivadas o transformadas."
            )
        
        # Usar solo las columnas disponibles
        request.features = available_features
        print(f"✅ Usando {len(available_features)} características válidas: {available_features}")
        
        if not request.target_column:
            raise ValueError("Debe seleccionar una variable objetivo")
        
        if request.target_column not in df.columns:
            raise ValueError(f"Columna objetivo no encontrada: {request.target_column}")
        
        # Verificar que target no esté en features
        if request.target_column in request.features:
            raise ValueError(f"La variable objetivo '{request.target_column}' no puede estar en las características")
        
        # Preparar datos - asegurar que sean numéricos
        # IMPORTANTE: Usar EXACTAMENTE las mismas columnas que se muestran en los gráficos
        X = df[request.features].copy()
        y = df[request.target_column].copy()
        
        print(f"✅ Datos preparados: X shape = {X.shape}, y shape = {y.shape}")
        
        # DIAGNÓSTICO: Verificar variabilidad ANTES de cualquier procesamiento
        print(f"🔍 DIAGNÓSTICO - Variable objetivo ANTES de conversión:")
        print(f"   Tipo: {y.dtype}")
        print(f"   Valores únicos: {y.nunique()}")
        print(f"   Primeros 10 valores: {y.head(10).tolist()}")
        if y.nunique() > 0:
            print(f"   Rango: [{y.min()}, {y.max()}]")
            print(f"   Media: {y.mean():.2f}, Std: {y.std():.2f}")
        
        # Convertir a numérico si es necesario
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        if not pd.api.types.is_numeric_dtype(y):
            y = pd.to_numeric(y, errors='coerce')
        
        # DIAGNÓSTICO: Verificar variabilidad DESPUÉS de conversión
        print(f"🔍 DIAGNÓSTICO - Variable objetivo DESPUÉS de conversión:")
        print(f"   Tipo: {y.dtype}")
        print(f"   Valores únicos: {y.nunique()}")
        print(f"   Primeros 10 valores: {y.head(10).tolist()}")
        if y.nunique() > 0:
            print(f"   Rango: [{y.min()}, {y.max()}]")
            print(f"   Media: {y.mean():.2f}, Std: {y.std():.2f}")
        
        # Eliminar filas con NaN (EXACTAMENTE como se hace en los gráficos)
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        print(f"🧹 Después de eliminar NaN: {len(X)} registros válidos de {len(df)} totales")
        
        # DIAGNÓSTICO: Verificar variabilidad DESPUÉS de eliminar NaN
        print(f"🔍 DIAGNÓSTICO - Variable objetivo DESPUÉS de eliminar NaN:")
        print(f"   Valores únicos: {y.nunique()}")
        if y.nunique() > 0:
            print(f"   Rango: [{y.min()}, {y.max()}]")
            print(f"   Media: {y.mean():.2f}, Std: {y.std():.2f}")
            if y.nunique() == 1:
                print(f"   ❌ PROBLEMA: Todos los valores son iguales: {y.iloc[0]}")
                print(f"   ❌ Esto puede ser porque:")
                print(f"      1. Los datos originales no tienen variabilidad")
                print(f"      2. El preprocesamiento eliminó toda la variabilidad")
                print(f"      3. La conversión a numérico causó pérdida de información")
        
        if len(X) == 0:
            raise ValueError("No hay datos válidos después de la limpieza. Verifique que las columnas seleccionadas contengan valores numéricos válidos.")
        
        # Verificar que tenemos suficientes datos
        if len(X) < 2:
            raise ValueError(f"Solo hay {len(X)} registros válidos. Se necesitan al menos 2 para entrenar un modelo.")
        
        print(f"✅ Datos finales para entrenamiento: {len(X)} registros, {len(X.columns)} características")
        print(f"📊 Estos son EXACTAMENTE los mismos datos que se muestran en los gráficos del frontend")
        
        # MEJORA 1: Eliminar multicolinealidad (features altamente correlacionadas)
        selected_features_list = []
        if request.remove_multicollinearity and len(X.columns) > 1:
            X = remove_multicollinearity(X, threshold=0.95)
        
        # MEJORA 2: Feature selection automática (seleccionar mejores features)
        # Nota: Se hace después de eliminar multicolinealidad pero antes del split
        # IMPORTANTE: Si es Regresión Lineal Simple, solo usar la mejor feature
        if request.auto_feature_selection:
            if request.algorithm == "Regresión Lineal Simple" and len(X.columns) > 1:
                # Para Regresión Lineal Simple, seleccionar SOLO la mejor feature
                # Pero primero verificar que las features tengan variabilidad
                print(f"🔍 Verificando variabilidad de features antes de selección...")
                valid_features = []
                for col in X.columns:
                    unique_count = X[col].nunique()
                    std_val = X[col].std()
                    print(f"   {col}: unique={unique_count}, std={std_val:.6f}")
                    if unique_count > 1 and std_val > 1e-10:
                        valid_features.append(col)
                    else:
                        print(f"   ⚠️ {col} tiene poca variabilidad (unique={unique_count}, std={std_val:.6f})")
                
                if len(valid_features) == 0:
                    raise ValueError(
                        "Ninguna feature tiene variabilidad suficiente. "
                        "Todas las features tienen valores constantes o casi constantes. "
                        "Verifica tus datos."
                    )
                
                # Filtrar X a solo features válidas
                X_valid = X[valid_features]
                print(f"✅ Features con variabilidad válida: {valid_features}")
                
                # Seleccionar la mejor feature de las válidas
                X_selected, selected_features_list = select_best_features(X_valid, y, k=1, method='f_regression')
                if selected_features_list and len(selected_features_list) > 0:
                    X = X_selected
                    print(f"✅ Feature selection para Regresión Lineal Simple: usando SOLO 1 feature: {selected_features_list[0]}")
                else:
                    # Si falla la selección, usar la primera feature válida
                    selected_features_list = [valid_features[0]]
                    X = X[[selected_features_list[0]]]
                    print(f"⚠️ Feature selection falló, usando primera feature válida: {selected_features_list[0]}")
            elif len(X.columns) > 3:
                X_selected, selected_features_list = select_best_features(X, y, method='f_regression')
                if selected_features_list and len(selected_features_list) < len(X.columns):
                    # Aplicar selección a todo X
                    X = X_selected
                    print(f"✅ Feature selection aplicada: {len(selected_features_list)}/{len(X.columns)} features seleccionadas")
                else:
                    selected_features_list = X.columns.tolist()
            else:
                selected_features_list = X.columns.tolist()
        else:
            # Si no hay feature selection, usar todas las features
            selected_features_list = X.columns.tolist()
        
        # CRÍTICO: Verificar que la feature seleccionada tenga variabilidad
        if len(selected_features_list) > 0:
            for feature in selected_features_list:
                if feature in X.columns:
                    unique_count = X[feature].nunique()
                    std_val = X[feature].std()
                    if unique_count == 1 or std_val < 1e-10:
                        raise ValueError(
                            f"La feature seleccionada '{feature}' no tiene variabilidad suficiente "
                            f"(unique={unique_count}, std={std_val:.6f}). "
                            f"Selecciona otra feature o verifica tus datos."
                        )
        
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
        
        # Dividir datos PRIMERO (antes de normalización)
        X_train_temp, X_test_temp, y_train, y_test = train_test_split(
            X, y, test_size=request.test_size, random_state=request.random_state
        )
        
        # Normalización si se solicita (después del split)
        scaler = None
        if request.normalize:
            scaler = StandardScaler()
            # Ajustar el scaler SOLO con los datos de entrenamiento
            X_train_scaled = scaler.fit_transform(X_train_temp)
            X_train = pd.DataFrame(
                X_train_scaled,
                columns=X_train_temp.columns,
                index=X_train_temp.index
            )
            # Transformar test con los parámetros del train
            X_test_scaled = scaler.transform(X_test_temp)
            X_test = pd.DataFrame(
                X_test_scaled,
                columns=X_test_temp.columns,
                index=X_test_temp.index
            )
            print(f"✅ Normalización aplicada:")
            print(f"   mean_ shape: {scaler.mean_.shape}, scale_ shape: {scaler.scale_.shape}")
            print(f"   mean_ values: {scaler.mean_}")
            print(f"   scale_ values: {scaler.scale_}")
            
            # CRÍTICO: Verificar que scale_ no sea cero o muy pequeño (causaría problemas)
            if np.any(scaler.scale_ == 0):
                zero_scale_features = [X_train.columns[i] for i in range(len(scaler.scale_)) if scaler.scale_[i] == 0]
                print(f"❌ ERROR CRÍTICO: Algunos valores de scale_ son cero!")
                print(f"❌ Features con scale_=0: {zero_scale_features}")
                print(f"❌ Esto significa que estas features tienen varianza cero (todos los valores son iguales)")
                print(f"❌ Estas features no aportan información y causarán problemas en predicción")
                raise ValueError(
                    f"Features con varianza cero (todos los valores iguales): {zero_scale_features}. "
                    f"Estas features no pueden usarse con normalización. SOLUCIÓN: Desactiva la normalización (uncheck 'Normalizar datos') y vuelve a entrenar."
                )
            
            # Verificar que scale_ no sea muy pequeño (puede causar problemas numéricos)
            min_scale = np.min(scaler.scale_)
            min_scale_idx = np.argmin(scaler.scale_)
            min_scale_feature = X_train.columns[min_scale_idx] if min_scale_idx < len(X_train.columns) else "unknown"
            
            if min_scale < 1e-6:
                print(f"❌ ERROR CRÍTICO: Algunos valores de scale_ son muy pequeños (< 1e-6): {min_scale}")
                print(f"❌ Feature con scale_ más pequeño: {min_scale_feature} (scale_={min_scale:.10f})")
                print(f"❌ Esto causará que los coeficientes sean cero o casi cero después de la normalización")
                raise ValueError(
                    f"La normalización tiene scale_ muy pequeño ({min_scale:.10f}) para la feature '{min_scale_feature}'. "
                    f"Esto causará que el modelo no aprenda. SOLUCIÓN: Desactiva la normalización (uncheck 'Normalizar datos') y vuelve a entrenar."
                )
            
            # Verificar que después de la normalización, X_train todavía tenga variabilidad
            for i, col in enumerate(X_train.columns):
                if i < len(scaler.scale_):
                    normalized_std = X_train[col].std()
                    if normalized_std < 1e-6:
                        print(f"❌ ERROR: Después de la normalización, '{col}' tiene std muy pequeño: {normalized_std:.10f}")
                        raise ValueError(
                            f"Después de la normalización, la feature '{col}' tiene varianza casi cero. "
                            f"SOLUCIÓN: Desactiva la normalización y vuelve a entrenar."
                        )
        else:
            # Sin normalización, usar los datos directamente
            X_train = X_train_temp
            X_test = X_test_temp
            print(f"✅ Sin normalización: usando datos originales")
            print(f"   X_train range: [{X_train.min().min():.2f}, {X_train.max().max():.2f}]")
            print(f"   X_test range: [{X_test.min().min():.2f}, {X_test.max().max():.2f}]")
        
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
        
        # Validar que X_train tenga al menos una columna
        if X_train.shape[1] == 0:
            raise ValueError(f"No hay características válidas para entrenar. X_train tiene {X_train.shape[1]} columnas")
        
        # Logging detallado antes de entrenar
        print(f"📊 Datos finales para entrenamiento:")
        print(f"   X_train shape: {X_train.shape}")
        print(f"   y_train shape: {y_train.shape}")
        print(f"   X_test shape: {X_test.shape}")
        print(f"   y_test shape: {y_test.shape}")
        print(f"   Features: {X_train.columns.tolist()}")
        print(f"   Primeras 3 filas de X_train:")
        for idx in range(min(3, len(X_train))):
            print(f"      Fila {idx}: {X_train.iloc[idx].to_dict()}")
        print(f"   Primeras 3 valores de y_train: {y_train.iloc[:3].tolist()}")
        
        # MEJORA: Learning curves (calcular error en diferentes tamaños de entrenamiento)
        learning_curve_data = []
        if len(X_train) >= 20:  # Solo si hay suficientes datos
            try:
                train_sizes = np.linspace(0.1, 1.0, min(10, len(X_train)//10))
                print(f"📈 Calculando learning curves con {len(train_sizes)} puntos...")
                
                for size in train_sizes:
                    n_samples = max(1, int(len(X_train) * size))
                    # Asegurar que no exceda el tamaño de X_train
                    n_samples = min(n_samples, len(X_train))
                    
                    # Resetear índices para evitar problemas con iloc
                    X_train_reset = X_train.reset_index(drop=True)
                    y_train_reset = y_train.reset_index(drop=True)
                    
                    X_train_subset = X_train_reset.iloc[:n_samples]
                    y_train_subset = y_train_reset.iloc[:n_samples]
                    
                    # Validar que hay suficientes datos
                    if len(X_train_subset) < 2:
                        continue
                    
                    try:
                        # Crear y entrenar modelo temporal
                        temp_model = type(model)(**model.get_params())
                        temp_model.fit(X_train_subset, y_train_subset)
                        
                        # Predecir en train y test
                        train_pred = temp_model.predict(X_train_subset)
                        test_pred = temp_model.predict(X_test)
                        
                        # Calcular errores
                        train_mse = mean_squared_error(y_train_subset, train_pred)
                        test_mse = mean_squared_error(y_test, test_pred)
                        
                        # Calcular RMSE (raíz del MSE)
                        train_rmse = np.sqrt(train_mse) if train_mse >= 0 else 0.0
                        test_rmse = np.sqrt(test_mse) if test_mse >= 0 else 0.0
                        
                        # Validar que los valores sean finitos
                        if not (np.isfinite(train_rmse) and np.isfinite(test_rmse)):
                            print(f"⚠️ Valores no finitos en learning curve (size={size}): train_rmse={train_rmse}, test_rmse={test_rmse}")
                            continue
                        
                        # Asegurar valores mínimos para evitar problemas de visualización
                        train_rmse = max(train_rmse, 1e-10) if train_rmse > 0 else 1e-10
                        test_rmse = max(test_rmse, 1e-10) if test_rmse > 0 else 1e-10
                        
                        learning_curve_data.append({
                            "train_size": int(n_samples),
                            "train_error": float(train_rmse),
                            "test_error": float(test_rmse)
                        })
                    except Exception as e:
                        print(f"⚠️ Error en learning curve para size={size}, n_samples={n_samples}: {e}")
                        continue
                
                print(f"✅ Learning curves calculadas: {len(learning_curve_data)} puntos válidos")
                if len(learning_curve_data) == 0:
                    print("⚠️ No se pudieron calcular learning curves válidas")
            except Exception as e:
                import traceback
                print(f"⚠️ Error calculando learning curves: {e}")
                print(traceback.format_exc())
                learning_curve_data = []
        
        # CRÍTICO: Validar variabilidad ANTES del split (en los datos completos)
        print(f"🔍 Validando variabilidad de datos ANTES del split...")
        
        # Validar X (features) - usar X completo, no X_train
        for col in X.columns:
            unique_count = X[col].nunique()
            std_val = X[col].std()
            if unique_count == 1:
                print(f"❌ ERROR: La feature '{col}' tiene todos los valores iguales ({X[col].iloc[0]})")
                raise ValueError(
                    f"La feature '{col}' no tiene variabilidad (todos los valores son iguales). "
                    f"Selecciona otra feature o verifica tus datos."
                )
            elif std_val < 1e-10:
                print(f"❌ ERROR: La feature '{col}' tiene desviación estándar muy pequeña ({std_val:.10f})")
                raise ValueError(
                    f"La feature '{col}' tiene varianza casi cero (std={std_val:.10f}). "
                    f"Esto causará problemas. Selecciona otra feature o desactiva la normalización."
                )
            else:
                print(f"✅ Feature '{col}': unique={unique_count}, std={std_val:.6f}, range=[{X[col].min():.2f}, {X[col].max():.2f}]")
        
        # Validar y (target) - usar y completo, no y_train
        y_unique = y.nunique()
        y_std = y.std()
        if y_unique == 1:
            print(f"❌ ERROR: La variable objetivo tiene todos los valores iguales ({y.iloc[0]})")
            raise ValueError(
                "La variable objetivo no tiene variabilidad (todos los valores son iguales). "
                "No se puede entrenar un modelo predictivo sin variabilidad en el target. "
                "Verifica que la columna 'precio_clp' tenga diferentes valores."
            )
        elif y_std < 1e-10:
            print(f"❌ ERROR: La variable objetivo tiene desviación estándar muy pequeña ({y_std:.10f})")
            raise ValueError(
                "La variable objetivo tiene varianza casi cero. "
                "No se puede entrenar un modelo predictivo sin variabilidad en el target."
            )
        else:
            print(f"✅ Variable objetivo: unique={y_unique}, std={y_std:.6f}, range=[{y.min():.2f}, {y.max():.2f}]")
        
        # Entrenar modelo final
        print(f"🔧 Entrenando modelo {request.algorithm}...")
        print(f"   X_train shape: {X_train.shape}")
        print(f"   y_train shape: {y_train.shape}")
        print(f"   Features: {X_train.columns.tolist()}")
        
        model.fit(X_train, y_train)
        print(f"✅ Modelo entrenado exitosamente")
        
        # Verificar que el modelo aprendió algo (para regresión lineal)
        if hasattr(model, 'coef_'):
            coef = model.coef_
            print(f"📊 Coeficientes del modelo: {coef}")
            print(f"📊 Tipo de coeficientes: {type(coef)}, Shape: {coef.shape if hasattr(coef, 'shape') else 'N/A'}")
            if hasattr(model, 'intercept_'):
                intercept = model.intercept_
                print(f"📊 Intercepto: {intercept}")
                print(f"📊 Tipo de intercepto: {type(intercept)}")
            
            # Verificar si todos los coeficientes son cero (problema)
            # PERO: Si la normalización causó que scale_ sea muy pequeño, los coeficientes pueden parecer cero
            # cuando en realidad el problema es la normalización
            if np.allclose(coef, 0, atol=1e-6):
                print(f"❌ ERROR CRÍTICO: Todos los coeficientes son cero o casi cero! El modelo no aprendió nada.")
                print(f"❌ Esto causará que todas las predicciones sean iguales al intercepto: {intercept}")
                print(f"❌ Posibles causas:")
                print(f"   - Las features no tienen relación con el target")
                print(f"   - Las features tienen varianza cero (todos los valores iguales)")
                print(f"   - Problema con la normalización (scale_=0 o muy pequeño)")
                
                # Si hay normalización, sugerir desactivarla
                if request.normalize and scaler is not None:
                    if hasattr(scaler, 'scale_'):
                        print(f"❌ Valores de scale_: {scaler.scale_}")
                        if np.any(scaler.scale_ < 1e-6):
                            print(f"💡 SOLUCIÓN: Desactiva la normalización y vuelve a entrenar")
                            raise ValueError(
                                "El modelo no aprendió nada. La normalización tiene scale_ muy pequeño o cero. "
                                "SOLUCIÓN: Desactiva la normalización (uncheck 'Normalizar datos') y vuelve a entrenar."
                            )
                
                # Si no hay normalización, el problema es que las features no tienen relación
                raise ValueError(
                    "El modelo no aprendió nada (todos los coeficientes son cero). "
                    "Posibles soluciones:\n"
                    "1. Verifica que las features tengan variabilidad (no todos los valores iguales)\n"
                    "2. Verifica que las features tengan relación con el target\n"
                    "3. Prueba con otras features\n"
                    "4. Si usas normalización, desactívala y vuelve a entrenar"
                )
            
            # Para Regresión Lineal Simple, verificar que el coeficiente no sea cero
            if request.algorithm == "Regresión Lineal Simple":
                if len(coef.shape) == 1 and len(coef) == 1:
                    coef_value = coef[0] if isinstance(coef, np.ndarray) else coef
                    if np.abs(coef_value) < 1e-10:
                        print(f"❌ ERROR CRÍTICO: El coeficiente es casi cero ({coef_value})! El modelo no aprenderá de la feature.")
                        print(f"❌ Esto significa que la feature '{X_train.columns[0]}' no tiene relación con el target")
                        raise ValueError(
                            f"El coeficiente es cero. La feature '{X_train.columns[0]}' no tiene relación con el target '{request.target_column}'. "
                            f"Prueba con otra feature o verifica los datos."
                        )
                    else:
                        print(f"✅ Coeficiente válido: {coef_value:.6f}, Intercepto: {intercept:.6f}")
                        print(f"✅ Fórmula del modelo: y = {intercept:.6f} + {coef_value:.6f} * {X_train.columns[0]}")
                        
                        # Probar la fórmula con un valor de ejemplo
                        example_x = X_train.iloc[0, 0]
                        example_y_pred = intercept + coef_value * example_x
                        example_y_real = y_train.iloc[0]
                        print(f"✅ Verificación: Para X={example_x:.2f}, predicción manual={example_y_pred:.2f}, valor real={example_y_real:.2f}")
        
        # Predecir
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        print(f"✅ Predicciones realizadas:")
        print(f"   y_train_pred: {y_train_pred[:5]}... (primeros 5)")
        print(f"   y_test_pred: {y_test_pred[:5]}... (primeros 5)")
        print(f"   y_train real: {y_train.iloc[:5].tolist()}... (primeros 5)")
        print(f"   y_test real: {y_test.iloc[:5].tolist()}... (primeros 5)")
        
        # CRÍTICO: Verificar que las predicciones NO sean todas iguales
        if len(np.unique(y_train_pred)) == 1:
            print(f"❌ ERROR CRÍTICO: Todas las predicciones de entrenamiento son iguales: {y_train_pred[0]}")
            print(f"❌ Esto indica que el modelo no está aprendiendo de las features")
            raise ValueError(
                "El modelo está prediciendo el mismo valor para todos los casos. "
                "Esto indica que las features no tienen relación con el target o hay un problema con el modelo."
            )
        
        if len(np.unique(y_test_pred)) == 1:
            print(f"⚠️ ADVERTENCIA: Todas las predicciones de test son iguales: {y_test_pred[0]}")
            print(f"⚠️ Esto puede indicar un problema con el modelo")
        
        # Validar que las predicciones tengan el mismo tamaño
        if len(y_test_pred) != len(y_test):
            raise ValueError(f"Error en predicciones: y_test_pred tiene {len(y_test_pred)} valores pero y_test tiene {len(y_test)}")
        
        # Verificar que las predicciones tengan sentido (no sean todas NaN o Inf)
        if np.any(np.isnan(y_train_pred)) or np.any(np.isinf(y_train_pred)):
            print(f"❌ ERROR: Las predicciones contienen NaN o Inf")
            raise ValueError("Las predicciones contienen valores inválidos (NaN o Inf)")
        
        if np.any(np.isnan(y_test_pred)) or np.any(np.isinf(y_test_pred)):
            print(f"❌ ERROR: Las predicciones de test contienen NaN o Inf")
            raise ValueError("Las predicciones de test contienen valores inválidos (NaN o Inf)")
        
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
        
        # Calcular cross-validation scores si se solicita
        cv_scores = None
        if hasattr(request, 'cross_validation') and request.cross_validation:
            try:
                if len(X_train) >= 10:  # Mínimo para cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(X_train)//5), scoring='r2')
                    print(f"✅ Cross-validation completada: R² CV = {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
            except Exception as e:
                print(f"⚠️ Error en cross-validation: {e}")
                cv_scores = None
        
        # Calcular feature importance para modelos basados en árboles (después de entrenar)
        feature_importance = {}
        try:
            if hasattr(model, 'feature_importances_'):
                feature_names = X_train.columns.tolist()
                importances = model.feature_importances_
                if len(feature_names) == len(importances):
                    for name, importance in zip(feature_names, importances):
                        feature_importance[name] = float(importance)
        except Exception as e:
            print(f"⚠️ Error calculando feature importance: {e}")
            feature_importance = {}
        
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
        
        # Serializar modelo y verificar que se guardó correctamente
        model_bytes = pickle.dumps(model)
        print(f"💾 Modelo serializado: {len(model_bytes)} bytes")
        
        # Verificar que el modelo serializado contiene los coeficientes correctos
        try:
            model_test = pickle.loads(model_bytes)
            if hasattr(model_test, 'coef_'):
                print(f"✅ Verificación: Modelo deserializado tiene coeficientes: {model_test.coef_}")
                if hasattr(model_test, 'intercept_'):
                    print(f"✅ Verificación: Intercepto: {model_test.intercept_}")
        except Exception as e:
            print(f"⚠️ Error al verificar modelo serializado: {e}")
        
        nombre_modelo = f"{request.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Guardar scaler y transformadores si existen
        # SIMPLIFICADO: No aplicar transformaciones automáticas complejas
        # IMPORTANTE: Usar las features finales que realmente se usaron (después de todas las transformaciones)
        final_features_used = X_train.columns.tolist()
        print(f"💾 Guardando pipeline de preprocesamiento:")
        print(f"   Features originales: {request.features}")
        print(f"   Features finales usadas: {final_features_used}")
        print(f"   Normalización: {request.normalize}")
        print(f"   Polynomial features: {request.use_polynomial_features}")
        
        preprocessing_pipeline = {
            'scaler': scaler,
            'polynomial_transformer': polynomial_transformer,
            'normalize': request.normalize,
            'use_polynomial_features': request.use_polynomial_features,
            'selected_features': final_features_used,  # Features que realmente se usaron
            'original_features': request.features  # Features originales antes de transformaciones
        }
        preprocessing_bytes = pickle.dumps(preprocessing_pipeline)
        
        cursor.execute("""
            INSERT INTO modelos_ml 
            (nombre_modelo, algoritmo, fecha_creacion, caracteristicas, variable_objetivo,
             metricas, modelo_serializado, scaler_serializado, r2_train, r2_test, rmse_train, rmse_test,
             mae_train, mae_test)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            nombre_modelo,
            request.algorithm,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            json.dumps(final_features_used),  # Guardar las features finales que realmente se usaron
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
            preprocessing_bytes,
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
            "selected_features": final_features_used,  # Features que realmente se usaron
            "improvements_applied": {
                "multicollinearity_removed": request.remove_multicollinearity,
                "feature_selection": request.auto_feature_selection,
                "polynomial_features": request.use_polynomial_features
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
        
        print(f"✅ Modelo entrenado exitosamente. Model ID: {model_id}")
        print(f"📊 Métricas: R² Train={train_r2:.4f}, R² Test={test_r2:.4f}")
        print(f"📤 Devolviendo respuesta con {len(response_data)} campos")
        
        # FastAPI maneja la serialización automáticamente
        return response_data
    except HTTPException as he:
        print(f"❌ HTTPException: {he.status_code} - {he.detail}")
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
               variable_objetivo, metricas, r2_train, r2_test, rmse_train, rmse_test, scaler_serializado
        FROM modelos_ml
        ORDER BY fecha_creacion DESC
    """)
    
    models = []
    for row in cursor.fetchall():
        model_id = row[0]
        preprocessing_bytes = row[11]
        
        # Obtener features originales del preprocessing pipeline si existe
        original_features = None
        if preprocessing_bytes:
            try:
                preprocessing_pipeline = pickle.loads(preprocessing_bytes)
                original_features = preprocessing_pipeline.get('original_features')
            except:
                pass
        
        models.append({
            "id": model_id,
            "nombre_modelo": row[1],
            "algoritmo": row[2],
            "fecha_creacion": row[3],
            "caracteristicas": json.loads(row[4]),
            "original_features": original_features,  # Features originales antes de transformaciones
            "variable_objetivo": row[5],
            "metricas": json.loads(row[6]),
            "r2_train": row[7],
            "r2_test": row[8],
            "rmse_train": row[9],
            "rmse_test": row[10]
        })
    
    conn.close()
    return {"models": models}


@app.get("/api/models/{model_id}")
def get_model_details(model_id: int):
    """Obtiene los detalles de un modelo específico, incluyendo las features originales requeridas"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT modelo_serializado, scaler_serializado, caracteristicas, variable_objetivo, algoritmo
        FROM modelos_ml 
        WHERE id = ?
    """, (model_id,))
    
    result = cursor.fetchone()
    if not result:
        conn.close()
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    
    model_bytes, preprocessing_bytes, features_json, target_column, algorithm = result
    
    try:
        features_list = json.loads(features_json)
    except:
        features_list = []
    
    # Obtener features originales del preprocessing pipeline
    original_features = features_list
    if preprocessing_bytes:
        try:
            preprocessing_pipeline = pickle.loads(preprocessing_bytes)
            original_features = preprocessing_pipeline.get('original_features', features_list)
        except:
            pass
    
    conn.close()
    
    return {
        "model_id": model_id,
        "algorithm": algorithm,
        "target_column": target_column,
        "features": features_list,  # Features finales que usa el modelo
        "original_features": original_features  # Features originales requeridas para predicción
    }


class PredictionRequest(BaseModel):
    model_id: int
    features: Dict[str, float]  # Diccionario con valores de features


@app.post("/api/model/predict/batch")
async def predict_batch_with_model(file: UploadFile = File(...), model_id: int = Form(...)):
    """Hace predicciones masivas usando un archivo CSV/Excel"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Cargar modelo y preprocesamiento
    cursor.execute("""
        SELECT modelo_serializado, scaler_serializado, caracteristicas, variable_objetivo, algoritmo
        FROM modelos_ml 
        WHERE id = ?
    """, (model_id,))
    
    result = cursor.fetchone()
    if not result:
        conn.close()
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    
    model_bytes, preprocessing_bytes, features_json, target_column, algorithm = result
    
    try:
        features_list = json.loads(features_json)
    except:
        features_list = []
    
    # Deserializar modelo
    model = pickle.loads(model_bytes)
    
    # Deserializar pipeline de preprocesamiento si existe
    preprocessing_pipeline = None
    if preprocessing_bytes:
        try:
            preprocessing_pipeline = pickle.loads(preprocessing_bytes)
        except:
            preprocessing_pipeline = None
    
    # Leer archivo
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.file, encoding='utf-8')
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file.file)
        else:
            conn.close()
            raise HTTPException(status_code=400, detail="Formato no soportado. Use CSV o Excel")
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=400, detail=f"Error al leer archivo: {str(e)}")
    
    # Validar que tenga las columnas necesarias
    missing_features = [f for f in features_list if f not in df.columns]
    if missing_features:
        conn.close()
        raise HTTPException(
            status_code=400,
            detail=f"Faltan columnas en el archivo: {missing_features}. Se requieren: {features_list}"
        )
    
    # Preparar datos - usar features originales si están disponibles
    original_features = preprocessing_pipeline.get('original_features', features_list) if preprocessing_pipeline else features_list
    
    # Validar que el archivo tenga las features originales
    missing_features = [f for f in original_features if f not in df.columns]
    if missing_features:
        conn.close()
        raise HTTPException(
            status_code=400,
            detail=f"Faltan columnas en el archivo: {missing_features}. Se requieren: {original_features}"
        )
    
    X_input = df[original_features].copy()
    
    # Convertir a numérico
    for col in X_input.columns:
        X_input[col] = pd.to_numeric(X_input[col], errors='coerce')
    
    # SIMPLIFICADO: Solo aplicar feature selection si se hizo durante el entrenamiento
    if preprocessing_pipeline:
        expected_features = preprocessing_pipeline.get('selected_features', features_list)
        if expected_features and len(expected_features) < len(X_input.columns):
            print(f"📋 Aplicando feature selection: usando {len(expected_features)} de {len(X_input.columns)} features")
            missing_features = [f for f in expected_features if f not in X_input.columns]
            if missing_features:
                conn.close()
                raise HTTPException(
                    status_code=400,
                    detail=f"Features faltantes: {missing_features}. "
                           f"Features disponibles: {X_input.columns.tolist()}, Features esperadas: {expected_features}"
                )
            X_input = X_input[expected_features]
            print(f"✅ Feature selection aplicada. Features finales: {X_input.columns.tolist()}")
    
    # Eliminar filas con NaN
    mask = ~X_input.isnull().any(axis=1)
    X_input = X_input[mask]
    
    if len(X_input) == 0:
        conn.close()
        raise HTTPException(status_code=400, detail="No hay filas válidas después de la limpieza")
    
    # Aplicar preprocesamiento específico del modelo si existe
    if preprocessing_pipeline:
        # Obtener las features que realmente usa el modelo
        model_features = preprocessing_pipeline.get('selected_features', features_list)
        
        if preprocessing_pipeline.get('polynomial_transformer') and preprocessing_pipeline.get('use_polynomial_features'):
            poly_transformer = preprocessing_pipeline['polynomial_transformer']
            X_input = poly_transformer.transform(X_input)
            # IMPORTANTE: Usar model_features (las que realmente usa el modelo)
            feature_names = poly_transformer.get_feature_names_out(model_features)
            X_input = pd.DataFrame(X_input, columns=feature_names)
        
        if preprocessing_pipeline.get('scaler') and preprocessing_pipeline.get('normalize'):
            scaler = preprocessing_pipeline['scaler']
            X_input = pd.DataFrame(
                scaler.transform(X_input),
                columns=X_input.columns
            )
    
    # Hacer predicciones
    try:
        predictions = model.predict(X_input)
        
        # Agregar predicciones al DataFrame original
        df_result = df.copy()
        df_result[f'prediccion_{target_column}'] = None
        df_result.loc[mask, f'prediccion_{target_column}'] = predictions
        
        conn.close()
        
        return {
            "success": True,
            "total_rows": len(df),
            "valid_rows": len(X_input),
            "predictions": df_result.to_dict(orient='records'),
            "target_column": target_column,
            "model_algorithm": algorithm
        }
    except Exception as e:
        conn.close()
        import traceback
        error_detail = traceback.format_exc()
        print(f"Error en predicción masiva:\n{error_detail}")
        raise HTTPException(status_code=400, detail=f"Error al hacer predicciones: {str(e)}")


@app.post("/api/model/predict")
def predict_with_model(request: PredictionRequest):
    """Hace una predicción usando un modelo entrenado guardado"""
    print(f"📥 Request de predicción recibido: model_id={request.model_id}, features={list(request.features.keys())}")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Cargar modelo y preprocesamiento
    cursor.execute("""
        SELECT modelo_serializado, scaler_serializado, caracteristicas, variable_objetivo, algoritmo
        FROM modelos_ml 
        WHERE id = ?
    """, (request.model_id,))
    
    result = cursor.fetchone()
    if not result:
        conn.close()
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    
    model_bytes, preprocessing_bytes, features_json, target_column, algorithm = result
    
    try:
        features_list = json.loads(features_json)
        print(f"✅ Features del modelo: {features_list}")
    except Exception as e:
        print(f"⚠️ Error parseando features_json: {e}")
        features_list = []
    
    # Deserializar modelo
    try:
        model = pickle.loads(model_bytes)
        print(f"✅ Modelo deserializado: {algorithm}")
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Error al deserializar modelo: {str(e)}")
    
    # Deserializar pipeline de preprocesamiento si existe
    preprocessing_pipeline = None
    if preprocessing_bytes:
        try:
            preprocessing_pipeline = pickle.loads(preprocessing_bytes)
            print(f"✅ Pipeline de preprocesamiento cargado")
        except Exception as e:
            print(f"⚠️ Error deserializando pipeline: {e}")
            preprocessing_pipeline = None
    
    # SIMPLIFICADO: Usar las features que el modelo realmente espera (después de feature selection)
    # Si hay feature selection, solo esas features importan
    expected_features = features_list
    if preprocessing_pipeline:
        selected_features = preprocessing_pipeline.get('selected_features')
        if selected_features and len(selected_features) > 0:
            expected_features = selected_features
            print(f"📋 Modelo usa feature selection: {len(selected_features)} features")
        else:
            print(f"📋 Modelo usa todas las features: {len(features_list)} features")
    
    print(f"📋 Features esperadas por el modelo: {expected_features}")
    print(f"📋 Features recibidas: {list(request.features.keys())}")
    print(f"📋 Algoritmo del modelo: {algorithm}")
    
    # ADVERTENCIA: Si es Regresión Lineal Simple, solo usa UNA feature
    if algorithm == "Regresión Lineal Simple" and len(expected_features) > 1:
        print(f"⚠️ ADVERTENCIA: Regresión Lineal Simple solo debería usar 1 feature, pero se recibieron {len(expected_features)}")
        print(f"⚠️ El modelo probablemente solo usa la primera feature: {expected_features[0]}")
    
    # Validar que se recibieron las features correctas
    if len(request.features) != len(expected_features):
        conn.close()
        raise HTTPException(
            status_code=400, 
            detail=f"Se esperaban {len(expected_features)} features, se recibieron {len(request.features)}. "
                   f"Esperadas: {expected_features}, Recibidas: {list(request.features.keys())}"
        )
    
    # Crear DataFrame con los valores en el orden correcto
    input_data = {}
    for feature in expected_features:
        if feature not in request.features:
            conn.close()
            raise HTTPException(
                status_code=400, 
                detail=f"Feature faltante: {feature}. Features requeridas: {expected_features}, Features recibidas: {list(request.features.keys())}"
            )
        input_data[feature] = [request.features[feature]]
    
    # Crear DataFrame con el orden correcto de las columnas
    X_input = pd.DataFrame(input_data, columns=expected_features)
    print(f"✅ DataFrame creado con {X_input.shape[0]} fila(s) y {X_input.shape[1]} columna(s)")
    print(f"📊 VALORES DE ENTRADA (ANTES de cualquier transformación):")
    for col in X_input.columns:
        print(f"   {col}: {X_input[col].iloc[0]} (tipo: {type(X_input[col].iloc[0])})")
    
    # SIMPLIFICADO: X_input ya tiene las features correctas (se validaron arriba)
    # No necesitamos aplicar feature selection de nuevo porque ya filtramos arriba
    
    # Aplicar preprocesamiento específico del modelo si existe (igual que durante el entrenamiento)
    # IMPORTANTE: Aplicar en el mismo orden que durante el entrenamiento
    if preprocessing_pipeline:
        # 1. Aplicar polynomial features PRIMERO (si se usaron)
        if preprocessing_pipeline.get('polynomial_transformer') and preprocessing_pipeline.get('use_polynomial_features'):
            print(f"🔄 Aplicando polynomial features...")
            poly_transformer = preprocessing_pipeline['polynomial_transformer']
            X_input_before = X_input.copy()
            
            # Verificar que el número de features coincida
            if X_input.shape[1] != len(expected_features):
                conn.close()
                raise HTTPException(
                    status_code=400,
                    detail=f"Error: El modelo espera {len(expected_features)} features para polynomial, pero se recibieron {X_input.shape[1]}"
                )
            
            X_input_transformed = poly_transformer.transform(X_input)
            feature_names = poly_transformer.get_feature_names_out(expected_features)
            X_input = pd.DataFrame(X_input_transformed, columns=feature_names)
            print(f"📊 Después de polynomial: {X_input.shape[1]} features")
            print(f"📊 Primeras 5 features: {X_input.columns.tolist()[:5]}")
        
        # 2. Aplicar scaler DESPUÉS (si se usó)
        if preprocessing_pipeline.get('scaler') and preprocessing_pipeline.get('normalize'):
            print(f"🔄 Aplicando scaler (normalización)...")
            scaler = preprocessing_pipeline['scaler']
            
            print(f"📊 VALORES ANTES DE SCALER (TODAS las features):")
            for col in X_input.columns:
                print(f"   {col}: {X_input[col].iloc[0]}")
            
            # Verificar parámetros del scaler para debugging
            if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                print(f"📊 Parámetros del scaler:")
                print(f"   mean_ shape: {scaler.mean_.shape}, scale_ shape: {scaler.scale_.shape}")
                print(f"   X_input shape: {X_input.shape}")
                
                # Verificar que las dimensiones coincidan
                if len(scaler.mean_) != X_input.shape[1]:
                    conn.close()
                    raise HTTPException(
                        status_code=400,
                        detail=f"Error: El scaler espera {len(scaler.mean_)} features, pero X_input tiene {X_input.shape[1]} features después de transformaciones"
                    )
                
                for i, col in enumerate(X_input.columns):
                    if i < len(scaler.mean_) and i < len(scaler.scale_):
                        mean_val = scaler.mean_[i]
                        scale_val = scaler.scale_[i]
                        print(f"   {col}: mean={mean_val:.6f}, scale={scale_val:.6f}")
            
            # Aplicar transformación
            try:
                X_input_scaled = scaler.transform(X_input)
                X_input = pd.DataFrame(X_input_scaled, columns=X_input.columns, index=X_input.index)
            except Exception as e:
                conn.close()
                raise HTTPException(
                    status_code=400,
                    detail=f"Error al aplicar scaler: {str(e)}. X_input shape: {X_input.shape}, Scaler espera: {len(scaler.mean_)} features"
                )
            
            print(f"📊 VALORES DESPUÉS DE SCALER (TODAS las features):")
            for col in X_input.columns:
                print(f"   {col}: {X_input[col].iloc[0]:.6f}")
            
            # CRÍTICO: Verificar si todos los valores son iguales después del scaler
            unique_values = X_input.iloc[0].nunique()
            if unique_values == 1:
                print(f"❌ ERROR CRÍTICO: Todos los valores después del scaler son iguales: {X_input.iloc[0].iloc[0]}")
                print(f"❌ Esto causará que la predicción sea siempre la misma!")
                if hasattr(scaler, 'scale_'):
                    print(f"❌ Valores de scale_: {scaler.scale_}")
                    zero_scale_indices = [i for i, s in enumerate(scaler.scale_) if s == 0 or abs(s) < 1e-10]
                    if zero_scale_indices:
                        zero_scale_features = [X_input.columns[i] for i in zero_scale_indices]
                        print(f"❌ Features con scale_=0 o muy pequeño: {zero_scale_features}")
                        conn.close()
                        raise HTTPException(
                            status_code=400,
                            detail=f"Error: El scaler tiene scale_=0 para algunas features ({zero_scale_features}). "
                                   f"Esto significa que estas features tienen varianza cero en los datos de entrenamiento. "
                                   f"Reentrena el modelo sin normalización o elimina estas features."
                        )
                conn.close()
                raise HTTPException(
                    status_code=400,
                    detail="Error: Todos los valores después del scaler son iguales. Esto causará predicciones constantes. "
                           "Reentrena el modelo sin normalización o verifica los datos de entrada."
                )
    
    # Hacer predicción
    try:
        print(f"🔮 Haciendo predicción con {X_input.shape[0]} fila(s) y {X_input.shape[1]} feature(s)")
        print(f"📊 Features finales: {X_input.columns.tolist()}")
        print(f"📊 VALORES FINALES QUE LLEGAN AL MODELO (TODAS las features):")
        for col in X_input.columns:
            print(f"   {col}: {X_input[col].iloc[0]}")
        
        # Crear un hash de los valores para verificar si cambian entre predicciones
        values_tuple = tuple(sorted(X_input.iloc[0].items()))
        values_hash = hash(str(values_tuple))
        print(f"📊 Hash de valores (para verificar cambios): {values_hash}")
        print(f"📊 Valores como tupla: {values_tuple}")
        
        # Verificar si el modelo tiene coeficientes (para debugging)
        if hasattr(model, 'coef_'):
            coef = model.coef_
            print(f"📊 Coeficientes del modelo: {coef}")
            print(f"📊 Tipo de coeficientes: {type(coef)}, Shape: {coef.shape if hasattr(coef, 'shape') else 'N/A'}")
            if hasattr(model, 'intercept_'):
                intercept = model.intercept_
                print(f"📊 Intercepto del modelo: {intercept}")
                print(f"📊 Tipo de intercepto: {type(intercept)}")
            
            # Calcular predicción manualmente para verificar
            if len(coef.shape) == 1 and len(coef) == X_input.shape[1]:
                X_values = X_input.iloc[0].values
                manual_pred = intercept + np.dot(X_values, coef)
                print(f"📊 Predicción manual (intercept + dot product):")
                print(f"   intercept: {intercept:.6f}")
                print(f"   coef: {coef}")
                print(f"   X_values: {X_values}")
                print(f"   dot product: {np.dot(X_values, coef):.6f}")
                print(f"   predicción manual: {manual_pred:.6f}")
        
        # Verificar dimensiones antes de predecir
        if hasattr(model, 'coef_'):
            if X_input.shape[1] != len(model.coef_):
                conn.close()
                raise HTTPException(
                    status_code=400,
                    detail=f"Error de dimensiones: X_input tiene {X_input.shape[1]} features pero el modelo tiene {len(model.coef_)} coeficientes. "
                           f"Features de X_input: {X_input.columns.tolist()}"
                )
            print(f"✅ Dimensiones correctas: X_input tiene {X_input.shape[1]} features, modelo tiene {len(model.coef_)} coeficientes")
        
        # Hacer predicción
        try:
            prediction = model.predict(X_input)
            print(f"📊 Resultado de model.predict: {prediction}")
            print(f"📊 Tipo: {type(prediction)}, Shape: {prediction.shape if hasattr(prediction, 'shape') else 'N/A'}")
            
            if isinstance(prediction, (np.ndarray, list)) and len(prediction) > 0:
                prediction_value = float(prediction[0])
            else:
                prediction_value = float(prediction)
            
            # Verificar que la predicción sea válida
            if np.isnan(prediction_value) or np.isinf(prediction_value):
                conn.close()
                raise HTTPException(
                    status_code=400,
                    detail=f"Error: La predicción es inválida (NaN o Inf). Revisa los valores de entrada y el modelo."
                )
            
            print(f"✅ Predicción exitosa: {prediction_value:.6f}")
            print(f"📊 Hash de valores de entrada: {values_hash}")
            print(f"📊 Si cambias los valores de entrada, este hash debería cambiar")
            
        except Exception as e:
            conn.close()
            import traceback
            error_detail = traceback.format_exc()
            print(f"❌ Error al hacer predicción:\n{error_detail}")
            raise HTTPException(
                status_code=400,
                detail=f"Error al hacer predicción: {str(e)}"
            )
        
        conn.close()
        
        # Obtener las features realmente usadas por el modelo (después de todo el preprocesamiento)
        actual_features_used = X_input.columns.tolist()
        
        return {
            "success": True,
            "prediction": prediction_value,
            "target_column": target_column,
            "features_used": actual_features_used,  # Features realmente usadas por el modelo
            "model_algorithm": algorithm
        }
    except Exception as e:
        conn.close()
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ Error en predicción:\n{error_detail}")
        print(f"❌ Shape de X_input: {X_input.shape}")
        print(f"❌ Columnas de X_input: {X_input.columns.tolist()}")
        raise HTTPException(status_code=400, detail=f"Error al hacer predicción: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

