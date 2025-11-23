# 📖 Manual de Usuario - Gestor de Tablas ML

## Para ChatGPT: Guía de Ayuda al Usuario

Este manual está diseñado para que ChatGPT pueda ayudar al usuario a:
1. Generar datos apropiados para la aplicación
2. Usar correctamente la aplicación para visualización y entrenamiento de modelos
3. Entender el flujo de trabajo de minería de datos

---

## 🎯 Propósito de la Aplicación

La aplicación **Gestor de Tablas - Machine Learning** es una herramienta web para:
- Cargar y editar datos tabulares (CSV, Excel)
- Visualizar relaciones entre variables
- Entrenar modelos de machine learning (regresión)
- Evaluar y guardar modelos entrenados

---

## 📊 Estructura de Datos Recomendada

### Para Generar Datos con ChatGPT

Cuando el usuario pida generar datos, debes crear datasets con estas características:

1. **Formato**: CSV o Excel
2. **Columnas numéricas**: Al menos 2-3 columnas numéricas para características (X)
3. **Variable objetivo**: Una columna numérica que será la variable a predecir (Y)
4. **Tamaño**: Mínimo 50-100 filas para entrenamiento efectivo
5. **Datos realistas**: Valores coherentes y sin errores obvios

### Ejemplo de Estructura de Datos

```csv
id_cliente,edad,ingreso_mensual,visitas_pagina,compras_previas,compra_en_promo
1,56,763263,16,2,1
2,69,845155,15,3,0
3,46,1041202,16,5,1
...
```

**Características (X)**: `edad`, `ingreso_mensual`, `visitas_pagina`, `compras_previas`
**Variable Objetivo (Y)**: `compra_en_promo`

---

## 🔄 Flujo de Trabajo Correcto

### Paso 1: Cargar Datos
1. Usuario hace clic en "📊 Cargar Archivo"
2. Selecciona archivo CSV o Excel
3. La aplicación automáticamente:
   - Detecta tipos de datos
   - Trata valores faltantes
   - Ajusta outliers
   - Codifica variables categóricas

### Paso 2: Explorar y Editar Datos (Opcional)
- El usuario puede editar celdas directamente
- Agregar/eliminar filas o columnas
- Los cambios se guardan automáticamente

### Paso 3: Seleccionar Variables para ML
En el panel derecho, sección "1️⃣ Selección de Variables":

1. **Variable Objetivo (Y)**: 
   - Seleccionar la columna que se quiere predecir
   - Debe ser numérica
   - Ejemplo: `compra_en_promo`, `precio`, `ventas`

2. **Características (X)**:
   - Seleccionar las columnas que se usarán para predecir
   - Marcar con checkboxes las columnas relevantes
   - Mínimo 1 característica, recomendado 2-5
   - Ejemplo: `edad`, `ingreso_mensual`, `visitas_pagina`

### Paso 4: Visualizar Relación (Automático)
- El gráfico se actualiza automáticamente
- Muestra: Primera característica seleccionada (X) vs Variable objetivo (Y)
- El usuario puede cambiar el tipo de gráfico en la sección "2️⃣ Configuración de Visualización"
- **Importante**: Esta visualización ayuda a entender si hay relación entre las variables antes de entrenar

### Paso 5: Configurar y Entrenar Modelo
En la misma sección "1️⃣ Selección de Variables":

1. **Algoritmo**: Seleccionar uno de los 8 algoritmos disponibles
   - **Regresión Lineal Simple**: Para una sola característica
   - **Regresión Lineal Múltiple**: Para múltiples características
   - **Ridge/Lasso**: Para evitar sobreajuste
   - **Random Forest/Gradient Boosting**: Para relaciones no lineales
   - **XGBoost** ⭐: **Recomendado** - Mejor rendimiento en la mayoría de casos
   - **Decision Tree**: Para interpretabilidad

2. **División Train/Test**: Ajustar el porcentaje de datos para test
   - Por defecto: 20% (80% train, 20% test)
   - Rango: 10% a 50% para test
   - Se muestra automáticamente cuántos datos van a cada conjunto
   - **Recomendación**: 20% es estándar, usar más test (30%) si tienes muchos datos

3. **Normalización** (Opcional): Marcar checkbox si quieres normalizar datos
   - Aplica StandardScaler (media=0, desviación=1)
   - Útil cuando las características tienen escalas muy diferentes
   - **Recomendación**: Usar si las características tienen rangos muy distintos (ej: edad 0-100 vs ingreso 0-1000000)

4. **Opciones Avanzadas** (Opcionales):
   - **Selección automática de características**: Selecciona las mejores features (SelectKBest)
   - **Eliminar multicolinealidad**: Elimina características altamente correlacionadas (correlación > 0.95)
   - **Características polinomiales**: Crea interacciones entre features (solo modelos lineales)
   - **Validación cruzada**: Usa K-Fold para estimación más robusta del rendimiento

5. **Entrenar Modelo**: Clic en "🚀 Entrenar Modelo"
   - El modelo se entrena con la división configurada
   - Se aplican las mejoras automáticas seleccionadas
   - Se muestran métricas automáticamente
   - Se generan learning curves automáticamente

### Paso 6: Ver Resultados y Validar Supuestos
Después del entrenamiento:

- **Learning Curves**: Se muestran automáticamente en el gráfico (si hay suficientes datos)
  - Ayuda a detectar overfitting/underfitting visualmente
  - Si las curvas de train y test se separan mucho → Overfitting
  - Si ambas son altas → Underfitting

### Paso 7: Hacer Predicciones (Nuevo)
Después de entrenar un modelo, aparece automáticamente el panel "🔮 Predicción con Modelo":
1. Ingresar valores para cada característica usada en el modelo
2. Clic en "🔮 Predecir"
3. Ver la predicción del modelo para la variable objetivo
- **Métricas** aparecen en la misma sección:
  - **R² Score (Train)**: Qué tan bien explica el modelo en datos de entrenamiento
  - **R² Score (Test)**: Qué tan bien explica el modelo en datos nuevos (MÁS IMPORTANTE)
  - **RMSE (Test)**: Error promedio en datos de prueba (más bajo mejor)
  - **MAE (Test)**: Error absoluto promedio en datos de prueba (más bajo mejor)
  - **División de datos**: Muestra cantidad y porcentaje de datos en train y test

- **Validación de Supuestos** (solo para regresión lineal):
  - **Test de Normalidad (Shapiro-Wilk)**: Verifica si los residuos siguen distribución normal
    - "Normal" (p > 0.05): Los residuos son normales ✅
    - "No normal" (p ≤ 0.05): Los residuos no son normales ⚠️
  - **Estadísticas de residuos**: Media y desviación estándar
  - **Gráfico de residuos**: Marcar checkbox "Ver gráfico de residuos" para visualizar
    - Los residuos deben estar distribuidos aleatoriamente alrededor de 0
    - Si hay patrones (curvas, conos), el modelo no cumple supuestos

- **Gráfico** cambia automáticamente a "Real vs Predicho" o muestra "Learning Curves"
  - **Learning Curves**: Muestra cómo el error cambia con el tamaño de entrenamiento
    - Si las curvas se separan mucho → Overfitting (sobreajuste)
    - Si ambas curvas son altas → Underfitting (subajuste)
    - Si ambas convergen y son bajas → Buen ajuste
- **Panel de Predicción**: Aparece automáticamente después del entrenamiento
  - Permite ingresar valores para las características
  - Obtiene predicción instantánea del modelo entrenado
- El modelo se guarda automáticamente en SQLite

---

## 🎨 Tipos de Gráficos Disponibles

1. **📊 Dispersión**: Para ver correlación entre dos variables
2. **📈 Línea**: Para tendencias temporales o secuenciales
3. **📊 Barras**: Para comparar categorías
4. **📉 Área**: Similar a línea con área rellena
5. **🥧 Pastel**: Para proporciones
6. **🔀 Combinado**: Combina área, barras y línea
7. **🕸️ Radar**: Para múltiples variables
8. **🗺️ Treemap**: Visualización jerárquica

**Recomendación**: Usar **Dispersión** o **Línea** para análisis de regresión.

---

## 💡 Consejos para ChatGPT al Ayudar al Usuario

### Al Generar Datos:
1. **Pregunta el contexto**: ¿Qué quiere predecir? ¿Qué variables tiene disponibles?
2. **Genera datos realistas**: Valores coherentes con el dominio
3. **Incluye variabilidad**: No todos los valores iguales
4. **Asegura relación**: Si es para ML, las características deben tener alguna relación con el target
5. **Formato correcto**: CSV con encabezados, valores numéricos donde corresponde

### Al Explicar el Uso:
1. **Enfatiza el flujo**: Variables → Visualizar → Entrenar
2. **Explica las métricas**: Qué significa R², RMSE, MAE
3. **Sugiere algoritmos**: Según el tipo de problema
4. **Interpreta gráficos**: Ayuda a entender qué muestra cada gráfico
5. **Valida selección**: Verifica que las variables seleccionadas tengan sentido

### Al Interpretar Resultados:
1. **R² Score (Test) > 0.7**: Buen modelo
2. **R² Score (Test) 0.5-0.7**: Modelo aceptable
3. **R² Score (Test) < 0.5**: Modelo pobre, revisar variables
4. **Compara Train vs Test**: Si R² Train >> R² Test, hay sobreajuste
5. **Compara algoritmos**: Sugiere probar diferentes algoritmos
6. **Revisa el gráfico**: Si Real vs Predicho está disperso, el modelo no es bueno
7. **Validación de supuestos**: Para regresión lineal, verificar normalidad de residuos
   - Si no son normales, considerar transformaciones o algoritmos no lineales

---

## 🔍 Ejemplos de Uso

### Ejemplo 1: Predicción de Ventas
**Variables**:
- Características (X): `publicidad`, `precio`, `temporada`
- Objetivo (Y): `ventas`

**Flujo**:
1. Cargar datos con estas columnas
2. Seleccionar `ventas` como objetivo
3. Seleccionar `publicidad`, `precio`, `temporada` como características
4. Visualizar relación (automático)
5. Configurar división Train/Test (20% por defecto)
6. Decidir si normalizar (si las escalas son muy diferentes)
7. Entrenar con "Regresión Lineal Múltiple"
8. Ver métricas (Train y Test), validación de supuestos
9. Ver gráfico Real vs Predicho y opcionalmente gráfico de residuos

### Ejemplo 2: Predicción de Precio
**Variables**:
- Características (X): `metros_cuadrados`, `habitaciones`, `años_construccion`
- Objetivo (Y): `precio`

**Flujo**:
1. Cargar datos inmobiliarios
2. Seleccionar `precio` como objetivo
3. Seleccionar características relevantes
4. Visualizar relación
5. Configurar división Train/Test
6. Considerar normalización (precios y metros pueden tener escalas diferentes)
7. Entrenar con "Random Forest" (mejor para relaciones no lineales)
8. Evaluar resultados: métricas, validación de supuestos (si aplica)
9. Comparar con otros algoritmos si es necesario

---

## ⚠️ Errores Comunes y Soluciones

### Error: "Seleccione variable objetivo y al menos una característica"
**Solución**: Asegurar que se seleccionó:
- Una variable objetivo (Y)
- Al menos una característica (X) marcada

### Error: "No hay datos válidos después de la limpieza"
**Solución**: 
- Verificar que las columnas seleccionadas tengan datos numéricos
- Revisar que no todas las filas tengan valores faltantes

### Gráfico vacío
**Solución**:
- Verificar que se seleccionaron características y variable objetivo
- Asegurar que los datos tienen valores numéricos válidos

### R² Score muy bajo (< 0.3)
**Solución**:
- Las características seleccionadas pueden no tener relación con el objetivo
- Probar diferentes características
- Considerar transformaciones de datos
- Probar algoritmos más complejos (Random Forest, Gradient Boosting)

---

## 📚 Conceptos Clave para Explicar

### Variable Objetivo (Y)
- Es lo que queremos predecir
- Debe ser numérica para regresión
- Ejemplos: precio, ventas, temperatura, tiempo

### Características (X)
- Son las variables que usamos para predecir
- Pueden ser múltiples
- Deben tener relación con el objetivo
- Ejemplos: edad, ingresos, tamaño, ubicación

### R² Score
- Mide qué tan bien el modelo explica la variabilidad
- Rango: 0 a 1 (o negativo si es muy malo)
- 1.0 = perfecto, 0.0 = no explica nada
- > 0.7 = bueno, > 0.5 = aceptable

### Train/Test Split
- Configurable: 10% a 50% para test (por defecto 20%)
- Evita sobreajuste
- Las métricas de "Test" son las importantes
- Si R² Train >> R² Test, hay sobreajuste (overfitting)

### Normalización
- StandardScaler: Escala datos a media 0 y desviación estándar 1
- Útil cuando características tienen escalas muy diferentes
- No siempre es necesario, depende del algoritmo y datos

### Validación de Supuestos (Regresión Lineal)
- **Normalidad de residuos**: Test de Shapiro-Wilk
  - p > 0.05: Residuos normales ✅
  - p ≤ 0.05: Residuos no normales ⚠️
- **Gráfico de residuos**: Debe mostrar distribución aleatoria alrededor de 0
- Si no se cumplen supuestos, considerar:
  - Transformaciones de datos
  - Algoritmos no lineales (Random Forest, Gradient Boosting)

---

## 🎓 Guía para ChatGPT: Preguntas Frecuentes

### "¿Qué datos necesito?"
- Datos tabulares (CSV/Excel)
- Al menos 50-100 filas
- Columnas numéricas para características y objetivo
- Datos realistas y coherentes

### "¿Qué algoritmo elegir?"
- **XGBoost** ⭐: **Recomendado** - Mejor rendimiento en la mayoría de casos, maneja relaciones no lineales
- **Regresión Lineal Múltiple**: Para empezar, relaciones lineales, interpretable
- **Ridge/Lasso**: Si hay muchas características, evitar sobreajuste
- **Random Forest/Gradient Boosting**: Si la relación no es lineal, robusto
- **Decision Tree**: Si necesitas interpretabilidad máxima

### "¿Por qué mi modelo tiene R² bajo?"
- Las características no tienen relación con el objetivo
- Necesitas más datos
- Necesitas diferentes características
- La relación no es lineal (probar Random Forest)

### "¿Cómo interpreto el gráfico Real vs Predicho?"
- Si los puntos están cerca de la línea diagonal = buen modelo
- Si están dispersos = mal modelo
- Si hay patrones = el modelo no captura algo importante

### "¿Cuándo usar normalización?"
- Cuando las características tienen escalas muy diferentes (ej: edad 0-100 vs ingreso 0-1000000)
- Generalmente útil para regresión lineal
- Random Forest y Decision Tree no necesitan normalización
- Si no estás seguro, prueba con y sin normalización

### "¿Qué porcentaje usar para test?"
- **20% (por defecto)**: Estándar, funciona bien en la mayoría de casos
- **30%**: Si tienes muchos datos (>1000 filas) y quieres más confianza
- **10-15%**: Si tienes pocos datos (<200 filas) y necesitas más para entrenar
- **No usar >50%**: Dejas muy poco para entrenar

### "¿Qué significa que los residuos no sean normales?"
- Los residuos deberían seguir una distribución normal para regresión lineal
- Si no son normales (p ≤ 0.05), el modelo puede tener sesgos
- Soluciones:
  - Transformar la variable objetivo (log, sqrt)
  - Usar algoritmos no lineales (Random Forest, Gradient Boosting)
  - Revisar si hay outliers o datos erróneos

### "¿Cómo interpreto el gráfico de residuos?"
- **Bien**: Residuos distribuidos aleatoriamente alrededor de 0, sin patrones
- **Mal**: 
  - Patrón de embudo: Varianza no constante (heterocedasticidad)
  - Curva: Relación no lineal no capturada
  - Tendencia: El modelo no captura algo importante

---

## 🆕 Nuevas Funcionalidades

### XGBoost - Algoritmo Recomendado
- **XGBoost** es ahora el algoritmo recomendado (marcado con ⭐)
- Generalmente proporciona mejor rendimiento que otros algoritmos
- Maneja relaciones no lineales de forma efectiva
- Hiperparámetros optimizados automáticamente según el tamaño de datos

### Learning Curves (Curvas de Aprendizaje)
- Se generan automáticamente después del entrenamiento (si hay suficientes datos)
- Muestran cómo el error cambia con el tamaño del conjunto de entrenamiento
- **Interpretación**:
  - Curvas que convergen y son bajas → Buen modelo
  - Curvas que se separan mucho → Overfitting (sobreajuste)
  - Ambas curvas altas → Underfitting (subajuste)
- Ayuda a diagnosticar problemas del modelo visualmente

### Predicción de Nuevos Valores
- Panel interactivo que aparece automáticamente después de entrenar
- Permite ingresar valores para las características del modelo
- Obtiene predicción instantánea de la variable objetivo
- Útil para usar el modelo entrenado en producción

### Mejoras Automáticas Avanzadas
- **Eliminación de multicolinealidad**: Elimina características altamente correlacionadas (correlación > 0.95)
- **Selección automática de características**: Selecciona las mejores features (SelectKBest)
- **Variables derivadas**: Crea automáticamente nuevas features (multiplicaciones, divisiones, cuadrados, ratios)
- **Transformaciones logarítmicas**: Aplica log a variables altamente sesgadas
- **Características polinomiales**: Crea interacciones entre features (solo modelos lineales)
- **Validación cruzada**: Opción para K-Fold cross-validation

### Comparación de Modelos
- El endpoint `/api/models` ahora incluye métricas completas
- Permite comparar múltiples modelos entrenados
- Ordenamiento automático por R² test (mejor primero)

## 🔧 Funcionalidades Técnicas

### Preprocesamiento Automático
- **Duplicados**: Se eliminan automáticamente al cargar
- **Valores faltantes**: Se rellenan con mediana (numéricos) o moda (categóricos)
- **Outliers**: Se ajustan (no se eliminan) usando método IQR
- **Infinitos**: Se eliminan y rellenan
- **Categóricas**: Se codifican automáticamente con LabelEncoder
- **Variables derivadas**: Se crean automáticamente (multiplicaciones, divisiones, cuadrados, ratios)
- **Transformaciones logarítmicas**: Se aplican a variables altamente sesgadas (skewness > 1.5)

### Preprocesamiento Manual (Opcional)
- **Normalización**: Opción para aplicar StandardScaler antes del entrenamiento
- **División Train/Test**: Configurable entre 10% y 50% para test
- **Edición de datos**: CRUD completo en la tabla antes del entrenamiento
- **Selección de características**: Opción para selección automática (SelectKBest)
- **Eliminación de multicolinealidad**: Opción para eliminar features correlacionadas (correlación > 0.95)
- **Características polinomiales**: Opción para crear interacciones (solo modelos lineales)
- **Validación cruzada**: Opción para usar K-Fold cross-validation

### Guardado de Modelos
- Se guardan en SQLite (`backend/modelos.db`)
- Incluyen: algoritmo, métricas (train y test), características, modelo serializado
- Incluyen: información de división de datos, estadísticas de residuos
- Incluyen: learning curves, feature importance, mejoras aplicadas
- Se pueden usar para hacer predicciones después
- Se pueden comparar con otros modelos entrenados

### Actualización Automática
- El gráfico se actualiza cuando cambias variables
- No necesitas recargar la página
- Los cambios en la tabla se reflejan inmediatamente
- Las learning curves se generan automáticamente después del entrenamiento
- El panel de predicción aparece automáticamente después de entrenar

---

## 💻 Compatibilidad con Windows

La aplicación **funciona perfectamente en Windows**. 

### Inicio en Windows:
1. **Opción 1 (Recomendada)**: Hacer doble clic en `start.bat`
2. **Opción 2**: Abrir PowerShell o CMD en la carpeta del proyecto y ejecutar:
   ```cmd
   start.bat
   ```

### Diferencias con Linux/Mac:
- En Windows, el script `start.bat` abre ventanas separadas para backend y frontend
- Los comandos de Python son los mismos (`python` en lugar de `python3`)
- La activación del entorno virtual es: `venv\Scripts\activate` (en lugar de `source venv/bin/activate`)

### Requisitos en Windows:
- Python 3.8+ instalado (descargar desde python.org)
- Node.js 16+ instalado (descargar desde nodejs.org)
- Asegurarse de marcar "Add Python to PATH" durante la instalación

## 📞 Soporte

Si el usuario tiene problemas:
1. Verificar que el backend esté corriendo (http://localhost:8000)
2. Verificar que el frontend esté corriendo (http://localhost:3000)
3. Revisar la consola del navegador (F12) para errores
4. Verificar que los datos tengan el formato correcto
5. **En Windows**: Verificar que las ventanas de backend y frontend estén abiertas

---

**Nota para ChatGPT**: Usa este manual para guiar al usuario paso a paso, generar datos apropiados según su necesidad, y ayudarle a interpretar los resultados de manera profesional.
