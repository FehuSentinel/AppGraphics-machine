import React, { useState, useEffect } from 'react'
import { trainModel, getCorrelations, predictWithModel } from '../services/api'
import PredictionPanel from './PredictionPanel'
import './MLPanel.css'

const MLControls = ({ sessionId, data, columns, mlState, onMlStateUpdate }) => {
  const [training, setTraining] = useState(false)
  const [correlations, setCorrelations] = useState(null)
  const [showCorrelations, setShowCorrelations] = useState(false)

  useEffect(() => {
    if (columns.length > 0) {
      // Auto-seleccionar última columna como target
      if (!mlState.targetColumn) {
        onMlStateUpdate({ targetColumn: columns[columns.length - 1] })
      }
      // Auto-seleccionar todas las demás como features
      if (!mlState.selectedFeatures || mlState.selectedFeatures.length === 0) {
        onMlStateUpdate({ selectedFeatures: columns.slice(0, -1) })
      }
      // Auto-seleccionar ejes
      if (columns.length >= 2 && !mlState.xAxis) {
        onMlStateUpdate({ 
          xAxis: columns[0],
          yAxis: columns[columns.length - 1]
        })
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [columns])

  // Cargar correlaciones cuando cambia el target
  useEffect(() => {
    const loadCorrelations = async () => {
      if (sessionId && mlState.targetColumn) {
        try {
          const result = await getCorrelations(sessionId, mlState.targetColumn)
          setCorrelations(result.target_correlations)
        } catch (error) {
          console.error('Error al cargar correlaciones:', error)
          setCorrelations(null)
        }
      } else {
        setCorrelations(null)
      }
    }
    loadCorrelations()
  }, [sessionId, mlState.targetColumn])

  const handleFeatureToggle = (feature) => {
    // No permitir seleccionar la variable objetivo como característica
    if (feature === mlState.targetColumn) {
      alert('La variable objetivo no puede ser una característica')
      return
    }
    
    const currentFeatures = mlState.selectedFeatures || []
    const newFeatures = currentFeatures.includes(feature)
      ? currentFeatures.filter(f => f !== feature)
      : [...currentFeatures, feature]
    onMlStateUpdate({ selectedFeatures: newFeatures })
  }

  const handleTrain = async () => {
    let features = mlState.selectedFeatures || []
    
    // Filtrar la variable objetivo de las características si está incluida
    features = features.filter(f => f !== mlState.targetColumn)
    
    if (!sessionId || !mlState.targetColumn || features.length === 0) {
      alert('Seleccione variable objetivo y al menos una característica')
      return
    }

    setTraining(true)
    try {
        console.log('🚀 Iniciando entrenamiento...', {
          session_id: sessionId,
          algorithm: mlState.algorithm,
          target_column: mlState.targetColumn,
          features: features
        })
        
        const result = await trainModel({
          session_id: sessionId,
          algorithm: mlState.algorithm,
          target_column: mlState.targetColumn,
          features: features,
          test_size: mlState.testSize || 0.2,
          random_state: 42,
          normalize: mlState.normalizeData || false,
          auto_feature_selection: true,  // Selección automática de mejores features
          remove_multicollinearity: true,  // Eliminar features altamente correlacionadas
          use_polynomial_features: false  // Features polinomiales (opcional, desactivado por defecto)
        })
      
      console.log('📥 Respuesta recibida:', result)
      
      // Verificar que result no sea null o undefined
      if (!result || result === null || result === undefined) {
        throw new Error('El servidor no devolvió una respuesta válida (result es null/undefined)')
      }
      
      if (result.success) {
        console.log('✅ Modelo entrenado exitosamente:', result)
        onMlStateUpdate({ modelResults: result })
        alert('✅ Modelo entrenado exitosamente!')
      } else {
        alert('Error: ' + (result.error || 'Error desconocido'))
      }
    } catch (error) {
      console.error('❌ Error al entrenar modelo:', error)
      console.error('Error completo:', error.response)
      console.error('Error status:', error.response?.status)
      console.error('Error data:', error.response?.data)
      const errorMessage = error.response?.data?.detail || error.response?.data?.message || error.message || 'Error desconocido'
      console.error('Mensaje de error:', errorMessage)
      alert('Error al entrenar modelo: ' + errorMessage)
    } finally {
      setTraining(false)
    }
  }

  if (!sessionId) {
    return (
      <div className="ml-panel">
        <div className="ml-header">
          <h2>🤖 Machine Learning</h2>
        </div>
        <div className="ml-content">
          <div className="ml-section">
            <p style={{ textAlign: 'center', color: '#666' }}>
              Carga un archivo para comenzar
            </p>
          </div>
        </div>
      </div>
    )
  }

  const numericColumns = columns.filter(col => {
    if (data.length === 0) return false
    const sample = data[0][col]
    return typeof sample === 'number' || !isNaN(parseFloat(sample))
  })

  return (
    <div className="ml-panel">
      <div className="ml-header">
        <h2>⚙️ Configuración de Modelo</h2>
      </div>

      <div className="ml-content">
        {/* 1. Selección de Variables */}
        <div className="ml-section">
          <h3>1️⃣ Selección de Variables</h3>
          
          <div className="form-group">
            <label>Variable Objetivo (Y):</label>
            <select 
              value={mlState.targetColumn} 
              onChange={(e) => onMlStateUpdate({ targetColumn: e.target.value })}
              className="form-select"
            >
              <option value="">Seleccione...</option>
              {numericColumns.map(col => (
                <option key={col} value={col}>{col}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>
              Características (X):
              {mlState.targetColumn && correlations && (
                <button
                  onClick={() => setShowCorrelations(!showCorrelations)}
                  style={{
                    marginLeft: '10px',
                    padding: '2px 8px',
                    fontSize: '10px',
                    background: '#3498DB',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}
                >
                  {showCorrelations ? 'Ocultar' : 'Mostrar'} Correlaciones
                </button>
              )}
            </label>
            {mlState.targetColumn && correlations && showCorrelations && (
              <div style={{
                background: '#EBF5FB',
                padding: '8px',
                borderRadius: '6px',
                marginBottom: '10px',
                fontSize: '11px',
                border: '1px solid #D6EAF8'
              }}>
                <strong>📊 Correlaciones con {mlState.targetColumn}:</strong>
                <div style={{ marginTop: '5px', maxHeight: '100px', overflowY: 'auto' }}>
                  {Object.entries(correlations).slice(0, 10).map(([feature, corr]) => {
                    const absCorr = Math.abs(corr)
                    const color = absCorr > 0.7 ? '#27AE60' : absCorr > 0.4 ? '#F39C12' : '#95A5A6'
                    return (
                      <div key={feature} style={{ 
                        display: 'flex', 
                        justifyContent: 'space-between',
                        padding: '2px 0',
                        borderBottom: '1px solid #D6EAF8'
                      }}>
                        <span>{feature}:</span>
                        <span style={{ 
                          color, 
                          fontWeight: 'bold',
                          background: absCorr > 0.5 ? 'rgba(255,255,255,0.7)' : 'transparent',
                          padding: '0 4px',
                          borderRadius: '3px'
                        }}>
                          {corr.toFixed(3)}
                        </span>
                      </div>
                    )
                  })}
                </div>
                <small style={{ color: '#666', display: 'block', marginTop: '5px' }}>
                  💡 Selecciona características con alta correlación (|r| &gt; 0.4) para mejores predicciones
                </small>
              </div>
            )}
            <div className="features-list">
              {numericColumns
                .filter(col => col !== mlState.targetColumn)
                .map(col => {
                  const currentFeatures = mlState.selectedFeatures || []
                  // Filtrar la variable objetivo de las características seleccionadas
                  const filteredFeatures = currentFeatures.filter(f => f !== mlState.targetColumn)
                  const isSelected = filteredFeatures.includes(col)
                  const corr = correlations && correlations[col] ? correlations[col] : null
                  const absCorr = corr ? Math.abs(corr) : 0
                  const highlight = absCorr > 0.4 && !isSelected
                  
                  return (
                    <label 
                      key={col} 
                      className="checkbox-label"
                      style={{
                        background: highlight ? '#FFF3CD' : 'transparent',
                        padding: highlight ? '4px' : '0',
                        borderRadius: highlight ? '4px' : '0',
                        border: highlight ? '1px solid #FFC107' : 'none'
                      }}
                    >
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => handleFeatureToggle(col)}
                      />
                      <span style={{ flex: 1 }}>{col}</span>
                      {corr !== null && (
                        <span style={{
                          fontSize: '10px',
                          color: absCorr > 0.7 ? '#27AE60' : absCorr > 0.4 ? '#F39C12' : '#95A5A6',
                          fontWeight: 'bold',
                          marginLeft: '8px'
                        }}>
                          r={corr.toFixed(2)}
                        </span>
                      )}
                    </label>
                  )
                })}
            </div>
            {mlState.targetColumn && (
              <small style={{ color: '#666', fontSize: '10px', display: 'block', marginTop: '5px' }}>
                Variable objetivo: <strong>{mlState.targetColumn}</strong> (no se incluye en características)
              </small>
            )}
          </div>

              <div className="form-group">
                <label>Algoritmo:</label>
                <select 
                  value={mlState.algorithm} 
                  onChange={(e) => onMlStateUpdate({ algorithm: e.target.value })}
                  className="form-select"
                >
                  <option value="Regresión Lineal Simple">Regresión Lineal Simple</option>
                  <option value="Regresión Lineal Múltiple">Regresión Lineal Múltiple</option>
                  <option value="Ridge Regression">Ridge Regression</option>
                  <option value="Lasso Regression">Lasso Regression</option>
                  <option value="Random Forest">Random Forest</option>
                  <option value="Gradient Boosting">Gradient Boosting</option>
                  <option value="XGBoost">XGBoost (Recomendado) ⭐</option>
                  <option value="Decision Tree">Decision Tree</option>
                </select>
                <small style={{ color: '#666', fontSize: '10px' }}>
                  💡 XGBoost suele dar mejores resultados en la mayoría de casos
                </small>
              </div>

          <div className="form-group">
            <label>División Train/Test (% Test):</label>
            <input
              type="number"
              min="0.1"
              max="0.5"
              step="0.05"
              value={mlState.testSize || 0.2}
              onChange={(e) => onMlStateUpdate({ testSize: parseFloat(e.target.value) })}
              className="form-select"
              style={{ width: '100%' }}
            />
            <small style={{ color: '#666', fontSize: '10px' }}>
              {mlState.testSize ? `Train: ${((1 - mlState.testSize) * 100).toFixed(0)}% | Test: ${(mlState.testSize * 100).toFixed(0)}%` : 'Train: 80% | Test: 20%'}
            </small>
          </div>

          <div className="form-group">
            <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={mlState.normalizeData || false}
                onChange={(e) => onMlStateUpdate({ normalizeData: e.target.checked })}
                style={{ marginRight: '8px', width: 'auto' }}
              />
              Normalizar datos (StandardScaler)
            </label>
            <small style={{ color: '#666', fontSize: '10px' }}>
              Escala las características a media 0 y desviación 1
            </small>
          </div>

          <button 
            className="btn-train"
            onClick={handleTrain}
            disabled={training || !mlState.targetColumn || !mlState.selectedFeatures || (mlState.selectedFeatures && mlState.selectedFeatures.length === 0)}
          >
            {training ? '⏳ Entrenando...' : '🚀 Entrenar Modelo'}
          </button>

          {/* Resultados del Modelo */}
          {mlState.modelResults && (
            <div className="results" style={{ marginTop: '15px' }}>
              <h4 style={{ margin: '0 0 10px 0', fontSize: '13px' }}>📈 Métricas del Modelo (Evaluación):</h4>
              
              {/* R² Score */}
              <div style={{ marginBottom: '8px', padding: '8px', background: '#F8F9FA', borderRadius: '4px' }}>
                <p style={{ margin: '4px 0', fontSize: '12px' }}>
                  <strong>R² Score (Coeficiente de Determinación):</strong>
                </p>
                <p style={{ margin: '2px 0', fontSize: '11px', color: '#666' }}>
                  Train: <strong style={{ color: mlState.modelResults.metrics.train_r2 > 0.7 ? '#27AE60' : mlState.modelResults.metrics.train_r2 > 0.4 ? '#F39C12' : '#E74C3C' }}>
                    {mlState.modelResults.metrics.train_r2.toFixed(4)}
                  </strong> | 
                  Test: <strong style={{ color: mlState.modelResults.metrics.test_r2 > 0.7 ? '#27AE60' : mlState.modelResults.metrics.test_r2 > 0.4 ? '#F39C12' : '#E74C3C' }}>
                    {mlState.modelResults.metrics.test_r2.toFixed(4)}
                  </strong>
                </p>
                <small style={{ color: '#666', fontSize: '10px' }}>
                  {mlState.modelResults.metrics.test_r2 > 0.7 ? '✅ Excelente' : 
                   mlState.modelResults.metrics.test_r2 > 0.4 ? '⚠️ Aceptable' : 
                   '❌ Bajo - El modelo explica menos del 40% de la variación'}
                </small>
              </div>

              {/* MSE y RMSE */}
              <div style={{ marginBottom: '8px', padding: '8px', background: '#F8F9FA', borderRadius: '4px' }}>
                <p style={{ margin: '4px 0', fontSize: '12px' }}>
                  <strong>MSE (Mean Squared Error):</strong>
                </p>
                <p style={{ margin: '2px 0', fontSize: '11px', color: '#666' }}>
                  Train: {mlState.modelResults.metrics.train_mse?.toFixed(2) || 'N/A'} | 
                  Test: {mlState.modelResults.metrics.test_mse?.toFixed(2) || 'N/A'}
                </p>
                <p style={{ margin: '4px 0', fontSize: '12px' }}>
                  <strong>RMSE (Root Mean Squared Error):</strong>
                </p>
                <p style={{ margin: '2px 0', fontSize: '11px', color: '#666' }}>
                  Train: <strong>{mlState.modelResults.metrics.train_rmse.toFixed(2)}</strong> | 
                  Test: <strong>{mlState.modelResults.metrics.test_rmse.toFixed(2)}</strong>
                </p>
                {mlState.modelResults.metrics.mean_y_test && (
                  <small style={{ color: '#666', fontSize: '10px' }}>
                    💡 El modelo se equivoca en promedio <strong>{mlState.modelResults.metrics.test_rmse.toFixed(2)}</strong> unidades
                    {mlState.modelResults.metrics.error_percentual && 
                      ` (${mlState.modelResults.metrics.error_percentual.toFixed(1)}% del valor promedio)`}
                  </small>
                )}
              </div>

              {/* MAE */}
              <div style={{ marginBottom: '8px', padding: '8px', background: '#F8F9FA', borderRadius: '4px' }}>
                <p style={{ margin: '4px 0', fontSize: '12px' }}>
                  <strong>MAE (Mean Absolute Error):</strong>
                </p>
                <p style={{ margin: '2px 0', fontSize: '11px', color: '#666' }}>
                  Train: {mlState.modelResults.metrics.train_mae.toFixed(2)} | 
                  Test: {mlState.modelResults.metrics.test_mae.toFixed(2)}
                </p>
              </div>

              {/* MAPE si está disponible */}
              {mlState.modelResults.metrics.mape_test !== null && mlState.modelResults.metrics.mape_test !== undefined && (
                <div style={{ marginBottom: '8px', padding: '8px', background: '#F8F9FA', borderRadius: '4px' }}>
                  <p style={{ margin: '4px 0', fontSize: '12px' }}>
                    <strong>MAPE (Mean Absolute Percentage Error):</strong>
                  </p>
                  <p style={{ margin: '2px 0', fontSize: '11px', color: '#666' }}>
                    Test: <strong>{mlState.modelResults.metrics.mape_test.toFixed(2)}%</strong>
                  </p>
                </div>
              )}

              {/* Detección de Overfitting */}
              {mlState.modelResults.metrics.r2_diff !== undefined && (
                <div style={{ 
                  marginBottom: '8px', 
                  padding: '8px', 
                  background: mlState.modelResults.metrics.r2_diff > 0.1 ? '#FFEBEE' : '#E8F5E9', 
                  borderRadius: '4px',
                  border: `1px solid ${mlState.modelResults.metrics.r2_diff > 0.1 ? '#E74C3C' : '#27AE60'}`
                }}>
                  <p style={{ margin: '4px 0', fontSize: '12px' }}>
                    <strong>🔍 Análisis de Overfitting:</strong>
                  </p>
                  <p style={{ margin: '2px 0', fontSize: '11px', color: '#666' }}>
                    Diferencia R² (Train - Test): <strong>{mlState.modelResults.metrics.r2_diff.toFixed(4)}</strong>
                  </p>
                  <small style={{ color: '#666', fontSize: '10px' }}>
                    {mlState.modelResults.metrics.r2_diff > 0.1 
                      ? '⚠️ Posible sobreajuste: El modelo funciona mejor en train que en test'
                      : '✅ Buen ajuste: El modelo generaliza bien'}
                  </small>
                </div>
              )}

              {/* Cross-Validation Score */}
              {mlState.modelResults.metrics.cv_r2_mean !== undefined && (
                <div style={{ marginBottom: '8px', padding: '8px', background: '#E8F5E9', borderRadius: '4px' }}>
                  <p style={{ margin: '4px 0', fontSize: '12px' }}>
                    <strong>✅ Validación Cruzada (Cross-Validation):</strong>
                  </p>
                  <p style={{ margin: '2px 0', fontSize: '11px', color: '#666' }}>
                    R² CV: <strong>{mlState.modelResults.metrics.cv_r2_mean.toFixed(4)}</strong> 
                    (±{mlState.modelResults.metrics.cv_r2_std.toFixed(4)})
                  </p>
                  <small style={{ color: '#666', fontSize: '10px' }}>
                    Estimación más robusta del rendimiento del modelo
                  </small>
                </div>
              )}

              {/* Feature Importance */}
              {mlState.modelResults.feature_importance && Object.keys(mlState.modelResults.feature_importance).length > 0 && (
                <div style={{ marginBottom: '8px', padding: '8px', background: '#F8F9FA', borderRadius: '4px' }}>
                  <p style={{ margin: '4px 0', fontSize: '12px' }}>
                    <strong>🎯 Importancia de Features (Top 5):</strong>
                  </p>
                  <div style={{ fontSize: '11px', color: '#666' }}>
                    {Object.entries(mlState.modelResults.feature_importance)
                      .slice(0, 5)
                      .map(([feature, importance], idx) => (
                        <div key={feature} style={{ 
                          display: 'flex', 
                          justifyContent: 'space-between',
                          padding: '2px 0',
                          borderBottom: idx < 4 ? '1px solid #E1E8ED' : 'none'
                        }}>
                          <span>{feature}:</span>
                          <span style={{ fontWeight: 'bold', color: '#3498DB' }}>
                            {(importance * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                  </div>
                </div>
              )}

              {/* Mejoras Aplicadas */}
              {mlState.modelResults.improvements_applied && (
                <div style={{ marginBottom: '8px', padding: '8px', background: '#EBF5FB', borderRadius: '4px', border: '1px solid #D6EAF8' }}>
                  <p style={{ margin: '4px 0', fontSize: '12px' }}>
                    <strong>⚙️ Mejoras Automáticas Aplicadas:</strong>
                  </p>
                  <div style={{ fontSize: '10px', color: '#666', marginTop: '5px' }}>
                    {mlState.modelResults.improvements_applied.derived_variables && (
                      <div>✅ Variables derivadas creadas</div>
                    )}
                    {mlState.modelResults.improvements_applied.log_transformations && (
                      <div>✅ Transformaciones log aplicadas</div>
                    )}
                    {mlState.modelResults.improvements_applied.multicollinearity_removed && (
                      <div>✅ Multicolinealidad eliminada</div>
                    )}
                    {mlState.modelResults.improvements_applied.feature_selection && (
                      <div>✅ Feature selection automática</div>
                    )}
                    {mlState.modelResults.improvements_applied.polynomial_features && (
                      <div>✅ Features polinomiales creadas</div>
                    )}
                  </div>
                </div>
              )}
              
              {mlState.modelResults.data_split && (
                <div style={{ marginTop: '10px', paddingTop: '10px', borderTop: '1px solid #BBDEFB' }}>
                  <p style={{ fontSize: '11px', margin: '5px 0' }}>
                    <strong>División:</strong> Train: {mlState.modelResults.data_split.train_size} ({((mlState.modelResults.data_split.train_percentage) * 100).toFixed(0)}%) | 
                    Test: {mlState.modelResults.data_split.test_size} ({((mlState.modelResults.data_split.test_percentage) * 100).toFixed(0)}%)
                  </p>
                </div>
              )}
              
              {mlState.modelResults.residuals_stats && mlState.modelResults.residuals_stats.normality_test !== 'No aplicable' && (
                <div style={{ marginTop: '10px', paddingTop: '10px', borderTop: '1px solid #BBDEFB' }}>
                  <p style={{ fontSize: '11px', margin: '5px 0' }}>
                    <strong>✅ Validación Supuestos:</strong>
                  </p>
                  <p style={{ fontSize: '11px', margin: '3px 0' }}>
                    Normalidad (Shapiro-Wilk): <strong>{mlState.modelResults.residuals_stats.normality_test}</strong>
                    {mlState.modelResults.residuals_stats.shapiro_pvalue !== null && 
                      ` (p=${mlState.modelResults.residuals_stats.shapiro_pvalue.toFixed(4)})`}
                  </p>
                  <p style={{ fontSize: '11px', margin: '3px 0' }}>
                    Media residuos: {mlState.modelResults.residuals_stats.mean.toFixed(4)} | 
                    Std: {mlState.modelResults.residuals_stats.std.toFixed(4)}
                  </p>
                  <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', marginTop: '8px' }}>
                    <input
                      type="checkbox"
                      checked={mlState.showResiduals || false}
                      onChange={(e) => onMlStateUpdate({ showResiduals: e.target.checked })}
                      style={{ marginRight: '8px', width: 'auto' }}
                    />
                    Ver gráfico de residuos
                  </label>
                </div>
              )}
            </div>
          )}

          {/* Panel de Predicción - Se muestra después de entrenar */}
          {mlState.modelResults && mlState.modelResults.model_id && (
            <div className="ml-section" style={{ 
              marginTop: '20px', 
              border: '2px solid #27AE60', 
              borderRadius: '8px',
              padding: '15px',
              background: 'linear-gradient(135deg, #F8FFF9 0%, #E8F5E9 100%)'
            }}>
              <h3 style={{ margin: '0 0 15px 0', fontSize: '14px', color: '#27AE60' }}>
                🔮 Probar Modelo - Hacer Predicción
              </h3>
              <p style={{ fontSize: '11px', color: '#666', marginBottom: '15px' }}>
                Ingresa valores para las características y el modelo predecirá el valor de <strong>{mlState.targetColumn}</strong>
              </p>
              <PredictionPanel
                modelId={mlState.modelResults.model_id}
                features={mlState.modelResults.selected_features || mlState.selectedFeatures || []}
                targetColumn={mlState.targetColumn}
                data={data}
                correlations={correlations}
                onPrediction={(prediction, inputValues) => {
                  // Actualizar el estado para mostrar la predicción en el gráfico
                  onMlStateUpdate({ 
                    currentPrediction: {
                      prediction: prediction.prediction,
                      inputValues: inputValues,
                      targetColumn: mlState.targetColumn
                    }
                  })
                }}
              />
            </div>
          )}
        </div>

        {/* 2. Configuración de Visualización */}
        <div className="ml-section">
          <h3>2️⃣ Configuración de Visualización</h3>
          
          <div className="form-group">
            <label>Tipo de Gráfico:</label>
            <select 
              value={mlState.graphType} 
              onChange={(e) => onMlStateUpdate({ graphType: e.target.value })}
              className="form-select"
            >
              <option value="scatter">📊 Dispersión</option>
              <option value="line">📈 Línea</option>
              <option value="bar">📊 Barras</option>
              <option value="area">📉 Área</option>
              <option value="pie">🥧 Pastel</option>
              <option value="composed">🔀 Combinado</option>
              <option value="radar">🕸️ Radar</option>
              <option value="treemap">🗺️ Treemap</option>
            </select>
          </div>
        </div>

      </div>
    </div>
  )
}

export default MLControls

