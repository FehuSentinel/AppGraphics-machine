import React, { useMemo, useEffect } from 'react'
import { 
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  LineChart, Line, BarChart, Bar, AreaChart, Area, PieChart, Pie, Cell,
  ComposedChart, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  Treemap, FunnelChart, Funnel, LabelList, Legend
} from 'recharts'
import './GraphView.css'

const GraphView = ({ data, columns, graphType, xAxis, yAxis, targetColumn, selectedFeatures, modelResults, mlState }) => {
  // Crear una key única para forzar re-render cuando cambian las selecciones
  const graphKey = useMemo(() => {
    const featuresStr = Array.isArray(selectedFeatures) ? selectedFeatures.join(',') : ''
    return `${targetColumn || ''}-${featuresStr}-${graphType || 'scatter'}-${modelResults ? 'model' : 'data'}-${data.length}`
  }, [targetColumn, selectedFeatures, graphType, modelResults, data.length])
  
  // Si hay modelo entrenado, mostrar resultados del modelo
  // Si no, mostrar relación entre features seleccionadas y target
  // IMPORTANTE: Usar EXACTAMENTE los mismos datos que se usan para entrenar
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return []
    
    // Si hay modelo entrenado, usar los datos del modelo
    if (modelResults && modelResults.predictions) {
      const predictions = modelResults.predictions.y_test.map((actual, idx) => {
        const predicted = modelResults.predictions.y_test_pred[idx]
        const error = Math.abs(actual - predicted)
        const errorPercent = actual !== 0 ? ((error / Math.abs(actual)) * 100).toFixed(2) : '0.00'
        return {
          x: actual,
          y: predicted,
          error: error,
          errorPercent: errorPercent,
          name: `Real: ${actual.toFixed(2)}, Predicho: ${predicted.toFixed(2)}, Error: ${error.toFixed(2)} (${errorPercent}%)`
        }
      })
      
      // Agregar puntos para la línea diagonal perfecta (y=x)
      const minVal = Math.min(...predictions.map(p => Math.min(p.x, p.y)))
      const maxVal = Math.max(...predictions.map(p => Math.max(p.x, p.y)))
      const diagonalLine = [
        { x: minVal, y: minVal, isDiagonal: true },
        { x: maxVal, y: maxVal, isDiagonal: true }
      ]
      
      return { predictions, diagonalLine, minVal, maxVal }
    }
    
    // Si hay target y features seleccionadas, mostrar relación automáticamente
    // Asegurar que selectedFeatures sea un array
    const features = Array.isArray(selectedFeatures) ? selectedFeatures : (selectedFeatures ? [selectedFeatures] : [])
    if (targetColumn && features.length > 0 && data.length > 0) {
      // Usar primera feature vs target (automático)
      const firstFeature = features[0]
      
      // Verificar que las columnas existan en los datos
      if (firstFeature && targetColumn && data[0] && data[0].hasOwnProperty(firstFeature) && data[0].hasOwnProperty(targetColumn)) {
        // IMPORTANTE: Filtrar y procesar los datos EXACTAMENTE como lo hace el backend
        const chartData = data
          .map(row => {
            const xVal = row[firstFeature]
            const yVal = row[targetColumn]
            
            // Convertir a número, manejar casos especiales (igual que el backend)
            let x = typeof xVal === 'number' ? xVal : parseFloat(xVal)
            let y = typeof yVal === 'number' ? yVal : parseFloat(yVal)
            
            // Si el parseo falla, usar NaN para que el filtro lo elimine
            if (isNaN(x)) x = NaN
            if (isNaN(y)) y = NaN
            
            return {
              x: x,
              y: y,
              name: `${firstFeature}: ${xVal}, ${targetColumn}: ${yVal}`
            }
          })
          .filter(item => !isNaN(item.x) && !isNaN(item.y) && item.x !== null && item.y !== null)
        
        // Solo retornar si hay datos válidos
        if (chartData.length > 0) {
          console.log(`📊 Gráfico: Mostrando ${chartData.length} puntos válidos de ${data.length} totales`)
          console.log(`📊 Rango X: [${Math.min(...chartData.map(d => d.x)).toFixed(2)}, ${Math.max(...chartData.map(d => d.x)).toFixed(2)}]`)
          console.log(`📊 Rango Y: [${Math.min(...chartData.map(d => d.y)).toFixed(2)}, ${Math.max(...chartData.map(d => d.y)).toFixed(2)}]`)
          return chartData
        }
      }
    }
    
    return []
    // Usar JSON.stringify para arrays para detectar cambios en el contenido
  }, [data, targetColumn, JSON.stringify(selectedFeatures), modelResults, graphType])

  const modelData = useMemo(() => {
    if (!modelResults || !modelResults.predictions) return []
    
    return modelResults.predictions.y_test.map((actual, idx) => ({
      actual,
      predicted: modelResults.predictions.y_test_pred[idx],
      index: idx
    }))
  }, [modelResults])

  // Determinar labels de ejes automáticamente
  const getXAxisLabel = () => {
    if (modelResults && modelResults.predictions) return 'Valores Reales'
    const features = selectedFeatures || []
    if (features.length > 0) return features[0]
    return 'Característica'
  }

  const getYAxisLabel = () => {
    if (modelResults && modelResults.predictions) return 'Valores Predichos'
    if (targetColumn) return targetColumn
    return 'Variable Objetivo'
  }

  // Datos de learning curve si están disponibles
  const learningCurveData = useMemo(() => {
    if (!modelResults || !modelResults.learning_curve || modelResults.learning_curve.length === 0) {
      return null
    }
    try {
      const data = modelResults.learning_curve
        .map(point => {
          const trainError = parseFloat(point.train_error) || 0
          const testError = parseFloat(point.test_error) || 0
          return {
            train_size: parseInt(point.train_size) || 0,
            train_error: isFinite(trainError) ? trainError : 0,
            test_error: isFinite(testError) ? testError : 0
          }
        })
        .filter(point => point.train_size > 0) // Filtrar puntos inválidos
      
      return data.length > 0 ? data : null
    } catch (error) {
      console.error('Error procesando learning curve data:', error)
      return null
    }
  }, [modelResults])

  // Datos de residuos para gráfico de validación (debe estar antes de cualquier return condicional)
  const residualsData = useMemo(() => {
    if (!modelResults || !modelResults.predictions || !modelResults.predictions.residuals) return []
    
    return modelResults.predictions.y_test.map((actual, idx) => ({
      predicted: modelResults.predictions.y_test_pred[idx],
      residual: modelResults.predictions.residuals[idx],
      index: idx
    }))
  }, [modelResults])

  // Datos de predicción actual para mostrar en el gráfico
  const currentPredictionData = useMemo(() => {
    if (!mlState?.currentPrediction || !mlState.currentPrediction.inputValues) return null
    
    const features = Array.isArray(selectedFeatures) ? selectedFeatures : (selectedFeatures ? [selectedFeatures] : [])
    if (features.length === 0) return null
    
    const firstFeature = features[0]
    const inputValue = mlState.currentPrediction.inputValues[firstFeature]
    const predictionValue = mlState.currentPrediction.prediction
    
    if (inputValue === undefined || predictionValue === undefined) return null
    
    return {
      x: parseFloat(inputValue),
      y: parseFloat(predictionValue),
      name: `Nueva Predicción: ${firstFeature}=${inputValue}, ${mlState.currentPrediction.targetColumn}=${predictionValue.toFixed(2)}`,
      isPrediction: true
    }
  }, [mlState?.currentPrediction, selectedFeatures])

  // Verificar si chartData es válido (array o objeto con predictions)
  const hasValidData = Array.isArray(chartData) ? chartData.length > 0 : (chartData && chartData.predictions && chartData.predictions.length > 0)
  
  if (!hasValidData) {
    const features = selectedFeatures || []
    const hasTarget = !!targetColumn
    const hasFeatures = features.length > 0
    
    let message = 'Seleccione variable objetivo y características para visualizar'
    if (!hasTarget && !hasFeatures) {
      message = 'Seleccione variable objetivo (Y) y al menos una característica (X) para visualizar'
    } else if (!hasTarget) {
      message = 'Seleccione una variable objetivo (Y) para visualizar'
    } else if (!hasFeatures) {
      message = 'Seleccione al menos una característica (X) para visualizar'
    } else if (data.length === 0) {
      message = 'No hay datos para visualizar'
    } else {
      message = 'No se pudieron procesar los datos. Verifique que las columnas seleccionadas contengan valores numéricos válidos.'
    }
    
    return (
      <div className="graph-placeholder">
        <p>{message}</p>
      </div>
    )
  }

  return (
    <div className="graph-container">
          {/* Learning Curve si está disponible y no se muestran residuos */}
          {learningCurveData && !mlState?.showResiduals && (
            <div style={{ width: '100%', height: '100%', marginBottom: '20px' }}>
              <h4>📈 Learning Curves (Curvas de Aprendizaje)</h4>
              <ResponsiveContainer key={`learning-curve-${graphKey}`} width="100%" height="80%">
                <LineChart 
                  data={learningCurveData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="train_size" 
                    name="Tamaño de Entrenamiento"
                    label={{ value: 'Tamaño de Entrenamiento', position: 'insideBottom', offset: -5 }}
                    type="number"
                    scale="linear"
                    allowDecimals={false}
                  />
                  <YAxis 
                    label={{ value: 'RMSE', angle: -90, position: 'insideLeft' }}
                    type="number"
                    scale="linear"
                    domain={['auto', 'auto']}
                    allowDataOverflow={false}
                  />
                  <Tooltip 
                    formatter={(value, name) => {
                      if (typeof value === 'number' && isFinite(value)) {
                        // Mostrar más decimales si el valor es muy pequeño
                        if (value < 0.01) {
                          return [value.toFixed(6), name]
                        }
                        return [value.toFixed(4), name]
                      }
                      return [value, name]
                    }}
                    labelFormatter={(label) => `Tamaño: ${label}`}
                  />
                  <Legend 
                    verticalAlign="top" 
                    height={36}
                    wrapperStyle={{ paddingTop: '10px' }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="train_error" 
                    stroke="#3498DB" 
                    strokeWidth={2}
                    name="Error Train"
                    dot={{ r: 4, fill: '#3498DB' }}
                    activeDot={{ r: 6 }}
                    connectNulls={false}
                    isAnimationActive={true}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="test_error" 
                    stroke="#E74C3C" 
                    strokeWidth={2}
                    name="Error Test"
                    dot={{ r: 4, fill: '#E74C3C' }}
                    activeDot={{ r: 6 }}
                    connectNulls={false}
                    isAnimationActive={true}
                  />
                </LineChart>
              </ResponsiveContainer>
              <p style={{ fontSize: '11px', color: '#666', marginTop: '5px' }}>
                💡 Si las curvas se separan mucho, hay overfitting. Si ambas son altas, hay underfitting.
              </p>
            </div>
          )}
          
          {modelResults && modelResults.predictions ? (
            <>
              {mlState?.showResiduals && residualsData.length > 0 ? (
            <div style={{ width: '100%', height: '100%' }}>
              <h4>Gráfico de Residuos (Validación de Supuestos)</h4>
              <ResponsiveContainer key={graphKey} width="100%" height="100%">
                <ScatterChart data={residualsData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="predicted" 
                    name="Valores Predichos" 
                    label={{ value: 'Valores Predichos', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis 
                    dataKey="residual" 
                    name="Residuos"
                    label={{ value: 'Residuos (Real - Predicho)', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip 
                    cursor={{ strokeDasharray: '3 3' }}
                    formatter={(value, name) => {
                      if (name === 'residual') {
                        return [value.toFixed(4), 'Residuo']
                      }
                      return [value.toFixed(2), name]
                    }}
                  />
                  <Scatter dataKey="residual" fill="#3498DB" />
                  <Line 
                    type="linear" 
                    dataKey={() => 0} 
                    stroke="#E74C3C" 
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={false}
                  />
                </ScatterChart>
              </ResponsiveContainer>
              <p style={{ fontSize: '11px', color: '#666', marginTop: '5px' }}>
                Los residuos deben estar distribuidos aleatoriamente alrededor de 0. Si hay patrones, el modelo no cumple los supuestos.
              </p>
            </div>
          ) : (
            <div style={{ width: '100%', height: '100%' }}>
              <h4>Real vs Predicho (Resultados del Modelo)</h4>
              {chartData && chartData.predictions && (
                <>
                  <ResponsiveContainer key={`${graphKey}-model`} width="100%" height="100%">
                    <ScatterChart data={chartData.predictions}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        dataKey="x" 
                        name="Real" 
                        label={{ value: 'Valores Reales', position: 'insideBottom', offset: -5 }}
                        domain={['dataMin', 'dataMax']}
                      />
                      <YAxis 
                        dataKey="y" 
                        name="Predicho"
                        label={{ value: 'Valores Predichos', angle: -90, position: 'insideLeft' }}
                        domain={['dataMin', 'dataMax']}
                      />
                      <Tooltip 
                        cursor={{ strokeDasharray: '3 3' }}
                        formatter={(value, name, props) => {
                          if (name === 'y') {
                            return [
                              `Predicho: ${value.toFixed(2)}`,
                              `Real: ${props.payload.x.toFixed(2)}`,
                              `Error: ${props.payload.error.toFixed(2)} (${props.payload.errorPercent}%)`
                            ]
                          }
                          return [value.toFixed(2), name]
                        }}
                        labelFormatter={() => ''}
                      />
                      <Legend 
                        verticalAlign="top" 
                        height={36}
                        wrapperStyle={{ paddingTop: '10px' }}
                      />
                      <Scatter 
                        name="Predicciones" 
                        dataKey="y" 
                        fill="#3498DB" 
                        fillOpacity={0.6}
                      />
                      {/* Línea diagonal perfecta (y=x) */}
                      <Line 
                        type="linear" 
                        dataKey="x" 
                        stroke="#27AE60" 
                        strokeWidth={2}
                        strokeDasharray="5 5"
                        dot={false}
                        name="Predicción Perfecta (y=x)"
                        connectNulls={false}
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                  {modelResults.metrics && (
                    <div style={{ 
                      marginTop: '10px', 
                      padding: '10px', 
                      background: '#F8F9FA', 
                      borderRadius: '4px',
                      fontSize: '11px'
                    }}>
                      <strong>📊 Métricas:</strong> R² Test: {modelResults.metrics.test_r2?.toFixed(4) || 'N/A'} | 
                      RMSE: {modelResults.metrics.test_rmse?.toFixed(2) || 'N/A'} | 
                      MAE: {modelResults.metrics.test_mae?.toFixed(2) || 'N/A'}
                    </div>
                  )}
                  <p style={{ fontSize: '11px', color: '#666', marginTop: '5px' }}>
                    💡 Los puntos cerca de la línea verde (y=x) indican predicciones precisas. 
                    Si hay muchos puntos alejados, el modelo necesita mejoras.
                  </p>
                </>
              )}
            </div>
          )}
        </>
      ) : (
        <>
          {graphType === 'scatter' && (
            <>
              <div style={{ marginBottom: '10px', fontSize: '12px', color: '#666' }}>
                <strong>📊 Visualización:</strong> {getXAxisLabel()} vs {getYAxisLabel()}
                {Array.isArray(selectedFeatures) && selectedFeatures.length > 1 && (
                  <span style={{ marginLeft: '10px', fontSize: '11px' }}>
                    (Mostrando {selectedFeatures[0]}, {selectedFeatures.length - 1} más seleccionadas)
                  </span>
                )}
              </div>
              <ResponsiveContainer key={graphKey} width="100%" height="100%">
                <ScatterChart data={chartData} margin={{ top: 20, right: 30, bottom: 20, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="x" 
                    name={getXAxisLabel()}
                    label={{ value: getXAxisLabel(), position: 'insideBottom', offset: -5 }}
                    domain={['dataMin', 'dataMax']}
                  />
                  <YAxis 
                    dataKey="y" 
                    name={getYAxisLabel()}
                    label={{ value: getYAxisLabel(), angle: -90, position: 'insideLeft' }}
                    domain={['dataMin', 'dataMax']}
                  />
                  <Tooltip 
                    cursor={{ strokeDasharray: '3 3' }}
                    formatter={(value, name, props) => {
                      if (props.payload && props.payload.name) {
                        return [props.payload.name, '']
                      }
                      return [value, name]
                    }}
                  />
                  <Legend 
                    verticalAlign="top" 
                    height={36}
                    wrapperStyle={{ paddingTop: '10px' }}
                  />
                  <Scatter 
                    name="Datos" 
                    dataKey="y" 
                    fill="#3498DB" 
                    fillOpacity={0.6}
                  />
                  {/* Mostrar predicción actual si existe */}
                  {currentPredictionData && (
                    <Scatter 
                      name="Nueva Predicción"
                      data={[currentPredictionData]} 
                      dataKey="y" 
                      fill="#E74C3C" 
                      shape={(props) => {
                        const { cx, cy } = props
                        return (
                          <g>
                            <circle cx={cx} cy={cy} r={8} fill="#E74C3C" stroke="#C0392B" strokeWidth={2} />
                            <circle cx={cx} cy={cy} r={12} fill="none" stroke="#E74C3C" strokeWidth={2} opacity={0.5} />
                          </g>
                        )
                      }}
                    />
                  )}
                </ScatterChart>
              </ResponsiveContainer>
              {currentPredictionData && (
                <div style={{ 
                  marginTop: '10px', 
                  padding: '8px', 
                  background: '#FFEBEE', 
                  borderRadius: '4px',
                  fontSize: '11px',
                  textAlign: 'center'
                }}>
                  🔴 <strong>Nueva Predicción:</strong> {getXAxisLabel()}={currentPredictionData.x.toFixed(2)}, 
                  {getYAxisLabel()}={currentPredictionData.y.toFixed(2)}
                </div>
              )}
              {Array.isArray(chartData) && chartData.length > 0 && (
                <div style={{ 
                  marginTop: '10px', 
                  padding: '8px', 
                  background: '#EBF5FB', 
                  borderRadius: '4px',
                  fontSize: '11px',
                  textAlign: 'center'
                }}>
                  📊 <strong>Datos mostrados:</strong> {chartData.length} puntos | 
                  Rango {getXAxisLabel()}: [{Math.min(...chartData.map(d => d.x)).toFixed(2)}, {Math.max(...chartData.map(d => d.x)).toFixed(2)}] | 
                  Rango {getYAxisLabel()}: [{Math.min(...chartData.map(d => d.y)).toFixed(2)}, {Math.max(...chartData.map(d => d.y)).toFixed(2)}]
                </div>
              )}
            </>
          )}
          
          {graphType === 'line' && (
            <ResponsiveContainer key={graphKey} width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="x" 
                  name={getXAxisLabel()}
                  label={{ value: getXAxisLabel(), position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  dataKey="y" 
                  name={getYAxisLabel()}
                  label={{ value: getYAxisLabel(), angle: -90, position: 'insideLeft' }}
                />
                <Tooltip />
                <Line type="monotone" dataKey="y" stroke="#3498DB" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          )}
          
          {graphType === 'bar' && (
            <ResponsiveContainer key={graphKey} width="100%" height="100%">
              <BarChart data={chartData.slice(0, 20)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="x" 
                  name={getXAxisLabel()}
                  label={{ value: getXAxisLabel(), position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  dataKey="y" 
                  name={getYAxisLabel()}
                  label={{ value: getYAxisLabel(), angle: -90, position: 'insideLeft' }}
                />
                <Tooltip />
                <Bar dataKey="y" fill="#3498DB" />
              </BarChart>
            </ResponsiveContainer>
          )}

          {graphType === 'area' && (
            <ResponsiveContainer key={graphKey} width="100%" height="100%">
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="x" 
                  name={getXAxisLabel()}
                  label={{ value: getXAxisLabel(), position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  dataKey="y" 
                  name={getYAxisLabel()}
                  label={{ value: getYAxisLabel(), angle: -90, position: 'insideLeft' }}
                />
                <Tooltip />
                <Area type="monotone" dataKey="y" stroke="#3498DB" fill="#3498DB" fillOpacity={0.6} />
              </AreaChart>
            </ResponsiveContainer>
          )}

          {graphType === 'pie' && (
            <ResponsiveContainer key={graphKey} width="100%" height="100%">
              <PieChart>
                <Pie
                  data={chartData.slice(0, 10)}
                  dataKey="y"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  fill="#3498DB"
                  label
                >
                  {chartData.slice(0, 10).map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={['#3498DB', '#2980B9', '#5DADE2', '#85C1E2', '#AED6F1'][index % 5]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          )}

          {graphType === 'composed' && (
            <ResponsiveContainer key={graphKey} width="100%" height="100%">
              <ComposedChart data={chartData.slice(0, 20)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="x" 
                  name={xAxis}
                  label={{ value: xAxis, position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  label={{ value: yAxis, angle: -90, position: 'insideLeft' }}
                />
                <Tooltip />
                <Area type="monotone" dataKey="y" fill="#3498DB" fillOpacity={0.3} stroke="#3498DB" />
                <Bar dataKey="y" fill="#2980B9" />
                <Line type="monotone" dataKey="y" stroke="#5DADE2" strokeWidth={2} />
              </ComposedChart>
            </ResponsiveContainer>
          )}

          {graphType === 'radar' && (
            <ResponsiveContainer key={graphKey} width="100%" height="100%">
              <RadarChart data={chartData.slice(0, 8)}>
                <PolarGrid />
                <PolarAngleAxis dataKey="name" />
                <PolarRadiusAxis />
                <Radar name={yAxis} dataKey="y" stroke="#3498DB" fill="#3498DB" fillOpacity={0.6} />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          )}

          {graphType === 'treemap' && (
            <ResponsiveContainer key={graphKey} width="100%" height="100%">
              <Treemap
                data={chartData.slice(0, 20).map(item => ({ name: `${item.x}`, value: item.y }))}
                dataKey="value"
                stroke="#fff"
                fill="#3498DB"
              >
                <Tooltip />
              </Treemap>
            </ResponsiveContainer>
          )}
        </>
      )}
    </div>
  )
}

export default GraphView

