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
  
  // Datos para visualización de features vs target (siempre disponible si hay target y features)
  const dataVisualizationChart = useMemo(() => {
    if (!data || data.length === 0) return []
    
    // Si hay target y features seleccionadas, mostrar relación automáticamente
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
          console.log(`📊 Gráfico de datos: Mostrando ${chartData.length} puntos válidos de ${data.length} totales`)
          return chartData
        }
      }
    }
    
    return []
  }, [data, targetColumn, JSON.stringify(selectedFeatures)])

  // Datos para visualización del modelo (solo si hay modelo entrenado)
  const modelChartData = useMemo(() => {
    if (!modelResults || !modelResults.predictions) {
      console.log('⚠️ modelChartData: No hay modelResults o predictions')
      return null
    }
    
    if (!modelResults.predictions.y_test || !modelResults.predictions.y_test_pred) {
      console.log('⚠️ modelChartData: Faltan y_test o y_test_pred', {
        hasYTest: !!modelResults.predictions.y_test,
        hasYTestPred: !!modelResults.predictions.y_test_pred
      })
      return null
    }
    
    try {
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
      
      console.log('✅ modelChartData generado:', predictions.length, 'puntos')
      return { predictions }
    } catch (error) {
      console.error('❌ Error generando modelChartData:', error)
      return null
    }
  }, [modelResults])

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

  // Verificar si hay datos para mostrar
  const hasDataVisualization = Array.isArray(dataVisualizationChart) && dataVisualizationChart.length > 0
  const hasModelVisualization = modelChartData && modelChartData.predictions && modelChartData.predictions.length > 0
  
  // Debug: Log para verificar qué se está mostrando
  useEffect(() => {
    console.log('🔍 GraphView Debug:', {
      hasDataVisualization,
      hasModelVisualization,
      modelResults: !!modelResults,
      hasPredictions: !!(modelResults?.predictions),
      predictionsLength: modelResults?.predictions?.y_test?.length || 0,
      modelChartData: !!modelChartData,
      learningCurveData: !!learningCurveData,
      learningCurveLength: learningCurveData?.length || 0
    })
  }, [hasDataVisualization, hasModelVisualization, modelResults, modelChartData, learningCurveData])
  
  if (!hasDataVisualization && !hasModelVisualization) {
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
    <div className="graph-container" style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: '30px', overflow: 'auto', padding: '20px' }}>
      {/* Gráfico 1: Visualización de Datos ANTES del entrenamiento (Features vs Target) */}
      {hasDataVisualization && !hasModelVisualization && (
        <div style={{ width: '100%', minHeight: '450px', maxHeight: '500px', border: '1px solid #E1E8ED', borderRadius: '8px', padding: '20px', background: 'white', marginBottom: '20px' }}>
          <h4 style={{ margin: '0 0 15px 0', fontSize: '14px', color: '#2C3E50' }}>
            📊 Visualización de Datos (Antes del Entrenamiento): {getXAxisLabel()} vs {getYAxisLabel()}
          </h4>
          <div style={{ height: '400px', width: '100%', marginTop: '10px' }}>
            {graphType === 'scatter' && (
              <>
                <ResponsiveContainer key={`data-${graphKey}`} width="100%" height="100%">
                  <ScatterChart data={dataVisualizationChart} margin={{ top: 20, right: 30, bottom: 20, left: 20 }}>
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
                {Array.isArray(dataVisualizationChart) && dataVisualizationChart.length > 0 && (
                  <div style={{ 
                    marginTop: '10px', 
                    padding: '8px', 
                    background: '#EBF5FB', 
                    borderRadius: '4px',
                    fontSize: '11px',
                    textAlign: 'center'
                  }}>
                    📊 <strong>Datos mostrados:</strong> {dataVisualizationChart.length} puntos | 
                    Rango {getXAxisLabel()}: [{Math.min(...dataVisualizationChart.map(d => d.x)).toFixed(2)}, {Math.max(...dataVisualizationChart.map(d => d.x)).toFixed(2)}] | 
                    Rango {getYAxisLabel()}: [{Math.min(...dataVisualizationChart.map(d => d.y)).toFixed(2)}, {Math.max(...dataVisualizationChart.map(d => d.y)).toFixed(2)}]
                  </div>
                )}
              </>
            )}
            
            {graphType === 'line' && (
              <ResponsiveContainer key={`data-line-${graphKey}`} width="100%" height="100%">
                <LineChart data={dataVisualizationChart}>
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
                  <Legend />
                  <Line type="monotone" dataKey="y" stroke="#3498DB" strokeWidth={2} name="Datos" />
                </LineChart>
              </ResponsiveContainer>
            )}
            
            {graphType === 'bar' && (
              <ResponsiveContainer key={`data-bar-${graphKey}`} width="100%" height="100%">
                <BarChart data={dataVisualizationChart.slice(0, 20)}>
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
                  <Legend />
                  <Bar dataKey="y" fill="#3498DB" name="Datos" />
                </BarChart>
              </ResponsiveContainer>
            )}

            {graphType === 'area' && (
              <ResponsiveContainer key={`data-area-${graphKey}`} width="100%" height="100%">
                <AreaChart data={dataVisualizationChart}>
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
                  <Legend />
                  <Area type="monotone" dataKey="y" stroke="#3498DB" fill="#3498DB" fillOpacity={0.6} name="Datos" />
                </AreaChart>
              </ResponsiveContainer>
            )}

            {graphType === 'pie' && (
              <ResponsiveContainer key={`data-pie-${graphKey}`} width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={dataVisualizationChart.slice(0, 10)}
                    dataKey="y"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    fill="#3498DB"
                    label
                  >
                    {dataVisualizationChart.slice(0, 10).map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={['#3498DB', '#2980B9', '#5DADE2', '#85C1E2', '#AED6F1'][index % 5]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            )}

            {graphType === 'composed' && (
              <ResponsiveContainer key={`data-composed-${graphKey}`} width="100%" height="100%">
                <ComposedChart data={dataVisualizationChart.slice(0, 20)}>
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
                  <Legend />
                  <Area type="monotone" dataKey="y" fill="#3498DB" fillOpacity={0.3} stroke="#3498DB" name="Datos" />
                  <Bar dataKey="y" fill="#2980B9" name="Datos" />
                  <Line type="monotone" dataKey="y" stroke="#5DADE2" strokeWidth={2} name="Datos" />
                </ComposedChart>
              </ResponsiveContainer>
            )}

            {graphType === 'radar' && (
              <ResponsiveContainer key={`data-radar-${graphKey}`} width="100%" height="100%">
                <RadarChart data={dataVisualizationChart.slice(0, 8)}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="name" />
                  <PolarRadiusAxis />
                  <Radar name={yAxis} dataKey="y" stroke="#3498DB" fill="#3498DB" fillOpacity={0.6} />
                  <Tooltip />
                  <Legend />
                </RadarChart>
              </ResponsiveContainer>
            )}

            {graphType === 'treemap' && (
              <ResponsiveContainer key={`data-treemap-${graphKey}`} width="100%" height="100%">
                <Treemap
                  data={dataVisualizationChart.slice(0, 20).map(item => ({ name: `${item.x}`, value: item.y }))}
                  dataKey="value"
                  stroke="#fff"
                  fill="#3498DB"
                >
                  <Tooltip />
                </Treemap>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      )}

      {/* Gráficos DESPUÉS del entrenamiento (si hay modelo entrenado) */}
      {modelResults && modelResults.predictions && modelResults.predictions.y_test && modelResults.predictions.y_test.length > 0 && (
        <>
          {/* Gráfico 2: Real vs Predicho (Resultados del Modelo) */}
          <div style={{ width: '100%', minHeight: '450px', maxHeight: '500px', border: '1px solid #E1E8ED', borderRadius: '8px', padding: '20px', background: 'white', marginBottom: '30px' }}>
            <h4 style={{ margin: '0 0 15px 0', fontSize: '14px', color: '#2C3E50' }}>📈 Resultados del Modelo: Real vs Predicho</h4>
            <div style={{ height: '400px', width: '100%', marginTop: '10px' }}>
              {modelChartData && modelChartData.predictions && modelChartData.predictions.length > 0 && graphType === 'scatter' && (
                  <>
                    <ResponsiveContainer key={`${graphKey}-model`} width="100%" height="100%">
                      <ScatterChart data={modelChartData.predictions}>
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
                
                {/* Si no hay modelChartData pero hay modelResults, mostrar mensaje */}
                {!modelChartData && (
                  <div style={{ padding: '20px', textAlign: 'center', color: '#666' }}>
                    <p>⚠️ No se pudieron generar los datos del gráfico.</p>
                    <p style={{ fontSize: '11px' }}>Verifique que el modelo tenga predicciones válidas.</p>
                  </div>
                )}
                
                {/* Si el tipo de gráfico no es scatter, mostrar el gráfico correspondiente */}
                {modelChartData && modelChartData.predictions && modelChartData.predictions.length > 0 && graphType !== 'scatter' && (
                  <>
                    {graphType === 'line' && (
                      <ResponsiveContainer key={`${graphKey}-model-line`} width="100%" height="100%">
                        <LineChart data={modelChartData.predictions}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis 
                            dataKey="x" 
                            label={{ value: 'Valores Reales', position: 'insideBottom', offset: -5 }}
                            domain={['dataMin', 'dataMax']}
                          />
                          <YAxis 
                            dataKey="y" 
                            label={{ value: 'Valores Predichos', angle: -90, position: 'insideLeft' }}
                            domain={['dataMin', 'dataMax']}
                          />
                          <Tooltip 
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
                          />
                          <Legend />
                          <Line 
                            type="monotone" 
                            dataKey="y" 
                            stroke="#3498DB" 
                            strokeWidth={2}
                            name="Predicciones"
                            dot={{ r: 3 }}
                          />
                          <Line 
                            type="linear" 
                            dataKey="x" 
                            stroke="#27AE60" 
                            strokeWidth={2}
                            strokeDasharray="5 5"
                            dot={false}
                            name="Predicción Perfecta (y=x)"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    )}
                    {graphType === 'bar' && (
                      <ResponsiveContainer key={`${graphKey}-model-bar`} width="100%" height="100%">
                        <BarChart data={modelChartData.predictions.slice(0, 20)}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis 
                            dataKey="x" 
                            label={{ value: 'Valores Reales', position: 'insideBottom', offset: -5 }}
                          />
                          <YAxis 
                            dataKey="y" 
                            label={{ value: 'Valores Predichos', angle: -90, position: 'insideLeft' }}
                          />
                          <Tooltip />
                          <Legend />
                          <Bar dataKey="y" fill="#3498DB" name="Predicciones" />
                        </BarChart>
                      </ResponsiveContainer>
                    )}
                    {graphType === 'area' && (
                      <ResponsiveContainer key={`${graphKey}-model-area`} width="100%" height="100%">
                        <AreaChart data={modelChartData.predictions}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis 
                            dataKey="x" 
                            label={{ value: 'Valores Reales', position: 'insideBottom', offset: -5 }}
                            domain={['dataMin', 'dataMax']}
                          />
                          <YAxis 
                            dataKey="y" 
                            label={{ value: 'Valores Predichos', angle: -90, position: 'insideLeft' }}
                            domain={['dataMin', 'dataMax']}
                          />
                          <Tooltip />
                          <Legend />
                          <Area 
                            type="monotone" 
                            dataKey="y" 
                            stroke="#3498DB" 
                            fill="#3498DB" 
                            fillOpacity={0.6}
                            name="Predicciones"
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    )}
                  </>
                )}
              </div>
            </div>
          )}

          {/* Gráfico 3: Learning Curves (Curvas de Aprendizaje) */}
          {modelResults && modelResults.learning_curve && learningCurveData && learningCurveData.length > 0 && !mlState?.showResiduals && (
            <div style={{ width: '100%', minHeight: '450px', maxHeight: '500px', border: '1px solid #E1E8ED', borderRadius: '8px', padding: '20px', background: 'white', marginBottom: '20px' }}>
              <h4 style={{ margin: '0 0 15px 0', fontSize: '14px', color: '#2C3E50' }}>📈 Learning Curves (Curvas de Aprendizaje)</h4>
              <div style={{ height: '400px', width: '100%', marginTop: '10px' }}>
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
              </div>
              <p style={{ fontSize: '11px', color: '#666', marginTop: '5px' }}>
                💡 Si las curvas se separan mucho, hay overfitting. Si ambas son altas, hay underfitting.
              </p>
            </div>
          )}
          
        </>
      )}
    </div>
  )
}

export default GraphView

