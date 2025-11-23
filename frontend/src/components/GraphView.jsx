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
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return []
    
    // Si hay modelo entrenado, usar los datos del modelo
    if (modelResults && modelResults.predictions) {
      return modelResults.predictions.y_test.map((actual, idx) => ({
        x: actual,
        y: modelResults.predictions.y_test_pred[idx],
        name: `Real: ${actual.toFixed(2)}, Predicho: ${modelResults.predictions.y_test_pred[idx].toFixed(2)}`
      }))
    }
    
    // Si hay target y features seleccionadas, mostrar relación automáticamente
    // Asegurar que selectedFeatures sea un array
    const features = Array.isArray(selectedFeatures) ? selectedFeatures : (selectedFeatures ? [selectedFeatures] : [])
    if (targetColumn && features.length > 0 && data.length > 0) {
      // Usar primera feature vs target (automático)
      const firstFeature = features[0]
      
      // Verificar que las columnas existan en los datos
      if (firstFeature && targetColumn && data[0].hasOwnProperty(firstFeature) && data[0].hasOwnProperty(targetColumn)) {
        const chartData = data.map(row => {
          const xVal = row[firstFeature]
          const yVal = row[targetColumn]
          
          // Convertir a número, manejar casos especiales
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
        }).filter(item => !isNaN(item.x) && !isNaN(item.y) && item.x !== null && item.y !== null)
        
        // Solo retornar si hay datos válidos
        if (chartData.length > 0) {
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
    return modelResults.learning_curve.map(point => ({
      train_size: point.train_size,
      train_error: point.train_error,
      test_error: point.test_error
    }))
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

  if (chartData.length === 0) {
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
                <LineChart data={learningCurveData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="train_size" 
                    name="Tamaño de Entrenamiento"
                    label={{ value: 'Tamaño de Entrenamiento', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis 
                    label={{ value: 'RMSE', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip />
                  <Line 
                    type="monotone" 
                    dataKey="train_error" 
                    stroke="#3498DB" 
                    strokeWidth={2}
                    name="Error Train"
                    dot={{ r: 4 }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="test_error" 
                    stroke="#E74C3C" 
                    strokeWidth={2}
                    name="Error Test"
                    dot={{ r: 4 }}
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
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} />
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
              <ResponsiveContainer key={`${graphKey}-model`} width="100%" height="100%">
                <ScatterChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="x" 
                    name="Real" 
                    label={{ value: 'Valores Reales', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis 
                    dataKey="y" 
                    name="Predicho"
                    label={{ value: 'Valores Predichos', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                  <Scatter dataKey="y" fill="#3498DB" />
                  <Line 
                    type="monotone" 
                    dataKey="x" 
                    stroke="#2C3E50" 
                    strokeWidth={2}
                    dot={false}
                  />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          )}
        </>
      ) : (
        <>
          {graphType === 'scatter' && (
            <ResponsiveContainer key={graphKey} width="100%" height="100%">
              <ScatterChart data={chartData}>
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
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Scatter dataKey="y" fill="#3498DB" />
              </ScatterChart>
            </ResponsiveContainer>
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

