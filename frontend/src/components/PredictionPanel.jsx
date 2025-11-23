import React, { useState, useEffect, useMemo } from 'react'
import { predictWithModel, predictBatchWithModel } from '../services/api'

// Componente para panel de predicción mejorado
const PredictionPanel = ({ modelId, features, targetColumn, data, correlations, onPrediction }) => {
  const [inputValues, setInputValues] = useState({})
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [statistics, setStatistics] = useState({})
  const [exampleRows, setExampleRows] = useState([])
  const [showMassPrediction, setShowMassPrediction] = useState(false)
  const [massPredictionFile, setMassPredictionFile] = useState(null)
  const [massPredictionResults, setMassPredictionResults] = useState(null)
  const [selectedExample, setSelectedExample] = useState(null)

  // Calcular estadísticas y ejemplos
  useEffect(() => {
    if (data && data.length > 0 && features.length > 0) {
      const stats = {}
      const examples = []
      
      features.forEach(feature => {
        const values = data
          .map(row => parseFloat(row[feature]))
          .filter(val => !isNaN(val))
        
        if (values.length > 0) {
          const sorted = [...values].sort((a, b) => a - b)
          stats[feature] = {
            mean: values.reduce((a, b) => a + b, 0) / values.length,
            min: Math.min(...values),
            max: Math.max(...values),
            median: sorted[Math.floor(sorted.length / 2)],
            q25: sorted[Math.floor(sorted.length * 0.25)],
            q75: sorted[Math.floor(sorted.length * 0.75)]
          }
        }
      })

      // Obtener 3-5 ejemplos reales del dataset
      const sampleSize = Math.min(5, data.length)
      const step = Math.floor(data.length / sampleSize)
      for (let i = 0; i < sampleSize; i++) {
        const idx = i * step
        if (idx < data.length) {
          const row = data[idx]
          // Verificar que tenga todos los features necesarios
          const hasAllFeatures = features.every(f => row.hasOwnProperty(f) && !isNaN(parseFloat(row[f])))
          if (hasAllFeatures) {
            examples.push(row)
          }
        }
      }
      
      setStatistics(stats)
      setExampleRows(examples)
      
      // Inicializar con media
      const initialValues = {}
      features.forEach(feature => {
        if (stats[feature]) {
          initialValues[feature] = stats[feature].mean.toFixed(2)
        } else {
          initialValues[feature] = 0
        }
      })
      setInputValues(initialValues)
    }
  }, [features, data])

  // Validar si un valor está en rango
  const isValueInRange = (feature, value) => {
    if (!statistics[feature]) return true
    const numValue = parseFloat(value)
    if (isNaN(numValue)) return false
    return numValue >= statistics[feature].min && numValue <= statistics[feature].max
  }

  // Obtener sugerencia basada en correlaciones
  const getSuggestion = (feature, currentValue) => {
    if (!correlations || !statistics[feature]) return null
    
    const currentNum = parseFloat(currentValue)
    if (isNaN(currentNum)) return null

    // Buscar features correlacionadas
    const correlatedFeatures = Object.entries(correlations)
      .filter(([f, corr]) => f !== feature && Math.abs(corr) > 0.5)
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      .slice(0, 2)

    if (correlatedFeatures.length === 0) return null

    const suggestions = {}
    correlatedFeatures.forEach(([otherFeature, corr]) => {
      if (statistics[otherFeature]) {
        // Si están positivamente correlacionadas, sugerir valor proporcional
        const ratio = currentNum / statistics[feature].mean
        const suggested = statistics[otherFeature].mean * ratio
        suggestions[otherFeature] = {
          value: Math.max(statistics[otherFeature].min, Math.min(statistics[otherFeature].max, suggested)),
          correlation: corr
        }
      }
    })

    return Object.keys(suggestions).length > 0 ? suggestions : null
  }

  const handlePredict = async () => {
    // Validar que todos los valores estén presentes
    const missing = features.filter(f => !inputValues[f] || inputValues[f] === '')
    if (missing.length > 0) {
      alert(`Por favor complete los valores para: ${missing.join(', ')}`)
      return
    }

    // Convertir a números
    const numericValues = {}
    features.forEach(f => {
      const val = parseFloat(inputValues[f])
      if (isNaN(val)) {
        alert(`El valor de ${f} no es un número válido`)
        return
      }
      numericValues[f] = val
    })

    if (Object.keys(numericValues).length !== features.length) return

    setLoading(true)
    try {
      const result = await predictWithModel(modelId, numericValues)
      setPrediction(result)
      if (onPrediction) {
        onPrediction(result, numericValues)
      }
    } catch (error) {
      alert('Error al hacer predicción: ' + (error.response?.data?.detail || error.message))
      setPrediction(null)
    } finally {
      setLoading(false)
    }
  }

  const handleInputChange = (feature, value) => {
    setInputValues(prev => ({
      ...prev,
      [feature]: value
    }))
  }

  const useExample = (example) => {
    const values = {}
    features.forEach(feature => {
      const val = example[feature]
      values[feature] = typeof val === 'number' ? val.toFixed(2) : parseFloat(val).toFixed(2)
    })
    setInputValues(values)
    setSelectedExample(example)
  }

  const setToStat = (feature, statType) => {
    if (statistics[feature]) {
      let value
      switch(statType) {
        case 'mean': value = statistics[feature].mean; break
        case 'median': value = statistics[feature].median; break
        case 'min': value = statistics[feature].min; break
        case 'max': value = statistics[feature].max; break
        default: return
      }
      handleInputChange(feature, value.toFixed(2))
    }
  }

  const applySuggestions = () => {
    const suggestions = {}
    features.forEach(feature => {
      const sugg = getSuggestion(feature, inputValues[feature])
      if (sugg) {
        Object.assign(suggestions, sugg)
      }
    })
    
    if (Object.keys(suggestions).length > 0) {
      const newValues = { ...inputValues }
      Object.entries(suggestions).forEach(([feature, { value }]) => {
        if (features.includes(feature)) {
          newValues[feature] = value.toFixed(2)
        }
      })
      setInputValues(newValues)
      alert(`✅ Se aplicaron sugerencias basadas en correlaciones para ${Object.keys(suggestions).length} características`)
    } else {
      alert('No hay sugerencias disponibles basadas en correlaciones')
    }
  }

  return (
    <div>
      {/* Tabs: Predicción Individual vs Masiva */}
      <div style={{ 
        display: 'flex', 
        gap: '5px', 
        marginBottom: '15px',
        borderBottom: '2px solid #E1E8ED'
      }}>
        <button
          onClick={() => setShowMassPrediction(false)}
          style={{
            padding: '8px 15px',
            background: !showMassPrediction ? '#27AE60' : 'transparent',
            color: !showMassPrediction ? 'white' : '#666',
            border: 'none',
            borderBottom: !showMassPrediction ? '3px solid #229954' : '3px solid transparent',
            cursor: 'pointer',
            fontWeight: !showMassPrediction ? '600' : '400',
            fontSize: '12px',
            borderRadius: '4px 4px 0 0'
          }}
        >
          🔮 Predicción Individual
        </button>
        <button
          onClick={() => setShowMassPrediction(true)}
          style={{
            padding: '8px 15px',
            background: showMassPrediction ? '#27AE60' : 'transparent',
            color: showMassPrediction ? 'white' : '#666',
            border: 'none',
            borderBottom: showMassPrediction ? '3px solid #229954' : '3px solid transparent',
            cursor: 'pointer',
            fontWeight: showMassPrediction ? '600' : '400',
            fontSize: '12px',
            borderRadius: '4px 4px 0 0'
          }}
        >
          📊 Predicción Masiva (CSV/Excel)
        </button>
      </div>

      {!showMassPrediction ? (
        <>
          {/* Plantillas con Ejemplos Reales */}
          {exampleRows.length > 0 && (
            <div style={{
              marginBottom: '15px',
              padding: '10px',
              background: '#EBF5FB',
              borderRadius: '6px',
              border: '1px solid #D6EAF8'
            }}>
              <p style={{ margin: '0 0 8px 0', fontSize: '11px', fontWeight: '600' }}>
                📋 Ejemplos Reales del Dataset:
              </p>
              <div style={{ display: 'flex', gap: '5px', flexWrap: 'wrap' }}>
                {exampleRows.slice(0, 5).map((example, idx) => (
                  <button
                    key={idx}
                    onClick={() => useExample(example)}
                    style={{
                      padding: '5px 10px',
                      background: selectedExample === example ? '#3498DB' : '#D6EAF8',
                      color: selectedExample === example ? 'white' : '#2C3E50',
                      border: '1px solid #85C1E2',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontSize: '10px',
                      whiteSpace: 'nowrap'
                    }}
                  >
                    Ejemplo {idx + 1}
                  </button>
                ))}
              </div>
              <small style={{ fontSize: '9px', color: '#666', display: 'block', marginTop: '5px' }}>
                💡 Haz clic en un ejemplo para copiar sus valores
              </small>
            </div>
          )}

          {/* Botón de Sugerencias Inteligentes */}
          {correlations && (
            <div style={{ marginBottom: '15px' }}>
              <button
                onClick={applySuggestions}
                style={{
                  width: '100%',
                  padding: '8px',
                  background: 'linear-gradient(135deg, #9B59B6 0%, #8E44AD 100%)',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '11px',
                  fontWeight: '600'
                }}
              >
                🧠 Aplicar Sugerencias Inteligentes (Basadas en Correlaciones)
              </button>
            </div>
          )}

          {/* Inputs de Valores */}
          <div style={{ marginBottom: '15px' }}>
            {features.map(feature => {
              const value = inputValues[feature] || ''
              const numValue = parseFloat(value)
              const inRange = isValueInRange(feature, value)
              const suggestions = getSuggestion(feature, value)
              
              return (
                <div key={feature} style={{ marginBottom: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '5px' }}>
                    <label style={{ fontSize: '12px', fontWeight: '600', display: 'block' }}>
                      {feature}:
                    </label>
                    {statistics[feature] && (
                      <div style={{ fontSize: '10px', display: 'flex', gap: '3px' }}>
                        <button
                          onClick={() => setToStat(feature, 'min')}
                          style={{ padding: '2px 5px', fontSize: '9px', background: '#E74C3C', color: 'white', border: 'none', borderRadius: '3px', cursor: 'pointer' }}
                          title={`Mínimo: ${statistics[feature].min.toFixed(2)}`}
                        >
                          Min
                        </button>
                        <button
                          onClick={() => setToStat(feature, 'mean')}
                          style={{ padding: '2px 5px', fontSize: '9px', background: '#3498DB', color: 'white', border: 'none', borderRadius: '3px', cursor: 'pointer' }}
                          title={`Media: ${statistics[feature].mean.toFixed(2)}`}
                        >
                          Media
                        </button>
                        <button
                          onClick={() => setToStat(feature, 'median')}
                          style={{ padding: '2px 5px', fontSize: '9px', background: '#9B59B6', color: 'white', border: 'none', borderRadius: '3px', cursor: 'pointer' }}
                          title={`Mediana: ${statistics[feature].median.toFixed(2)}`}
                        >
                          Med
                        </button>
                        <button
                          onClick={() => setToStat(feature, 'max')}
                          style={{ padding: '2px 5px', fontSize: '9px', background: '#E67E22', color: 'white', border: 'none', borderRadius: '3px', cursor: 'pointer' }}
                          title={`Máximo: ${statistics[feature].max.toFixed(2)}`}
                        >
                          Max
                        </button>
                      </div>
                    )}
                  </div>
                  <input
                    type="number"
                    step="any"
                    value={value}
                    onChange={(e) => handleInputChange(feature, e.target.value)}
                    placeholder={statistics[feature] ? `Rango: ${statistics[feature].min.toFixed(2)} - ${statistics[feature].max.toFixed(2)}` : 'Ingrese valor'}
                    style={{
                      width: '100%',
                      padding: '8px',
                      border: `2px solid ${!value || inRange ? '#BDC3C7' : '#E74C3C'}`,
                      borderRadius: '6px',
                      fontSize: '12px',
                      transition: 'border-color 0.3s',
                      background: !value || inRange ? 'white' : '#FFEBEE'
                    }}
                    onFocus={(e) => e.target.style.borderColor = '#27AE60'}
                    onBlur={(e) => e.target.style.borderColor = !value || inRange ? '#BDC3C7' : '#E74C3C'}
                  />
                  {!inRange && value && (
                    <small style={{ fontSize: '9px', color: '#E74C3C', display: 'block', marginTop: '3px' }}>
                      ⚠️ Valor fuera del rango histórico
                    </small>
                  )}
                  {statistics[feature] && (
                    <small style={{ fontSize: '10px', color: '#666', display: 'block', marginTop: '3px' }}>
                      📊 Media: {statistics[feature].mean.toFixed(2)} | 
                      Mediana: {statistics[feature].median.toFixed(2)} | 
                      Rango: [{statistics[feature].min.toFixed(2)}, {statistics[feature].max.toFixed(2)}]
                    </small>
                  )}
                  {suggestions && Object.keys(suggestions).length > 0 && (
                    <div style={{ marginTop: '5px', padding: '5px', background: '#F8F9FA', borderRadius: '4px', fontSize: '9px' }}>
                      💡 Sugerencias basadas en correlaciones:
                      {Object.entries(suggestions).map(([otherFeature, { value, correlation }]) => (
                        <div key={otherFeature} style={{ marginTop: '2px' }}>
                          {otherFeature}: {value.toFixed(2)} (corr: {correlation.toFixed(2)})
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )
            })}
          </div>

          <button
            onClick={handlePredict}
            disabled={loading}
            style={{
              width: '100%',
              padding: '12px',
              background: loading ? '#95A5A6' : 'linear-gradient(135deg, #27AE60 0%, #229954 100%)',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: loading ? 'not-allowed' : 'pointer',
              fontWeight: '600',
              fontSize: '13px',
              boxShadow: loading ? 'none' : '0 2px 4px rgba(0,0,0,0.2)',
              transition: 'all 0.3s'
            }}
          >
            {loading ? '⏳ Prediciendo...' : '🔮 Predecir Valor'}
          </button>

          {prediction && (
            <div style={{
              marginTop: '15px',
              padding: '15px',
              background: 'linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%)',
              borderRadius: '8px',
              border: '2px solid #27AE60',
              boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
            }}>
              <p style={{ margin: '0 0 8px 0', fontSize: '12px', fontWeight: '600', color: '#2E7D32' }}>
                ✅ Predicción de <strong>{targetColumn}</strong>:
              </p>
              <p style={{ margin: '0', fontSize: '24px', fontWeight: 'bold', color: '#27AE60' }}>
                {prediction.prediction.toFixed(2)}
              </p>
              <p style={{ margin: '8px 0 0 0', fontSize: '10px', color: '#666' }}>
                Modelo: {prediction.model_algorithm}
              </p>
            </div>
          )}
        </>
      ) : (
        <div>
          <p style={{ fontSize: '11px', color: '#666', marginBottom: '10px' }}>
            Carga un archivo CSV o Excel con las mismas columnas que las características del modelo. 
            El archivo debe tener las columnas: <strong>{features.join(', ')}</strong>
          </p>
          <input
            type="file"
            accept=".csv,.xlsx,.xls"
            onChange={(e) => setMassPredictionFile(e.target.files[0])}
            style={{ marginBottom: '10px', fontSize: '11px' }}
          />
          <button
            onClick={async () => {
              if (!massPredictionFile) {
                alert('Por favor selecciona un archivo')
                return
              }
              
              setLoading(true)
              try {
                const result = await predictBatchWithModel(modelId, massPredictionFile)
                setMassPredictionResults(result)
                
                // Crear CSV con resultados
                const csvContent = [
                  // Encabezados
                  [...features, `prediccion_${targetColumn}`].join(','),
                  // Datos
                  ...result.predictions.map(row => {
                    const values = features.map(f => row[f] || '')
                    values.push(row[`prediccion_${targetColumn}`] || '')
                    return values.join(',')
                  })
                ].join('\n')
                
                // Descargar CSV
                const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
                const link = document.createElement('a')
                const url = URL.createObjectURL(blob)
                link.setAttribute('href', url)
                link.setAttribute('download', `predicciones_${new Date().toISOString().split('T')[0]}.csv`)
                link.style.visibility = 'hidden'
                document.body.appendChild(link)
                link.click()
                document.body.removeChild(link)
                
                alert(`✅ ${result.valid_rows} predicciones completadas. Archivo descargado.`)
              } catch (error) {
                alert('Error al hacer predicción masiva: ' + (error.response?.data?.detail || error.message))
                setMassPredictionResults(null)
              } finally {
                setLoading(false)
              }
            }}
            disabled={!massPredictionFile || loading}
            style={{
              width: '100%',
              padding: '10px',
              background: (massPredictionFile && !loading) ? '#27AE60' : '#95A5A6',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: (massPredictionFile && !loading) ? 'pointer' : 'not-allowed',
              fontSize: '12px',
              fontWeight: '600'
            }}
          >
            {loading ? '⏳ Procesando...' : '📊 Predecir Archivo Completo'}
          </button>
          {massPredictionResults && (
            <div style={{
              marginTop: '15px',
              padding: '10px',
              background: '#E8F5E9',
              borderRadius: '6px',
              fontSize: '11px'
            }}>
              <p style={{ margin: '0 0 5px 0', fontWeight: '600' }}>
                ✅ Predicción Masiva Completada
              </p>
              <p style={{ margin: '0' }}>
                Total filas: {massPredictionResults.total_rows} | 
                Válidas: {massPredictionResults.valid_rows}
              </p>
              <p style={{ margin: '5px 0 0 0', fontSize: '10px', color: '#666' }}>
                El archivo CSV con las predicciones se ha descargado automáticamente
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default PredictionPanel

