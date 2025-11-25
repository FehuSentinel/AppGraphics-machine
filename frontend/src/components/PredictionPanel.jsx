import React, { useState, useEffect, useMemo } from 'react'
import { predictWithModel, getModelDetails } from '../services/api'

// Componente para panel de predicción mejorado
const PredictionPanel = ({ modelId, features, targetColumn, data, correlations, onPrediction }) => {
  const [inputValues, setInputValues] = useState({})
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [statistics, setStatistics] = useState({})
  const [selectedFeaturesToEdit, setSelectedFeaturesToEdit] = useState(new Set()) // Features seleccionadas para editar
  const [requiredFeatures, setRequiredFeatures] = useState(features) // Features requeridas por el modelo

  // Obtener features que el modelo realmente usa
  useEffect(() => {
    const fetchModelFeatures = async () => {
      if (modelId) {
        try {
          const modelDetails = await getModelDetails(modelId)
          // CRÍTICO: El modelo solo puede predecir usando las features que realmente usa
          // Si hay feature selection, solo esas features importan
          const modelFeatures = modelDetails.features || []
          
          console.log('📋 Features que el modelo REALMENTE usa:', modelFeatures)
          console.log('📋 Features originales (no usadas):', modelDetails.original_features)
          
          if (modelFeatures.length === 0) {
            console.warn('⚠️ No se pudieron obtener las features del modelo, usando features por defecto')
            setRequiredFeatures(features)
          } else {
            // Usar SOLO las features que el modelo realmente usa
            // Esto asegura que cualquier cambio que haga el usuario tenga efecto
            setRequiredFeatures(modelFeatures)
          }
        } catch (error) {
          console.error('❌ Error obteniendo detalles del modelo:', error)
          // Usar features por defecto si falla
          setRequiredFeatures(features)
        }
      }
    }
    fetchModelFeatures()
  }, [modelId, features])

  // Calcular estadísticas
  useEffect(() => {
    if (data && data.length > 0 && requiredFeatures.length > 0) {
      const stats = {}
      
      requiredFeatures.forEach(feature => {
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

      setStatistics(stats)
      
      // No inicializar con valores predefinidos - dejar vacío para que el usuario elija
      setInputValues({})
    }
  }, [requiredFeatures, data])

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
    // Validar que haya al menos un campo seleccionado
    if (selectedFeaturesToEdit.size === 0) {
      alert('Por favor selecciona al menos un campo para modificar')
      return
    }
    
    // Validar que los valores estén presentes para los campos seleccionados
    const missing = Array.from(selectedFeaturesToEdit).filter(f => !inputValues[f] || inputValues[f] === '')
    if (missing.length > 0) {
      alert(`Por favor complete los valores para: ${missing.join(', ')}`)
      return
    }

    // Construir valores finales: TODOS los campos requeridos deben tener valor
    // Si el usuario seleccionó un campo, usar su valor. Si no, usar valor inteligente basado en correlaciones
    const finalValues = {}
    const modifiedFeatures = []
    const defaultFeatures = []
    
    // Obtener el primer valor modificado para calcular valores inteligentes basados en correlación
    const firstModifiedFeature = Array.from(selectedFeaturesToEdit).find(f => 
      inputValues[f] !== undefined && inputValues[f] !== ''
    )
    const firstModifiedValue = firstModifiedFeature ? parseFloat(inputValues[firstModifiedFeature]) : null
    
    requiredFeatures.forEach(f => {
      if (selectedFeaturesToEdit.has(f) && inputValues[f] !== undefined && inputValues[f] !== '') {
        // Usar valor ingresado por el usuario
        const val = inputValues[f]
        // Intentar convertir a número si es posible
        const numVal = parseFloat(val)
        if (!isNaN(numVal)) {
          finalValues[f] = numVal
          modifiedFeatures.push(f)
        } else {
          // Mantener como string (para features categóricas)
          finalValues[f] = val
          modifiedFeatures.push(f)
        }
      } else {
        // Para campos no seleccionados, usar valor inteligente:
        // 1. Si hay correlación con la feature modificada, usar valor proporcional
        // 2. Si no, usar media
        if (statistics[f]) {
          let defaultValue = statistics[f].mean
          
          // Si hay una feature modificada y hay correlación, ajustar el valor por defecto
          if (firstModifiedFeature && firstModifiedValue && !isNaN(firstModifiedValue) && correlations) {
            const corr = correlations[firstModifiedFeature]?.[f] || correlations[f]?.[firstModifiedFeature]
            if (corr && Math.abs(corr) > 0.3) {
              // Hay correlación significativa: ajustar el valor por defecto proporcionalmente
              const meanModified = statistics[firstModifiedFeature]?.mean || firstModifiedValue
              const ratio = firstModifiedValue / meanModified
              // Ajustar el valor por defecto proporcionalmente (pero no demasiado extremo)
              const adjustedValue = statistics[f].mean * (1 + (ratio - 1) * Math.abs(corr) * 0.5)
              // Mantener dentro de rangos razonables
              defaultValue = Math.max(
                statistics[f].min,
                Math.min(statistics[f].max, adjustedValue)
              )
              console.log(`📊 Ajustando ${f} basado en correlación (${corr.toFixed(2)}) con ${firstModifiedFeature}`)
            }
          }
          
          finalValues[f] = defaultValue
          defaultFeatures.push(f)
        } else {
          // Feature categórica - usar primer valor único disponible
          const uniqueValues = [...new Set(data.map(row => row[f]).filter(v => v != null && v !== ''))]
          finalValues[f] = uniqueValues.length > 0 ? uniqueValues[0] : ''
          defaultFeatures.push(f)
        }
      }
    })
    
    console.log('📊 Valores finales construidos:', finalValues)
    console.log('📊 Campos MODIFICADOS por usuario:', modifiedFeatures)
    console.log('📊 Campos con valores por DEFECTO (ajustados inteligentemente):', defaultFeatures)
    console.log('📊 Valores modificados:', Object.fromEntries(modifiedFeatures.map(f => [f, finalValues[f]])))
    console.log('📊 Valores por defecto:', Object.fromEntries(defaultFeatures.map(f => [f, finalValues[f]])))
    
    // Mostrar resumen visual antes de enviar
    if (defaultFeatures.length > 0) {
      const defaultSummary = defaultFeatures.map(f => `${f}=${statistics[f] ? finalValues[f].toFixed(2) : finalValues[f]}`).join(', ')
      console.log(`ℹ️ Valores por defecto que se usarán: ${defaultSummary}`)
    }

    if (Object.keys(finalValues).length !== requiredFeatures.length) {
      alert(`Error: Faltan valores para algunas features. Requeridas: ${requiredFeatures.length}, Obtenidas: ${Object.keys(finalValues).length}`)
      return
    }

    setLoading(true)
    try {
      console.log('🔮 Enviando predicción:')
      console.log('   Model ID:', modelId)
      console.log('   Features requeridas:', requiredFeatures)
      console.log('   Valores enviados:', finalValues)
      console.log('   Campos seleccionados:', Array.from(selectedFeaturesToEdit))
      console.log('   Campos con valores por defecto:', requiredFeatures.filter(f => !selectedFeaturesToEdit.has(f)))
      
      const result = await predictWithModel(modelId, finalValues)
      console.log('✅ Predicción exitosa:', result)
      setPrediction(result)
      if (onPrediction) {
        onPrediction(result, finalValues)
      }
    } catch (error) {
      console.error('❌ Error en predicción:', error)
      console.error('❌ Error response:', error.response)
      console.error('❌ Error data:', error.response?.data)
      const errorMessage = error.response?.data?.detail || error.response?.data?.message || error.message
      alert('Error al hacer predicción: ' + errorMessage)
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
      {/* Seleccionar qué campos modificar */}
          <div style={{ 
            marginBottom: '15px', 
            padding: '10px', 
            background: '#F8F9FA', 
            borderRadius: '6px',
            border: '1px solid #DEE2E6'
          }}>
            <p style={{ fontSize: '11px', fontWeight: '600', marginBottom: '8px', color: '#2C3E50' }}>
              🎯 Selecciona los campos que quieres modificar:
            </p>
            <p style={{ fontSize: '10px', color: '#666', marginBottom: '8px' }}>
              Haz clic en los campos que deseas cambiar. Los demás usarán valores por defecto.
            </p>
            <div style={{ display: 'flex', gap: '5px', flexWrap: 'wrap' }}>
              {requiredFeatures.map(feature => (
                <button
                  key={feature}
                  onClick={() => {
                    const newSet = new Set(selectedFeaturesToEdit)
                    if (newSet.has(feature)) {
                      newSet.delete(feature)
                    } else {
                      newSet.add(feature)
                    }
                    setSelectedFeaturesToEdit(newSet)
                  }}
                  style={{
                    padding: '6px 12px',
                    background: selectedFeaturesToEdit.has(feature) ? '#27AE60' : '#ECF0F1',
                    color: selectedFeaturesToEdit.has(feature) ? 'white' : '#2C3E50',
                    border: `2px solid ${selectedFeaturesToEdit.has(feature) ? '#229954' : '#BDC3C7'}`,
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '11px',
                    fontWeight: selectedFeaturesToEdit.has(feature) ? '600' : '400',
                    transition: 'all 0.2s',
                    boxShadow: selectedFeaturesToEdit.has(feature) ? '0 2px 4px rgba(0,0,0,0.1)' : 'none'
                  }}
                >
                  {selectedFeaturesToEdit.has(feature) ? '✓ ' : ''}{feature}
                </button>
              ))}
            </div>
            {selectedFeaturesToEdit.size === 0 && (
              <small style={{ fontSize: '10px', color: '#E74C3C', display: 'block', marginTop: '8px', fontWeight: '600' }}>
                ⚠️ Selecciona al menos un campo para modificar antes de predecir
              </small>
            )}
          </div>

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

          {/* Inputs de Valores - Solo mostrar campos seleccionados */}
          {selectedFeaturesToEdit.size > 0 && (
            <div style={{ marginBottom: '15px' }}>
              <p style={{ fontSize: '11px', fontWeight: '600', marginBottom: '10px', color: '#2C3E50' }}>
                ✏️ Ingresa los valores para los campos seleccionados:
              </p>
              {Array.from(selectedFeaturesToEdit).map(feature => {
              
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
                  {statistics[feature] ? (
                    // Input numérico
                    <input
                      type="number"
                      step="any"
                      value={value}
                      onChange={(e) => handleInputChange(feature, e.target.value)}
                      placeholder={`Rango: ${statistics[feature].min.toFixed(2)} - ${statistics[feature].max.toFixed(2)}`}
                    style={{
                      width: '100%',
                      padding: '8px',
                      border: `2px solid ${!value || inRange ? '#BDC3C7' : '#F39C12'}`,
                      borderRadius: '6px',
                      fontSize: '12px',
                      transition: 'border-color 0.3s',
                      background: !value || inRange ? 'white' : '#FFF9E6'
                    }}
                    onFocus={(e) => e.target.style.borderColor = '#27AE60'}
                    onBlur={(e) => e.target.style.borderColor = !value || inRange ? '#BDC3C7' : '#F39C12'}
                    />
                  ) : (
                    // Input de texto (para features categóricas)
                    <input
                      type="text"
                      value={value}
                      onChange={(e) => handleInputChange(feature, e.target.value)}
                      placeholder="Ingrese valor"
                      style={{
                        width: '100%',
                        padding: '8px',
                        border: '2px solid #BDC3C7',
                        borderRadius: '6px',
                        fontSize: '12px',
                        transition: 'border-color 0.3s',
                        background: 'white'
                      }}
                      onFocus={(e) => e.target.style.borderColor = '#27AE60'}
                      onBlur={(e) => e.target.style.borderColor = '#BDC3C7'}
                    />
                  )}
                  {!inRange && value && (
                    <small style={{ fontSize: '9px', color: '#F39C12', display: 'block', marginTop: '3px' }}>
                      💡 Valor fuera del rango de entrenamiento - La predicción puede ser menos confiable
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
          )}

          {/* Resumen de valores que se usarán */}
          {selectedFeaturesToEdit.size > 0 && (
            <div style={{
              marginBottom: '15px',
              padding: '10px',
              background: '#E8F5E9',
              borderRadius: '6px',
              border: '1px solid #81C784',
              fontSize: '10px'
            }}>
              <p style={{ margin: '0 0 8px 0', fontWeight: '600', color: '#2E7D32' }}>
                📋 Valores que se usarán para la predicción:
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
                {/* Valores modificados por el usuario */}
                {Array.from(selectedFeaturesToEdit).filter(f => inputValues[f] && inputValues[f] !== '').map(f => (
                  <div key={f} style={{ 
                    padding: '5px', 
                    background: '#C8E6C9', 
                    borderRadius: '4px',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center'
                  }}>
                    <span style={{ fontWeight: '600', color: '#1B5E20' }}>
                      ✓ {f}:
                    </span>
                    <span style={{ color: '#2E7D32', fontWeight: '600' }}>
                      {inputValues[f]} (modificado)
                    </span>
                  </div>
                ))}
                {/* Valores por defecto */}
                {requiredFeatures.filter(f => !selectedFeaturesToEdit.has(f) || !inputValues[f] || inputValues[f] === '').map(f => {
                  const defaultValue = statistics[f] ? statistics[f].mean : (() => {
                    const uniqueValues = [...new Set(data.map(row => row[f]).filter(v => v != null && v !== ''))]
                    return uniqueValues.length > 0 ? uniqueValues[0] : ''
                  })()
                  return (
                    <div key={f} style={{ 
                      padding: '5px', 
                      background: '#F1F8E9', 
                      borderRadius: '4px',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      opacity: 0.8
                    }}>
                      <span style={{ color: '#558B2F' }}>
                        ○ {f}:
                      </span>
                      <span style={{ color: '#689F38', fontSize: '9px' }}>
                        {statistics[f] ? defaultValue.toFixed(2) : defaultValue} (por defecto)
                      </span>
                    </div>
                  )
                })}
              </div>
              <small style={{ fontSize: '9px', color: '#558B2F', display: 'block', marginTop: '8px', fontStyle: 'italic' }}>
                💡 Los valores por defecto se ajustan automáticamente según las correlaciones con los valores modificados
              </small>
            </div>
          )}
          
          {/* Resumen de valores (versión antigua - mantener por compatibilidad) */}
          {selectedFeaturesToEdit.size > 0 && false && (
            <div style={{
              marginBottom: '15px',
              padding: '10px',
              background: '#FFF9E6',
              borderRadius: '6px',
              border: '1px solid #F7DC6F',
              fontSize: '10px'
            }}>
              <p style={{ margin: '0 0 5px 0', fontWeight: '600', color: '#856404' }}>
                📋 Resumen de valores:
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
                {requiredFeatures.map(feature => {
                  const isEditing = selectedFeaturesToEdit.has(feature)
                  const value = isEditing 
                    ? (inputValues[feature] || '') 
                    : (statistics[feature] ? statistics[feature].mean.toFixed(2) : '0.00')
                  
                  return (
                    <div key={feature} style={{ 
                      display: 'flex', 
                      justifyContent: 'space-between',
                      padding: '3px 0',
                      borderBottom: '1px solid #F7DC6F'
                    }}>
                      <span style={{ fontWeight: isEditing ? '600' : '400' }}>
                        {isEditing ? '✏️' : '📌'} {feature}:
                      </span>
                      <span style={{ 
                        color: isEditing ? '#27AE60' : '#666',
                        fontWeight: isEditing ? '600' : '400'
                      }}>
                        {value} {!isEditing && '(por defecto)'}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          <button
            onClick={handlePredict}
            disabled={loading || selectedFeaturesToEdit.size === 0}
            style={{
              width: '100%',
              padding: '12px',
              background: (loading || selectedFeaturesToEdit.size === 0) 
                ? '#95A5A6' 
                : 'linear-gradient(135deg, #27AE60 0%, #229954 100%)',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: (loading || selectedFeaturesToEdit.size === 0) 
                ? 'not-allowed' 
                : 'pointer',
              fontWeight: '600',
              fontSize: '13px',
              boxShadow: (loading || selectedFeaturesToEdit.size === 0) 
                ? 'none' 
                : '0 2px 4px rgba(0,0,0,0.2)',
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
    </div>
  )
}

export default PredictionPanel

