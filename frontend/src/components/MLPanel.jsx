import React, { useState, useEffect } from 'react'
import { trainModel } from '../services/api'
import GraphView from './GraphView'
import './MLPanel.css'

const MLPanel = ({ sessionId, data, columns, onDataUpdate }) => {
  const [targetColumn, setTargetColumn] = useState('')
  const [selectedFeatures, setSelectedFeatures] = useState([])
  const [algorithm, setAlgorithm] = useState('Regresión Lineal Múltiple')
  const [graphType, setGraphType] = useState('scatter')
  const [xAxis, setXAxis] = useState('')
  const [yAxis, setYAxis] = useState('')
  const [modelResults, setModelResults] = useState(null)
  const [training, setTraining] = useState(false)

  useEffect(() => {
    if (columns.length > 0) {
      // Auto-seleccionar última columna como target
      setTargetColumn(columns[columns.length - 1])
      // Auto-seleccionar todas las demás como features
      setSelectedFeatures(columns.slice(0, -1))
      // Auto-seleccionar ejes
      if (columns.length >= 2) {
        setXAxis(columns[0])
        setYAxis(columns[columns.length - 1])
      }
    }
  }, [columns])

  const handleFeatureToggle = (feature) => {
    setSelectedFeatures(prev => 
      prev.includes(feature)
        ? prev.filter(f => f !== feature)
        : [...prev, feature]
    )
  }

  const handleTrain = async () => {
    if (!sessionId || !targetColumn || selectedFeatures.length === 0) {
      alert('Seleccione variable objetivo y al menos una característica')
      return
    }

    setTraining(true)
    try {
      const result = await trainModel({
        session_id: sessionId,
        algorithm,
        target_column: targetColumn,
        features: selectedFeatures,
        test_size: 0.2,
        random_state: 42
      })
      setModelResults(result)
    } catch (error) {
      alert('Error al entrenar modelo: ' + error.message)
    } finally {
      setTraining(false)
    }
  }

  const numericColumns = columns.filter(col => {
    if (data.length === 0) return false
    const sample = data[0][col]
    return typeof sample === 'number' || !isNaN(parseFloat(sample))
  })

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

  return (
    <div className="ml-panel">
      <div className="ml-header">
        <h2>🤖 Machine Learning</h2>
      </div>

      <div className="ml-content">
        {/* Selección de Variables */}
        <div className="ml-section">
          <h3>📊 Selección de Variables</h3>
          
          <div className="form-group">
            <label>Variable Objetivo (Y):</label>
            <select 
              value={targetColumn} 
              onChange={(e) => setTargetColumn(e.target.value)}
              className="form-select"
            >
              <option value="">Seleccione...</option>
              {numericColumns.map(col => (
                <option key={col} value={col}>{col}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Características (X):</label>
            <div className="features-list">
              {numericColumns
                .filter(col => col !== targetColumn)
                .map(col => (
                  <label key={col} className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={selectedFeatures.includes(col)}
                      onChange={() => handleFeatureToggle(col)}
                    />
                    {col}
                  </label>
                ))}
            </div>
          </div>
        </div>

        {/* Configuración del Modelo */}
        <div className="ml-section">
          <h3>⚙️ Configuración del Modelo</h3>
          
          <div className="form-group">
            <label>Algoritmo:</label>
            <select 
              value={algorithm} 
              onChange={(e) => setAlgorithm(e.target.value)}
              className="form-select"
            >
              <option value="Regresión Lineal Simple">Regresión Lineal Simple</option>
              <option value="Regresión Lineal Múltiple">Regresión Lineal Múltiple</option>
              <option value="Ridge Regression">Ridge Regression</option>
              <option value="Lasso Regression">Lasso Regression</option>
              <option value="Random Forest">Random Forest</option>
              <option value="Gradient Boosting">Gradient Boosting</option>
              <option value="Decision Tree">Decision Tree</option>
            </select>
          </div>

          <button 
            className="btn-train"
            onClick={handleTrain}
            disabled={training || !targetColumn || selectedFeatures.length === 0}
          >
            {training ? '⏳ Entrenando...' : '🚀 Entrenar Modelo'}
          </button>
        </div>

        {/* Resultados */}
        {modelResults && (
          <div className="ml-section">
            <h3>📈 Resultados</h3>
            <div className="results">
              <p><strong>R² Score (Test):</strong> {modelResults.metrics.test_r2.toFixed(4)}</p>
              <p><strong>RMSE (Test):</strong> {modelResults.metrics.test_rmse.toFixed(4)}</p>
              <p><strong>MAE (Test):</strong> {modelResults.metrics.test_mae.toFixed(4)}</p>
            </div>
          </div>
        )}

        {/* Configuración de Gráfico */}
        <div className="ml-section">
          <h3>📊 Visualización</h3>
          
          <div className="form-group">
            <label>Tipo de Gráfico:</label>
            <select 
              value={graphType} 
              onChange={(e) => setGraphType(e.target.value)}
              className="form-select"
            >
              <option value="scatter">Dispersión</option>
              <option value="line">Línea</option>
              <option value="bar">Barras</option>
            </select>
          </div>

          <div className="form-group">
            <label>Eje X:</label>
            <select 
              value={xAxis} 
              onChange={(e) => setXAxis(e.target.value)}
              className="form-select"
            >
              <option value="">Auto</option>
              {numericColumns.map(col => (
                <option key={col} value={col}>{col}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Eje Y:</label>
            <select 
              value={yAxis} 
              onChange={(e) => setYAxis(e.target.value)}
              className="form-select"
            >
              <option value="">Auto</option>
              {numericColumns.map(col => (
                <option key={col} value={col}>{col}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Gráfico - Siempre visible */}
        <div className="ml-section">
          <h3>📊 Gráfico</h3>
          {data.length > 0 ? (
            <GraphView
              data={data}
              columns={columns}
              graphType={graphType}
              xAxis={xAxis || (numericColumns[0] || '')}
              yAxis={yAxis || targetColumn || (numericColumns[1] || '')}
              modelResults={modelResults}
            />
          ) : (
            <div className="graph-placeholder">
              <p>Carga datos para ver el gráfico</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default MLPanel

