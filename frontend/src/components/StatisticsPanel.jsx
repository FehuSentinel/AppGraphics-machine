import React, { useState, useEffect } from 'react'
import { getStatistics } from '../services/api'
import './StatisticsPanel.css'

const StatisticsPanel = ({ sessionId, data }) => {
  const [statistics, setStatistics] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (sessionId && data.length > 0) {
      const timer = setTimeout(() => {
        loadStatistics()
      }, 500)
      return () => clearTimeout(timer)
    }
  }, [sessionId, data.length])

  const loadStatistics = async () => {
    if (!sessionId) return
    setLoading(true)
    try {
      const result = await getStatistics(sessionId)
      setStatistics(result.statistics)
    } catch (error) {
      if (error.response?.status !== 404) {
        console.error('Error al cargar estadísticas:', error)
      }
    } finally {
      setLoading(false)
    }
  }

  if (!sessionId || data.length === 0) {
    return (
      <div className="statistics-panel-container">
        <div className="statistics-panel-header">
          <h3>📊 Estadísticas Descriptivas</h3>
        </div>
        <div className="statistics-panel-content">
          <p style={{ textAlign: 'center', color: '#666', padding: '20px' }}>
            Carga un archivo para ver las estadísticas
          </p>
        </div>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="statistics-panel-container">
        <div className="statistics-panel-header">
          <h3>📊 Estadísticas Descriptivas (Post-Procesamiento)</h3>
        </div>
        <div className="statistics-panel-content">
          <p style={{ textAlign: 'center', color: '#666', padding: '20px' }}>
            Cargando estadísticas...
          </p>
        </div>
      </div>
    )
  }

  if (!statistics || Object.keys(statistics).length === 0) {
    return (
      <div className="statistics-panel-container">
        <div className="statistics-panel-header">
          <h3>📊 Estadísticas Descriptivas (Post-Procesamiento)</h3>
        </div>
        <div className="statistics-panel-content">
          <p style={{ textAlign: 'center', color: '#666', padding: '20px' }}>
            No hay estadísticas disponibles
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="statistics-panel-container">
      <div className="statistics-panel-header">
        <h3>📊 Estadísticas Descriptivas (Post-Procesamiento)</h3>
      </div>
      <div className="statistics-panel-content">
        <div className="statistics-table-wrapper">
          <table className="statistics-table">
            <thead>
              <tr>
                <th>Columna</th>
                <th>Count</th>
                <th>Mean</th>
                <th>Std</th>
                <th>Min</th>
                <th>25%</th>
                <th>50%</th>
                <th>75%</th>
                <th>Max</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(statistics).map(([col, stats]) => (
                <tr key={col}>
                  <td className="stat-col-name">{col}</td>
                  <td>{stats.count.toFixed(0)}</td>
                  <td>{stats.mean.toFixed(2)}</td>
                  <td>{stats.std.toFixed(2)}</td>
                  <td>{stats.min.toFixed(2)}</td>
                  <td>{stats['25%'].toFixed(2)}</td>
                  <td>{stats['50%'].toFixed(2)}</td>
                  <td>{stats['75%'].toFixed(2)}</td>
                  <td>{stats.max.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

export default StatisticsPanel

