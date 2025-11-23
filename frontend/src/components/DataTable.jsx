import React, { useState, useRef, useEffect } from 'react'
import { uploadCSV, uploadExcel, updateData, addRow, deleteRow, addColumn, deleteColumn, getData, getStatistics } from '../services/api'
import './DataTable.css'

const DataTable = ({ sessionId, data, columns, onDataUpdate, onDataLoaded }) => {
  const [loading, setLoading] = useState(false)
  const [statistics, setStatistics] = useState(null)
  const [showStats, setShowStats] = useState(true)
  const fileInputRef = useRef(null)

  // Cargar estadísticas cuando cambian los datos
  useEffect(() => {
    if (sessionId && data.length > 0) {
      // Esperar un poco para asegurar que el backend procesó los datos
      const timer = setTimeout(() => {
        loadStatistics()
      }, 500)
      return () => clearTimeout(timer)
    }
  }, [sessionId, data.length])

  const loadStatistics = async () => {
    if (!sessionId) return
    try {
      const result = await getStatistics(sessionId)
      setStatistics(result.statistics)
    } catch (error) {
      // No mostrar error si es 404, puede ser que la sesión aún no esté lista
      if (error.response?.status !== 404) {
        console.error('Error al cargar estadísticas:', error)
      }
    }
  }

  const handleFileUpload = async (event) => {
    const file = event.target.files[0]
    if (!file) return

    setLoading(true)
    try {
      let result
      if (file.name.endsWith('.csv')) {
        result = await uploadCSV(file)
      } else if (file.name.endsWith('.xlsx') || file.name.endsWith('.xls')) {
        result = await uploadExcel(file)
      } else {
        alert('Formato no soportado. Use CSV o Excel.')
        return
      }

      onDataLoaded(result.session_id, result.data, result.columns_list, result.filename)
      // Cargar estadísticas después de cargar datos
      setTimeout(() => loadStatistics(), 500)
    } catch (error) {
      alert('Error al cargar archivo: ' + error.message)
    } finally {
      setLoading(false)
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  const handleCellChange = async (rowIndex, colIndex, value) => {
    if (!sessionId) return
    
    try {
      await updateData(sessionId, rowIndex, colIndex, value)
      const updated = await getData(sessionId)
      onDataUpdate(updated.data)
      // Recargar estadísticas después de un pequeño delay para asegurar que el backend procesó
      setTimeout(() => loadStatistics(), 300)
    } catch (error) {
      alert('Error al actualizar: ' + error.message)
    }
  }

  const handleAddRow = async () => {
    if (!sessionId || columns.length === 0) return
    
    const newRow = {}
    columns.forEach(col => {
      newRow[col] = 0
    })
    
    try {
      await addRow(sessionId, newRow)
      const updated = await getData(sessionId)
      onDataUpdate(updated.data)
      loadStatistics()
    } catch (error) {
      alert('Error al agregar fila: ' + error.message)
    }
  }

  const handleDeleteRow = async (rowIndex) => {
    if (!sessionId) return
    
    if (!confirm('¿Eliminar esta fila?')) return
    
    try {
      await deleteRow(sessionId, rowIndex)
      const updated = await getData(sessionId)
      onDataUpdate(updated.data)
      loadStatistics()
    } catch (error) {
      alert('Error al eliminar fila: ' + error.message)
    }
  }

  const handleAddColumn = async () => {
    if (!sessionId) return
    
    const colName = prompt('Nombre de la nueva columna:')
    if (!colName) return
    
    try {
      await addColumn(sessionId, colName)
      const updated = await getData(sessionId)
      onDataUpdate(updated.data)
      loadStatistics()
      // Actualizar columnas
      window.location.reload()
    } catch (error) {
      alert('Error al agregar columna: ' + error.message)
    }
  }

  const handleDeleteColumn = async (colName) => {
    if (!sessionId) return
    
    if (!confirm(`¿Eliminar la columna "${colName}"?`)) return
    
    try {
      await deleteColumn(sessionId, colName)
      const updated = await getData(sessionId)
      onDataUpdate(updated.data)
      loadStatistics()
      window.location.reload()
    } catch (error) {
      alert('Error al eliminar columna: ' + error.message)
    }
  }

  return (
    <div className="data-table-container">
      <div className="table-toolbar">
        <button 
          className="btn-primary"
          onClick={() => fileInputRef.current?.click()}
          disabled={loading}
        >
          {loading ? 'Cargando...' : '📊 Cargar Archivo'}
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.xlsx,.xls"
          onChange={handleFileUpload}
          style={{ display: 'none' }}
        />
        {sessionId && (
          <>
            <button className="btn-secondary" onClick={handleAddRow}>
              ➕ Agregar Fila
            </button>
            <button className="btn-secondary" onClick={handleAddColumn}>
              ➕ Agregar Columna
            </button>
          </>
        )}
        {sessionId && (
          <div className="data-info">
            {data.length} filas × {columns.length} columnas
          </div>
        )}
      </div>

      {data.length > 0 ? (
        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>#</th>
                {columns.map((col, idx) => (
                  <th key={idx}>
                    <div className="header-cell">
                      <span>{col}</span>
                      <button 
                        className="btn-delete-small"
                        onClick={() => handleDeleteColumn(col)}
                        title="Eliminar columna"
                      >
                        ×
                      </button>
                    </div>
                  </th>
                ))}
                <th>Acciones</th>
              </tr>
            </thead>
            <tbody>
              {data.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  <td className="row-number">{rowIndex + 1}</td>
                  {columns.map((col, colIndex) => (
                    <td key={colIndex}>
                      <input
                        type="text"
                        value={row[col] ?? ''}
                        onChange={(e) => handleCellChange(rowIndex, colIndex, e.target.value)}
                        className="cell-input"
                      />
                    </td>
                  ))}
                  <td>
                    <button
                      className="btn-delete"
                      onClick={() => handleDeleteRow(rowIndex)}
                      title="Eliminar fila"
                    >
                      🗑️
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="empty-state">
          <p>📊 Carga un archivo CSV o Excel para comenzar</p>
        </div>
      )}

      {/* Panel de Estadísticas Descriptivas */}
      {sessionId && data.length > 0 && statistics && Object.keys(statistics).length > 0 && (
        <div className="statistics-panel">
          <div className="statistics-header" onClick={() => setShowStats(!showStats)}>
            <h3>📊 Estadísticas Descriptivas (Post-Procesamiento)</h3>
            <span className="toggle-icon">{showStats ? '▼' : '▶'}</span>
          </div>
          {showStats && (
            <div className="statistics-content">
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
          )}
        </div>
      )}
    </div>
  )
}

export default DataTable

