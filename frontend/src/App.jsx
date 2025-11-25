import React, { useState } from 'react'
import DataTable from './components/DataTable'
import MLControls from './components/MLControls'
import GraphPanel from './components/GraphPanel'
import ResizablePanels from './components/ResizablePanels'
import './App.css'

function App() {
  const [sessionId, setSessionId] = useState(null)
  const [data, setData] = useState([])
  const [columns, setColumns] = useState([])
  const [filename, setFilename] = useState('')
  const [mlState, setMlState] = useState({
    targetColumn: '',
    selectedFeatures: [],
    algorithm: 'Regresión Lineal Múltiple',
    graphType: 'scatter',
    xAxis: '',
    yAxis: '',
    modelResults: null,
    testSize: 0.2,
    normalizeData: false,
    showResiduals: false
  })

  const handleDataLoaded = (session, loadedData, cols, name) => {
    setSessionId(session)
    setData(loadedData)
    setColumns(cols)
    setFilename(name)
    // Resetear estado de ML cuando se carga un nuevo archivo
    setMlState({
      targetColumn: '',
      selectedFeatures: [],
      algorithm: 'Regresión Lineal Múltiple',
      graphType: 'scatter',
      xAxis: '',
      yAxis: '',
      modelResults: null,
      testSize: 0.2,
      normalizeData: false,
      showResiduals: false
    })
  }

  const handleDataUpdate = (updatedData) => {
    setData(updatedData)
  }

  const handleMlStateUpdate = (updates) => {
    setMlState(prev => ({ ...prev, ...updates }))
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>📊 Gestor de Tablas - Machine Learning</h1>
        {filename && <span className="file-info">{filename}</span>}
      </header>
      <div className="app-grid">
        {/* Izquierda: 2 Paneles redimensionables */}
        <div className="left-panels">
          <ResizablePanels direction="vertical">
            {/* Panel 1: Tabla de datos (incluye estadísticas) */}
            <DataTable
              sessionId={sessionId}
              data={data}
              columns={columns}
              onDataUpdate={handleDataUpdate}
              onDataLoaded={handleDataLoaded}
            />
            
            {/* Panel 2: Visualización de Datos (Gráficos) */}
            <GraphPanel
              sessionId={sessionId}
              data={data}
              columns={columns}
              mlState={mlState}
              onMlStateUpdate={handleMlStateUpdate}
            />
          </ResizablePanels>
        </div>
        
        {/* Derecha: Ajustes Continuos */}
        <div className="grid-cell controls-panel">
          <MLControls
            sessionId={sessionId}
            data={data}
            columns={columns}
            mlState={mlState}
            onMlStateUpdate={handleMlStateUpdate}
          />
        </div>
      </div>
    </div>
  )
}

export default App

