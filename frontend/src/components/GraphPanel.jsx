import React, { useRef } from 'react'
import GraphView from './GraphView'
import './GraphPanel.css'

const GraphPanel = ({ sessionId, data, columns, mlState }) => {
  const graphRef = useRef(null)

  const numericColumns = columns.filter(col => {
    if (data.length === 0) return false
    const sample = data[0][col]
    return typeof sample === 'number' || !isNaN(parseFloat(sample))
  })

  const handleDownloadGraph = () => {
    if (!graphRef.current) return

    const svgElement = graphRef.current.querySelector('svg')
    if (!svgElement) {
      alert('No se pudo encontrar el gráfico para descargar')
      return
    }

    try {
      // Obtener el SVG y sus dimensiones
      const svgData = new XMLSerializer().serializeToString(svgElement)
      const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' })
      const url = URL.createObjectURL(svgBlob)
      
      // Crear imagen
      const img = new Image()
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')

      img.onload = () => {
        // Establecer dimensiones del canvas
        canvas.width = img.width || 800
        canvas.height = img.height || 600
        
        // Dibujar imagen en canvas
        ctx.fillStyle = 'white'
        ctx.fillRect(0, 0, canvas.width, canvas.height)
        ctx.drawImage(img, 0, 0)
        
        // Convertir a PNG y descargar
        canvas.toBlob((blob) => {
          if (blob) {
            const downloadUrl = URL.createObjectURL(blob)
            const link = document.createElement('a')
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
            link.download = `grafico_${timestamp}.png`
            link.href = downloadUrl
            document.body.appendChild(link)
            link.click()
            document.body.removeChild(link)
            URL.revokeObjectURL(downloadUrl)
          }
        }, 'image/png')
        
        URL.revokeObjectURL(url)
      }

      img.onerror = () => {
        // Fallback: descargar SVG directamente
        const link = document.createElement('a')
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
        link.download = `grafico_${timestamp}.svg`
        link.href = url
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
        URL.revokeObjectURL(url)
      }

      img.src = url
    } catch (error) {
      console.error('Error al descargar gráfico:', error)
      alert('Error al descargar el gráfico. Intente nuevamente.')
    }
  }

  if (!sessionId) {
    return (
      <div className="graph-panel">
        <div className="graph-panel-header">
          <h2>📊 Gráfico</h2>
        </div>
        <div className="graph-panel-content">
          <div className="graph-placeholder">
            <p>Carga un archivo para ver el gráfico</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="graph-panel">
      <div className="graph-panel-header">
        <h2>📊 Visualización de Datos</h2>
        {data.length > 0 && (
          <button 
            className="btn-download"
            onClick={handleDownloadGraph}
            title="Descargar gráfico como imagen"
          >
            💾 Descargar
          </button>
        )}
      </div>
      <div className="graph-panel-content" ref={graphRef}>
        {data.length > 0 ? (
          <GraphView
            data={data}
            columns={columns}
            graphType={mlState.graphType}
            targetColumn={mlState.targetColumn}
            selectedFeatures={mlState.selectedFeatures}
            modelResults={mlState.modelResults}
            mlState={mlState}
          />
        ) : (
          <div className="graph-placeholder">
            <p>Carga datos para ver el gráfico</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default GraphPanel

