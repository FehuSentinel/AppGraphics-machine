import React, { useState, useRef, useEffect } from 'react'
import './ResizablePanels.css'

const ResizablePanels = ({ children, direction = 'vertical' }) => {
  const [sizes, setSizes] = useState(children.map(() => 100 / children.length))
  const [isResizing, setIsResizing] = useState(null)
  const containerRef = useRef(null)

  const handleMouseDown = (index) => {
    setIsResizing(index)
  }

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (isResizing === null || !containerRef.current) return

      const container = containerRef.current
      const rect = container.getBoundingClientRect()
      
      if (direction === 'vertical') {
        const totalHeight = rect.height
        const y = e.clientY - rect.top
        const percentage = (y / totalHeight) * 100
        
        // Calcular nuevos tamaños
        const newSizes = [...sizes]
        const currentSize = sizes[isResizing]
        const nextSize = sizes[isResizing + 1]
        const totalCurrent = currentSize + nextSize
        
        // Limitar el tamaño mínimo (10% cada panel)
        const minSize = 10
        const maxSize = totalCurrent - minSize
        
        if (percentage >= minSize && percentage <= maxSize) {
          newSizes[isResizing] = percentage
          newSizes[isResizing + 1] = totalCurrent - percentage
          setSizes(newSizes)
        }
      } else {
        const totalWidth = rect.width
        const x = e.clientX - rect.left
        const percentage = (x / totalWidth) * 100
        
        const newSizes = [...sizes]
        const currentSize = sizes[isResizing]
        const nextSize = sizes[isResizing + 1]
        const totalCurrent = currentSize + nextSize
        
        const minSize = 10
        const maxSize = totalCurrent - minSize
        
        if (percentage >= minSize && percentage <= maxSize) {
          newSizes[isResizing] = percentage
          newSizes[isResizing + 1] = totalCurrent - percentage
          setSizes(newSizes)
        }
      }
    }

    const handleMouseUp = () => {
      setIsResizing(null)
    }

    if (isResizing !== null) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = direction === 'vertical' ? 'row-resize' : 'col-resize'
      document.body.style.userSelect = 'none'
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
  }, [isResizing, sizes, direction])

  return (
    <div 
      ref={containerRef}
      className={`resizable-panels resizable-panels-${direction}`}
    >
      {children.map((child, index) => (
        <React.Fragment key={index}>
          <div 
            className="resizable-panel"
            style={{ 
              [direction === 'vertical' ? 'height' : 'width']: `${sizes[index]}%` 
            }}
          >
            {child}
          </div>
          {index < children.length - 1 && (
            <div
              className={`resizer resizer-${direction} ${isResizing === index ? 'resizing' : ''}`}
              onMouseDown={() => handleMouseDown(index)}
            />
          )}
        </React.Fragment>
      ))}
    </div>
  )
}

export default ResizablePanels

