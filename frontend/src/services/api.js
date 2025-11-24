import axios from 'axios'

const API_BASE = 'http://localhost:8000/api'

export const uploadCSV = async (file) => {
  const formData = new FormData()
  formData.append('file', file)
  const response = await axios.post(`${API_BASE}/upload/csv`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
  return response.data
}

export const uploadExcel = async (file) => {
  const formData = new FormData()
  formData.append('file', file)
  const response = await axios.post(`${API_BASE}/upload/excel`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
  return response.data
}

export const getData = async (sessionId) => {
  const response = await axios.get(`${API_BASE}/data/${sessionId}`)
  return response.data
}

export const updateData = async (sessionId, row, col, value) => {
  const response = await axios.put(`${API_BASE}/data/${sessionId}`, {
    row,
    col,
    value
  })
  return response.data
}

export const addRow = async (sessionId, rowData) => {
  const response = await axios.post(`${API_BASE}/data/${sessionId}/row`, {
    data: rowData
  })
  return response.data
}

export const deleteRow = async (sessionId, rowIndex) => {
  const response = await axios.delete(`${API_BASE}/data/${sessionId}/row/${rowIndex}`)
  return response.data
}

export const addColumn = async (sessionId, columnName) => {
  const response = await axios.post(`${API_BASE}/data/${sessionId}/column`, {
    column_name: columnName
  })
  return response.data
}

export const deleteColumn = async (sessionId, columnName) => {
  const response = await axios.delete(`${API_BASE}/data/${sessionId}/column/${columnName}`)
  return response.data
}

export const trainModel = async (request) => {
  try {
    console.log('📤 Enviando request a:', `${API_BASE}/model/train`)
    console.log('📤 Request data:', request)
    const response = await axios.post(`${API_BASE}/model/train`, request)
    console.log('📥 Response status:', response.status)
    console.log('📥 Response data:', response.data)
    console.log('📥 Response data type:', typeof response.data)
    console.log('📥 Response data is null?', response.data === null)
    
    if (response.data === null || response.data === undefined) {
      throw new Error('El servidor devolvió una respuesta vacía (null/undefined)')
    }
    
    return response.data
  } catch (error) {
    console.error('❌ Error en trainModel API:', error)
    console.error('❌ Error response:', error.response)
    console.error('❌ Error status:', error.response?.status)
    console.error('❌ Error data:', error.response?.data)
    throw error
  }
}

export const getModels = async () => {
  const response = await axios.get(`${API_BASE}/models`)
  return response.data
}

export const getModelDetails = async (modelId) => {
  const response = await axios.get(`${API_BASE}/models/${modelId}`)
  return response.data
}

export const predictWithModel = async (modelId, features) => {
  const response = await axios.post(`${API_BASE}/model/predict`, {
    model_id: modelId,
    features: features
  })
  return response.data
}

export const getStatistics = async (sessionId) => {
  const response = await axios.get(`${API_BASE}/data/${sessionId}/statistics`)
  return response.data
}

export const getCorrelations = async (sessionId, targetColumn = null) => {
  const url = targetColumn 
    ? `${API_BASE}/data/${sessionId}/correlations?target_column=${targetColumn}`
    : `${API_BASE}/data/${sessionId}/correlations`
  const response = await axios.get(url)
  return response.data
}

export const predictBatchWithModel = async (modelId, file) => {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('model_id', modelId)
  const response = await axios.post(`${API_BASE}/model/predict/batch`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
  return response.data
}

