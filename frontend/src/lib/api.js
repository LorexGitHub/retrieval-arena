const API_BASE = "/api"

export async function fetchDatasets() {
  try {
    const res = await fetch(`${API_BASE}/datasets`)
    if (!res.ok) return []
    const data = await res.json()
    return data.available_datasets ?? []
  } catch {
    return []
  }
}

export async function fetchDatasetDocuments(name) {
  try {
    const res = await fetch(`${API_BASE}/datasets/${encodeURIComponent(name)}/documents`)
    if (!res.ok) return []
    const data = await res.json()
    return data.documents ?? []
  } catch {
    return []
  }
}

export async function createDataset(name, documents) {
  try {
    await fetch(`${API_BASE}/datasets/${encodeURIComponent(name)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ documents }),
    })
  } catch {
    // silent
  }
}

export async function deleteDataset(name) {
  try {
    await fetch(`${API_BASE}/datasets/${encodeURIComponent(name)}`, {
      method: "DELETE",
    })
  } catch {
    // silent
  }
}

export async function fetchQueries() {
  try {
    const res = await fetch(`${API_BASE}/queries`)
    if (!res.ok) return []
    const data = await res.json()
    return data.queries ?? []
  } catch {
    return []
  }
}

export async function fetchLLMs() {
  try {
    const res = await fetch(`${API_BASE}/llms`)
    if (!res.ok) return []
    const data = await res.json()
    return data.available_llms ?? []
  } catch {
    return []
  }
}

export async function ingestUrl(url, name, chunkSize = 500, chunkOverlap = 50) {
  try {
    const res = await fetch(`${API_BASE}/ingest-url`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, name, chunk_size: chunkSize, chunk_overlap: chunkOverlap }),
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err.detail || `HTTP ${res.status}`)
    }
    return await res.json()
  } catch (e) {
    throw e
  }
}

export function getCompareURL() {
  return `${API_BASE}/compare`
}

export function getCompareMultiURL() {
  return `${API_BASE}/compare-multi`
}
