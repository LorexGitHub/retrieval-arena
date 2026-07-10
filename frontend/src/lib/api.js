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

export function getCompareURL() {
  return `${API_BASE}/compare`
}

export function getCompareMultiURL() {
  return `${API_BASE}/compare-multi`
}
