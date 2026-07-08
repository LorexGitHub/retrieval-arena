const API_BASE = "/api"

export async function fetchDatasets(): Promise<string[]> {
  try {
    const res = await fetch(`${API_BASE}/datasets`)
    if (!res.ok) return []
    const data = await res.json()
    return data.available_datasets ?? []
  } catch {
    return []
  }
}

export async function fetchDatasetDocuments(name: string): Promise<string[] | Record<string, string>[]> {
  try {
    const res = await fetch(`${API_BASE}/datasets/${encodeURIComponent(name)}/documents`)
    if (!res.ok) return []
    const data = await res.json()
    return data.documents ?? []
  } catch {
    return []
  }
}

export async function createDataset(name: string, documents: string[] | Record<string, string>[]): Promise<void> {
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

export async function deleteDataset(name: string): Promise<void> {
  try {
    await fetch(`${API_BASE}/datasets/${encodeURIComponent(name)}`, {
      method: "DELETE",
    })
  } catch {
    // silent
  }
}

export async function fetchQueries(): Promise<{ query: string; ground_truth: string; relevant_dataset: string }[]> {
  try {
    const res = await fetch(`${API_BASE}/queries`)
    if (!res.ok) return []
    const data = await res.json()
    return data.queries ?? []
  } catch {
    return []
  }
}

export async function fetchLLMs(): Promise<{ key: string; model_name: string; size: string; memory: string; speed: string }[]> {
  try {
    const res = await fetch(`${API_BASE}/llms`)
    if (!res.ok) return []
    const data = await res.json()
    return data.available_llms ?? []
  } catch {
    return []
  }
}

export function getCompareURL(): string {
  return `${API_BASE}/compare`
}

export function getCompareMultiURL(): string {
  return `${API_BASE}/compare-multi`
}
