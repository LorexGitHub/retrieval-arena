import { useState, useEffect, useCallback } from "react"
import { fetchDatasets, fetchDatasetDocuments, createDataset, deleteDataset } from "@/lib/api"

function extractId(text) {
  const trimmed = text.trim()
  const dot = trimmed.indexOf(".")
  return dot > 0 ? trimmed.slice(0, dot).trim() : trimmed.split(",")[0].trim().slice(0, 60)
}

export function useDatasets() {
  const [datasets, setDatasets] = useState([])
  const [selectedDataset, setSelectedDataset] = useState("")
  const [editText, setEditText] = useState("")

  const refreshDatasets = useCallback(async () => {
    const list = await fetchDatasets()
    setDatasets(list)
    return list
  }, [])

  const refresh = useCallback(async () => {
    const list = await refreshDatasets()
    if (list.length > 0 && !list.includes(selectedDataset)) {
      setSelectedDataset(list[0])
    }
  }, [selectedDataset, refreshDatasets])

  useEffect(() => {
    refresh()
  }, [refresh])

  useEffect(() => {
    if (selectedDataset) {
      fetchDatasetDocuments(selectedDataset).then((docs) => {
        if (docs.length > 0 && typeof docs[0] === "object") {
          setEditText(docs.map((d) => d.text).join("\n"))
        } else {
          setEditText(docs.join("\n"))
        }
      })
    }
  }, [selectedDataset])

  const handleSave = useCallback(async () => {
    const lines = editText.split("\n").filter((d) => d.trim())
    const docs = lines.map((line) => ({ id: extractId(line), text: line }))
    await createDataset(selectedDataset, docs)
    await refresh()
  }, [selectedDataset, editText, refresh])

  const handleDelete = useCallback(async () => {
    await deleteDataset(selectedDataset)
    setSelectedDataset("")
    await refresh()
  }, [selectedDataset, refresh])

  return {
    datasets,
    selectedDataset,
    setSelectedDataset,
    editText,
    setEditText,
    handleSave,
    handleDelete,
    refreshDatasets,
  }
}
