import { useState, useEffect, useCallback, useRef } from "react"
import { HeroBackground } from "@/components/layout/HeroBackground"
import { Sidebar } from "@/components/layout/Sidebar"
import { RunPanel } from "@/components/benchmark/RunPanel"
import { ResultList } from "@/components/benchmark/ResultList"
import { useDatasets } from "@/hooks/useDatasets"
import { getCompareMultiURL, fetchQueries } from "@/lib/api"
import { createDataset } from "@/lib/api"
import type { QueryItem, CompareReport } from "@/types"

export default function App() {
  const {
    datasets,
    selectedDataset,
    setSelectedDataset,
    editText,
    setEditText,
    handleSave,
    handleDelete,
    refreshDatasets,
  } = useDatasets()

  const [queries, setQueries] = useState<QueryItem[]>([])
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([])
  const [reports, setReports] = useState<CompareReport[]>([])
  const [lastLlmModel, setLastLlmModel] = useState("")
  const [progress, setProgress] = useState<string | null>(null)
  const [progressPercent, setProgressPercent] = useState(0)
  const [loading, setLoading] = useState(false)
  const [abortController, setAbortController] = useState<AbortController | null>(null)
  const totalStepsRef = useRef(0)
  const processedRef = useRef(0)

  useEffect(() => {
    fetchQueries().then(setQueries)
  }, [])

  useEffect(() => {
    if (datasets.length > 0) {
      setSelectedDatasets(datasets.slice(0, 2))
    }
  }, [datasets])

  const handleAddDataset = useCallback(async (name: string, documents: string[]) => {
    await createDataset(name, documents)
    await refreshDatasets()
  }, [refreshDatasets])

  const handleRun = useCallback(async (
    items: { query: string; ground_truth: string }[],
    llmModel: string,
    models: string[],
  ) => {
    if (selectedDatasets.length === 0 || items.length === 0) return

    setReports([])
    setProgress(null)
    setProgressPercent(0)
    setLastLlmModel(llmModel)
    setLoading(true)

    totalStepsRef.current = items.length * models.length
    processedRef.current = 0
    const abort = new AbortController()
    setAbortController(abort)

    const dataset = selectedDatasets[0]
    const url = getCompareMultiURL()
    const body = {
      queries: items,
      dataset_name: dataset,
      embedding_models: models,
      llm_model: llmModel,
      top_k: 5,
    }

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: abort.signal,
      })

      if (!response.ok) throw new Error(`HTTP ${response.status}`)

      const reader = response.body?.getReader()
      if (!reader) throw new Error("No response body")
      const decoder = new TextDecoder()
      let buffer = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() || ""

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const event = JSON.parse(line.slice(6))
              if (event.type === "stage") {
                setProgress(event.message)
                processedRef.current += 1
                const pct = Math.min(95, Math.round((processedRef.current / totalStepsRef.current) * 100))
                setProgressPercent(pct)
              } else if (event.type === "result") {
                setReports(event.result.reports ?? [])
                setProgressPercent(100)
              } else if (event.type === "error") {
                console.error("SSE error:", event.message)
              }
            } catch {
              // skip malformed lines
            }
          }
        }
      }
    } catch (err: any) {
      if (err.name !== "AbortError") {
        console.error("Fetch error:", err)
      }
    } finally {
      setLoading(false)
      setAbortController(null)
    }
  }, [selectedDatasets])

  const handleCancel = useCallback(() => {
    abortController?.abort()
  }, [abortController])

  const handleClear = useCallback(() => {
    setReports([])
    setProgress(null)
    setProgressPercent(0)
  }, [])

  return (
    <>
      <HeroBackground />

      <div className="h-screen flex overflow-hidden">
        <aside className="w-72 shrink-0 border-r border-border bg-card/50">
          <Sidebar
            datasets={datasets}
            selectedDataset={selectedDataset}
            onDatasetChange={setSelectedDataset}
            editText={editText}
            onEditTextChange={setEditText}
            onSave={handleSave}
            onDelete={handleDelete}
            onAddDataset={handleAddDataset}
          />
        </aside>

        <main className="flex-1 min-w-0 overflow-y-auto">
          <div className="h-[50vh] bg-gradient-to-b from-transparent to-bg" />

          <div className="max-w-[920px] mx-auto px-6 pb-12">
            <div className="animate-fade-up" style={{ animationDelay: "0.1s" }}>
              <div className="text-sm font-semibold tracking-tight text-text pt-3 pb-0.5">Benchmark</div>
              <div className="text-[0.68rem] text-text-faint pb-2">
                10 models &middot; {datasets.length} datasets &middot; {queries.length} queries
              </div>
            </div>

            {datasets.length > 0 && (
              <div className="flex flex-wrap gap-2 mb-4 animate-fade-up" style={{ animationDelay: "0.15s" }}>
                {datasets.map((ds) => (
                  <button
                    key={ds}
                    onClick={() =>
                      setSelectedDatasets((prev) =>
                        prev.includes(ds) ? prev.filter((d) => d !== ds) : [...prev, ds]
                      )
                    }
                    className={`px-3 py-1.5 rounded-[8px] text-xs font-medium border transition-all cursor-pointer ${
                      selectedDatasets.includes(ds)
                        ? "bg-accent/10 border-accent/30 text-accent"
                        : "bg-surface border-border text-text-sec hover:border-border-h"
                    }`}
                  >
                    {ds}
                  </button>
                ))}
              </div>
            )}

            <div className="animate-fade-up" style={{ animationDelay: "0.2s" }}>
              <RunPanel
                queries={queries}
                selectedDatasets={selectedDatasets}
                onRun={handleRun}
                onClear={handleClear}
                progress={progress}
                progressPercent={progressPercent}
                loading={loading}
                onCancel={handleCancel}
              />
            </div>

            {reports.length > 0 && (
              <div className="mt-8">
                <ResultList reports={reports} topK={5} llmModel={lastLlmModel} />
              </div>
            )}

            <div className="text-center pt-6 pb-2 text-[0.68rem] text-text-faint">
              retrieval-arena
            </div>
          </div>
        </main>
      </div>
    </>
  )
}
