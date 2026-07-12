import { useState } from "react"
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"

export function Sidebar({ datasets, selectedDataset, onDatasetChange, editText, onEditTextChange, onSave, onDelete, onAddDataset, onIngestUrl }) {
  const [adding, setAdding] = useState(false)
  const [ingesting, setIngesting] = useState(false)
  const [newName, setNewName] = useState("")
  const [newDocs, setNewDocs] = useState("")
  const [url, setUrl] = useState("")
  const [urlName, setUrlName] = useState("")
  const [chunkSize, setChunkSize] = useState("500")

  const handleAdd = async () => {
    if (!newName.trim()) return
    await onAddDataset(newName.trim(), newDocs.split("\n").filter((d) => d.trim()))
    setAdding(false)
    setNewName("")
    setNewDocs("")
  }

  const handleIngest = async () => {
    if (!url.trim() || !urlName.trim() || ingesting) return
    setIngesting(true)
    try {
      await onIngestUrl(url.trim(), urlName.trim(), parseInt(chunkSize, 10) || 500)
      setUrl("")
      setUrlName("")
      setIngesting(false)
      setAdding(false)
    } catch {
      setIngesting(false)
    }
  }

  return (
    <div className="h-full flex flex-col p-4 gap-3">
      <div className="flex items-center justify-between">
        <h3 className="text-[0.7rem] font-semibold uppercase tracking-wider text-text-faint m-0">Edit Dataset</h3>
        <div className="flex gap-1">
          <button
            onClick={() => { setAdding(!adding); setIngesting(false) }}
            className="text-xs text-accent hover:text-accent-h cursor-pointer bg-transparent border-none"
          >
            + New
          </button>
          <span className="text-text-faint">·</span>
          <button
            onClick={() => { setIngesting(!ingesting); setAdding(false) }}
            className="text-xs text-accent hover:text-accent-h cursor-pointer bg-transparent border-none"
          >
            URL
          </button>
        </div>
      </div>

      {adding && (
        <div className="flex flex-col gap-2 p-3 rounded-[10px] border border-border bg-surface">
          <input
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder="Dataset name"
            className="h-[34px] rounded-[8px] border border-border bg-bg px-2.5 text-xs text-text placeholder:text-text-faint caret-accent focus:outline-none focus:border-accent"
          />
          <Textarea
            value={newDocs}
            onChange={(e) => setNewDocs(e.target.value)}
            placeholder="Documents (one per line)"
            className="min-h-[100px] text-xs"
          />
          <Button variant="primary" size="sm" onClick={handleAdd}>Create</Button>
        </div>
      )}

      {ingesting && (
        <div className="flex flex-col gap-2 p-3 rounded-[10px] border border-border bg-surface">
          <input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://example.com/page"
            className="h-[34px] rounded-[8px] border border-border bg-bg px-2.5 text-xs text-text placeholder:text-text-faint caret-accent focus:outline-none focus:border-accent"
          />
          <input
            value={urlName}
            onChange={(e) => setUrlName(e.target.value)}
            placeholder="Dataset name"
            className="h-[34px] rounded-[8px] border border-border bg-bg px-2.5 text-xs text-text placeholder:text-text-faint caret-accent focus:outline-none focus:border-accent"
          />
          <input
            value={chunkSize}
            onChange={(e) => setChunkSize(e.target.value)}
            placeholder="Chunk size"
            type="number"
            min="100"
            max="5000"
            className="h-[34px] rounded-[8px] border border-border bg-bg px-2.5 text-xs text-text placeholder:text-text-faint caret-accent focus:outline-none focus:border-accent"
          />
          <Button variant="primary" size="sm" onClick={handleIngest} disabled={ingesting}>
            {ingesting ? "Fetching..." : "Ingest"}
          </Button>
        </div>
      )}

      <Select value={selectedDataset} onValueChange={onDatasetChange}>
        <SelectTrigger>
          <SelectValue placeholder="Select dataset" />
        </SelectTrigger>
        <SelectContent>
          {datasets.map((ds) => (
            <SelectItem key={ds} value={ds}>{ds}</SelectItem>
          ))}
        </SelectContent>
      </Select>

      <Textarea
        value={editText}
        onChange={(e) => onEditTextChange(e.target.value)}
        placeholder="Documents (one per line)"
        className="flex-1 min-h-0"
      />

      <div className="flex gap-2">
        <Button variant="primary" onClick={onSave} className="flex-1">Save</Button>
        <Button variant="secondary" onClick={onDelete} className="flex-1">Delete</Button>
      </div>
    </div>
  )
}
