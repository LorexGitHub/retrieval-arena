export const EMBEDDING_MODELS = {
  "minilm-l12": { model_name: "sentence-transformers/all-MiniLM-L12-v2", size: "33M", memory: "~90 MB", speed: "fast" },
  "bge-small": { model_name: "BAAI/bge-small-en-v1.5", size: "33M", memory: "~90 MB", speed: "fast" },
  "gte-small": { model_name: "thenlper/gte-small", size: "33M", memory: "~100 MB", speed: "fast" },
  "granite": { model_name: "ibm-granite/granite-embedding-small-english-r2", size: "102M", memory: "~200 MB", speed: "fast" },
  "harrier": { model_name: "microsoft/harrier-oss-v1-270m", size: "270M", memory: "~550 MB", speed: "fast" },
  "bge-base": { model_name: "BAAI/bge-base-en-v1.5", size: "110M", memory: "~250 MB", speed: "medium" },
  "mpnet": { model_name: "sentence-transformers/all-mpnet-base-v2", size: "110M", memory: "~250 MB", speed: "medium" },
  "qwen3": { model_name: "Qwen/Qwen3-Embedding-0.6B", size: "600M", memory: "~1.2 GB", speed: "medium" },
  "jina": { model_name: "jinaai/jina-embeddings-v5-text-small", size: "580M", memory: "~1.2 GB", speed: "medium" },
  "bge-large": { model_name: "BAAI/bge-large-en-v1.5", size: "335M", memory: "~700 MB", speed: "slow" },
}

export const MODEL_KEYS = Object.keys(EMBEDDING_MODELS)

export const LLM_MODELS = {
  smol: { model_name: "HuggingFaceTB/SmolLM2-135M-Instruct", size: "135M", memory: "~270 MB", speed: "fast" },
  smol360: { model_name: "HuggingFaceTB/SmolLM2-360M-Instruct", size: "360M", memory: "~720 MB", speed: "fast" },
  qwen2: { model_name: "Qwen/Qwen2.5-0.5B-Instruct", size: "0.5B", memory: "~1 GB", speed: "fast" },
  qwen1_5: { model_name: "Qwen/Qwen2.5-1.5B-Instruct", size: "1.5B", memory: "~3 GB", speed: "medium" },
  tinyllama: { model_name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0", size: "1.1B", memory: "~2.5 GB", speed: "medium" },
  glm_edge: { model_name: "THUDM/glm-edge-1.5b-chat", size: "1.5B", memory: "~3 GB", speed: "medium" },
  glm4: { model_name: "THUDM/glm-4-9b-chat", size: "9B", memory: "~18 GB", speed: "slow" },
}
