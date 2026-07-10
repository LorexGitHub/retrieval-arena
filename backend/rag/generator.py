import os
import json
import urllib.request
import urllib.error
import multiprocessing as mp

from .config import LLM_MODELS, DEFAULT_LLM, LLM_BASE_URL, LLM_API_KEY, GENERATOR


class Generator:
    def generate(self, query: str, context: list[str]) -> str:
        raise NotImplementedError


class _TemplateGenerator(Generator):
    def generate(self, query: str, context: list[str]) -> str:
        if not context:
            return "No relevant information found."
        return context[0]


class _OpenAIGenerator(Generator):
    def __init__(self, base_url: str, model: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def generate(self, query: str, context: list[str]) -> str:
        context_str = ", ".join(context)
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": f"Answer based only on this context: {context_str}"},
                {"role": "user", "content": query},
            ],
            "temperature": GENERATOR["temperature"],
            "max_tokens": GENERATOR["max_tokens"],
            "stream": False,
        }).encode()

        base = self.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        url = f"{base}/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(url, data=payload, headers=headers)
        try:
            resp = urllib.request.urlopen(req, timeout=120)
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            raise RuntimeError(f"API error {e.code}: {body}")
        data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()


def _llm_worker(model_id: str, query: str, context: list[str], result_queue: mp.Queue):
    """Load model, generate, and return result. Runs in child process so memory is freed on exit."""
    try:
        from transformers import pipeline as hf_pipeline
        kwargs = {
            "model": model_id,
            "dtype": "auto",
            "model_kwargs": {"low_cpu_mem_usage": True},
        }
        try:
            import accelerate  # noqa: F401
            kwargs["device_map"] = "auto"
        except ImportError:
            pass
        pipe = hf_pipeline("text-generation", **kwargs)
        context_str = ", ".join(context)

        if pipe.tokenizer.chat_template:
            messages = [
                {"role": "system", "content": f"You have access to this context: {context_str}. Answer concisely."},
                {"role": "user", "content": query},
            ]
            prompt = pipe.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = (
                f"Context: {context_str}\n"
                f"Question: {query}\n"
                f"Answer:"
            )

        result = pipe(
            prompt,
            max_new_tokens=64,
            temperature=0.1,
            do_sample=False,
            repetition_penalty=1.2,
            eos_token_id=pipe.tokenizer.eos_token_id,
            pad_token_id=pipe.tokenizer.pad_token_id or pipe.tokenizer.eos_token_id,
        )
        text = result[0]["generated_text"][len(prompt):].strip()
        result_queue.put({"answer": text.split("\n")[0][:200]})
    except Exception as e:
        result_queue.put({"error": str(e)})


class _LocalLLM(Generator):
    """Load any HuggingFace causal LM via subprocess (frees memory after each call)."""

    def __init__(self, model_id: str):
        self.model_id = model_id

    def generate(self, query: str, context: list[str]) -> str:
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(
            target=_llm_worker,
            args=(self.model_id, query, context, q),
        )
        p.start()
        try:
            data = q.get(timeout=600)
        except Exception:
            p.terminate()
            p.join(timeout=10)
            raise RuntimeError("LLM subprocess timed out")
        p.join(timeout=10)

        if "error" in data:
            raise RuntimeError(data["error"])
        return data["answer"]


class _OllamaGenerator(Generator):
    def __init__(self):
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags")
            resp = urllib.request.urlopen(req)
            models = json.loads(resp.read()).get("models", [])
            if not models:
                raise RuntimeError("Ollama is running but no models installed.")
            self.model = models[0]["name"]
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Ollama: {e}")

    def generate(self, query: str, context: list[str]) -> str:
        prompt = (
            f"Context: {', '.join(context)}\n\n"
            f"Question: {query}\n\n"
            f"Answer concisely based only on the context above:"
        )
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": GENERATOR["temperature"],
                "num_predict": GENERATOR["max_tokens"],
            },
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=120)
        data = json.loads(resp.read())
        return data.get("response", "").strip()


def get_generator(llm_model: str | None = None) -> Generator:
    """Create a generator for the given LLM model key.
    
    - llm_model == ""  → TemplateGenerator (user explicitly chose no LLM)
    - llm_model is a key in LLM_MODELS → that HuggingFace model
    - llm_model is None → fall back to environment-based configuration
    """
    if llm_model == "":
        return _TemplateGenerator()
    if llm_model and llm_model in LLM_MODELS:
        model_id = LLM_MODELS[llm_model]["model_name"]
        return _LocalLLM(model_id)

    model_key_or_path = DEFAULT_LLM or os.getenv("LOCAL_LLM_MODEL", "")
    base_url = LLM_BASE_URL

    if base_url:
        model = model_key_or_path or "local-model"
        return _OpenAIGenerator(base_url, model, LLM_API_KEY)
    if model_key_or_path:
        entry = LLM_MODELS.get(model_key_or_path)
        model_id = entry["model_name"] if isinstance(entry, dict) else model_key_or_path
        return _LocalLLM(model_id)
    if os.getenv("LLM_USE_OLLAMA") == "1":
        return _OllamaGenerator()
    return _TemplateGenerator()
