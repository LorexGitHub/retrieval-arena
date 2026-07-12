import math
import re

from sentence_transformers import SentenceTransformer, util
from .config import EVAL_SEMANTIC_MODEL, EVAL_JUDGE_MODEL
from .schemas import EvaluationMetrics
from .generator import get_generator, _TemplateGenerator, _LocalLLM

# Module-level caches survive across Evaluator instances
_SEMANTIC_MODEL_CACHE: dict[str, SentenceTransformer] = {}
_JUDGE_CACHE: dict[str, _LocalLLM | None] = {}


class Evaluator:
    def _get_semantic_model(self):
        key = EVAL_SEMANTIC_MODEL
        if key not in _SEMANTIC_MODEL_CACHE:
            _SEMANTIC_MODEL_CACHE[key] = SentenceTransformer(EVAL_SEMANTIC_MODEL)
        return _SEMANTIC_MODEL_CACHE[key]

    def _get_judge(self):
        key = EVAL_JUDGE_MODEL
        if key in _JUDGE_CACHE:
            return _JUDGE_CACHE[key]
        gen = get_generator()
        if isinstance(gen, _TemplateGenerator):
            from .config import LLM_MODELS
            if EVAL_JUDGE_MODEL in LLM_MODELS:
                try:
                    model_id = LLM_MODELS[EVAL_JUDGE_MODEL]["model_name"]
                    gen = _LocalLLM(model_id)
                except Exception:
                    _JUDGE_CACHE[key] = None
                    return None
            else:
                _JUDGE_CACHE[key] = None
                return None
        _JUDGE_CACHE[key] = gen
        return _JUDGE_CACHE[key]

    # --- Retrieval metrics ---

    def _match_ids(self, ids: list[str], ground_truth: str) -> bool:
        return any(ground_truth.lower() == i.lower() for i in ids)

    def hit_rate(self, documents: list[str], ground_truth: str, document_ids: list[str] | None = None) -> float:
        ids = document_ids if document_ids else documents
        return 1.0 if self._match_ids(ids, ground_truth) else 0.0

    def mrr(self, documents: list[str], ground_truth: str, document_ids: list[str] | None = None) -> float:
        ids = document_ids if document_ids else documents
        for i, doc_id in enumerate(ids):
            if ground_truth.lower() == doc_id.lower():
                return 1.0 / (i + 1)
        return 0.0

    def precision(self, documents: list[str], ground_truth: str, document_ids: list[str] | None = None) -> float:
        if not documents:
            return 0.0
        ids = document_ids if document_ids else documents
        return 1.0 / len(documents) if self._match_ids(ids, ground_truth) else 0.0

    def ndcg(self, documents: list[str], ground_truth: str, document_ids: list[str] | None = None) -> float:
        ids = document_ids if document_ids else documents
        dcg = 0.0
        for i, doc_id in enumerate(ids):
            rel = 1.0 if ground_truth.lower() == doc_id.lower() else 0.0
            dcg += rel / math.log2(i + 2)
        if dcg == 0.0:
            return 0.0
        idcg = 1.0 / math.log2(2)
        return dcg / idcg

    # --- Generation metrics ---

    def rouge_l_f1(self, generated: str, reference: str) -> float:
        gen_tokens = generated.lower().split()
        ref_tokens = reference.lower().split()
        if not gen_tokens or not ref_tokens:
            return 0.0
        m = len(ref_tokens)
        n = len(gen_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i - 1] == gen_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs = dp[m][n]
        precision = lcs / n if n > 0 else 0.0
        recall = lcs / m if m > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    def semantic_similarity(self, generated: str, reference: str) -> float:
        model = self._get_semantic_model()
        emb_gen = model.encode(generated, convert_to_tensor=True)
        emb_ref = model.encode(reference, convert_to_tensor=True)
        return float(util.cos_sim(emb_gen, emb_ref)[0][0])

    def _extract_number(self, text: str) -> float | None:
        """Extract the first floating-point number from text."""
        nums = re.findall(r"(\d+\.?\d*)", text)
        if nums:
            try:
                return float(nums[0])
            except (ValueError, TypeError):
                return None
        return None

    def faithfulness(self, answer: str, context: list[str]) -> float | None:
        judge = self._get_judge()
        if not judge:
            return None
        try:
            prompt = (
                "You are an evaluation assistant. Decompose the following Answer into individual factual claims. "
                "For each claim, determine if it is supported by the provided Context. "
                "Return ONLY a single number between 0 and 1 representing "
                "the fraction of claims that are supported by the context.\n\n"
                f"Context: {' '.join(context)}\n"
                f"Answer: {answer}\n\n"
                "Fraction of supported claims:"
            )
            result = judge.generate(prompt, [])
            num = self._extract_number(result)
            if num is not None:
                return min(max(num, 0.0), 1.0)
            return None
        except Exception:
            return None

    def answer_relevancy(self, query: str, answer: str) -> float:
        model = self._get_semantic_model()
        emb_q = model.encode(query, convert_to_tensor=True)
        emb_a = model.encode(answer, convert_to_tensor=True)
        return float(util.cos_sim(emb_q, emb_a)[0][0])

    def llm_quality_score(self, query: str, answer: str, context: list[str]) -> float | None:
        judge = self._get_judge()
        if not judge:
            return None
        try:
            text = judge.generate(
                f"Rate the following answer on a scale of 1-5 (one decimal allowed) "
                f"based on how well it answers the question using only the provided context.\n\n"
                f"Question: {query}\n"
                f"Context: {', '.join(context)}\n"
                f"Answer: {answer}\n\n"
                f"Only output a number between 1 and 5:",
                [],
            )
            num = self._extract_number(text)
            if num is not None:
                return min(max(num, 1.0), 5.0)
            return None
        except Exception:
            return None

    def evaluate(
        self,
        query: str,
        generated_answer: str,
        retrieved_context: list[str],
        ground_truth: str,
        document_ids: list[str] | None = None,
    ) -> EvaluationMetrics:
        return EvaluationMetrics(
            hit_rate=self.hit_rate(retrieved_context, ground_truth, document_ids),
            mrr=self.mrr(retrieved_context, ground_truth, document_ids),
            precision=self.precision(retrieved_context, ground_truth, document_ids),
            ndcg=self.ndcg(retrieved_context, ground_truth, document_ids),
            exact_match=ground_truth.lower() in generated_answer.lower(),
            rouge_l_f1=self.rouge_l_f1(generated_answer, ground_truth),
            semantic_similarity=self.semantic_similarity(generated_answer, ground_truth),
            faithfulness=self.faithfulness(generated_answer, retrieved_context),
            answer_relevancy=self.answer_relevancy(query, generated_answer),
            llm_quality_score=self.llm_quality_score(query, generated_answer, retrieved_context),
        )
