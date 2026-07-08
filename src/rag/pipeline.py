from .retriever import Retriever
from .generator import get_generator
from .evaluator import Evaluator
from .schemas import RAGResult, GenerationResult


class RAGPipeline:
    def run(
        self,
        query: str,
        documents: list[str],
        ground_truth: str,
        dataset_name: str,
        embedding_model: str,
        top_k: int = 5,
        llm_model: str | None = None,
        on_stage: callable = None,
    ) -> RAGResult:
        if on_stage:
            on_stage("Loading model...")
        retrieval = Retriever.retrieve(query, documents, embedding_model, top_k, dataset_name)
        if on_stage:
            on_stage("Generating answer...")
        generator = get_generator(llm_model)
        answer = generator.generate(query, retrieval.documents)
        if on_stage:
            on_stage("Evaluating...")
        evaluation = self.evaluator.evaluate(
            query, answer, retrieval.documents, ground_truth, document_ids=retrieval.document_ids,
        )
        return RAGResult(
            query=query,
            ground_truth=ground_truth,
            dataset_used=dataset_name,
            retrieval=retrieval,
            generation=GenerationResult(
                answer=answer,
                model_name=embedding_model,
                retrieval_model=embedding_model,
                retrieved_documents=retrieval.documents,
            ),
            evaluation=evaluation,
        )

    @property
    def evaluator(self):
        from .evaluator import Evaluator
        return Evaluator()
