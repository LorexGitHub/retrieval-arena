"""LCEL RAG pipeline demonstrating LangChain Expression Language integration.

This module shows how the existing RAG pipeline can be expressed
using LangChain's composable chain syntax. The subprocess isolation
for model memory management is preserved.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.vectorstores import InMemoryVectorStore


RAG_PROMPT = PromptTemplate.from_template(
    "Answer based on this context: {context}\n\nQuestion: {question}\nAnswer:"
)


def create_retriever(model_name: str, documents: list[str]):
    """Create a LangChain retriever from documents."""
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = InMemoryVectorStore.from_texts(documents, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})


def create_llm(model_id: str):
    """Create a LangChain LLM wrapper around a HuggingFace pipeline."""
    from transformers import pipeline as hf_pipeline

    kwargs = {
        "model": model_id,
        "dtype": "auto",
        "model_kwargs": {"low_cpu_mem_usage": True},
    }
    try:
        import accelerate
        kwargs["device_map"] = "auto"
    except ImportError:
        pass

    pipe = hf_pipeline("text-generation", **kwargs)
    return HuggingFacePipeline(pipeline=pipe, return_full_text=False)


def create_rag_chain(llm, retriever):
    """Build an LCEL RAG chain.

    Usage:
        chain = create_rag_chain(llm, retriever)
        result = chain.invoke("What is Python?")
    """
    def format_docs(docs):
        return ", ".join(d.doc.page_content for d in docs)

    chain = (
        RunnableParallel(
            context=(retriever | format_docs),
            question=RunnablePassthrough(),
        )
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain
