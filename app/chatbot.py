from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker 
from app.error import DocsNotFoundError
from typing_extensions import List, TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline

def get_bot_response():
    return "Ini respons"

class Retriever:
    def __init__(self, chroma : Chroma, k : int, reranker : CrossEncoderReranker):
        self.vector_store = chroma
        self.reranker = reranker
        self.retriever = self._initialize_retriever()

    def _initialize_retriever(self):
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k" : self.k}
        )
    
    def get_relevant_docs(self, query, filter = {}):
        # Get similar documents based on query
        results = self.retriever.invoke(query, filter=filter)

        if len(results) <= 0:
            raise DocsNotFoundError("Similar document not found")

        # Rerank the retrieved documents
        reranked_docs = self.reranker.compress_documents(results)
        
        # Get top 5 reranked documents
        top_docs = reranked_docs[:5]

        return top_docs

class State(TypedDict):
    query: str
    context: List[Document]
    answer : str

class Generator:
    def __init__(self, llm : HuggingFacePipeline, template):
        self.llm = llm
        self.prompt = PromptTemplate.from_template(template) 

    def generate(self, query, docs : List[Document], skip_prompt : bool = False):
        docs_content = "\n\n".join(doc.page_content for doc in docs)
        
        chain = self.prompt | self.llm.bind(skip_prompt=skip_prompt)
        response = chain.invoke({"query" : query, "context" : docs_content})
        return response
