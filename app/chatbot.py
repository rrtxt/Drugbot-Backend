from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker 

from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface.chat_models import ChatHuggingFace

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from app.custom_history import CustomMongoDBChatMessageHistory

from typing_extensions import List, TypedDict
from datetime import datetime, timezone

def get_bot_response():
    return "Ini respons"

class Retriever:
    def __init__(self, chroma : Chroma, k : int, reranker : CrossEncoderReranker):
        self.vector_store = chroma
        self.reranker = reranker
        self.retriever = self._initialize_retriever(k)

    def _initialize_retriever(self, k : int):
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k" : k}
        )
    
    def get_relevant_docs(self, query, filter=None):
        # Get similar documents based on query
        results = self.retriever.invoke(query, filter=filter)

        # Rerank the retrieved documents
        reranked_docs = self.reranker.compress_documents(results, query)
        
        # Get top 5 reranked documents
        top_docs = reranked_docs[:5]

        result_docs = [ f'Drugs {docs.metadata["drug_name"]}: {docs.page_content}' for docs in top_docs]

        return result_docs

class State(TypedDict):
    query: str
    context: List[Document]
    answer : str

class Generator:
    def __init__(self, llm : HuggingFacePipeline, template):
        self.llm = llm
        self.chat = ChatHuggingFace(llm=self.llm)
        if type(template) == str:
            self.prompt = PromptTemplate.from_template(template)
        else:
            self.prompt = template

    
    def _add_rag_info_to_ai_message(self, ai_message : BaseMessage, is_rag : bool):
        if hasattr(ai_message, "additional_kwargs"):
            ai_message.additional_kwargs["is_rag"] = is_rag
            ai_message.additional_kwargs["created_at"] = datetime.now(timezone.utc).isoformat()
        return {"answer" : ai_message}

    def generate(self, query, docs : List[Document], session_id, conn_str, db_name, skip_prompt : bool = True, tools : List[BaseTool] = None):
        docs_content = "\n\n".join(doc for doc in docs)
        chat_with_tools = self.chat.bind_tools(tools)
        is_rag = bool(docs_content.strip())
        
        base_chain = self.prompt | chat_with_tools.bind(skip_prompt=skip_prompt)
        chain_with_answer_key = base_chain | RunnableLambda(
            lambda ai_message : self._add_rag_info_to_ai_message(ai_message, is_rag)
        )

        chain_with_history =  RunnableWithMessageHistory(
            chain_with_answer_key,
            lambda session_id : CustomMongoDBChatMessageHistory(
                connection_string=conn_str,
                session_id=session_id, 
                collection_name="chat_history",
                database_name=db_name,
            ),
            input_messages_key="query",
            history_messages_key="history",
            output_messages_key="answer",
        )

        response = chain_with_history.invoke(
            {"query" : query, "context" : docs_content},
            {"configurable" : {"session_id" : session_id}}
        )

        return response
