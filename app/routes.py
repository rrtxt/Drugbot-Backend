from flask import Blueprint, request, current_app
from app.chatbot import Retriever, Generator
from app.db import VectorStoreSingleton
from app.llm import LLMPipelineSingleton, CrossRerankerSingleton
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from uuid import uuid4

main = Blueprint('main', __name__)

@main.route("/", methods=["GET"])
def home():
    return "ini response"

@main.route("/health", methods=["GET"])
def health_check():
    return {"status" : "ok"}
    
@main.route("/chat", methods=["GET"])
def chat_get():
    session_id = request.query.get("session_id")
    if not session_id:
        return {"error" : "Session ID is required"}, 400
    conn_str = current_app.config["MONGODB_URI"]
    chat_history = MongoDBChatMessageHistory(
        connection_string=conn_str,
        session_id=session_id,
        collection_name="chat_history",
        database_name=current_app.config["MONGODB_DBNAME"],
    )

    messages = chat_history.messages

    history = [{"role": msg.type, "content": msg.content, "created_at": msg.additional_kwargs.get("created_at")} for msg in messages]

    return history

@main.route("/chat", methods=["POST"])
def chat():
    query = request.body.get("query")
    session_id = request.query.get("session_id") if request.query.get("session_id") else str(uuid4())
    is_using_rag = request.query.get("is_using_rag", "false").lower() == "true"

    if not query:
        return {"error" : "Query is required"}, 400 

    if is_using_rag:
        chroma = VectorStoreSingleton.get_vector_store()
        reranker = CrossRerankerSingleton.get_reranker()
        retriever = Retriever(chroma, 10, reranker) 
        docs = retriever.get_relevant_docs(query)
        template = """Anda adalah seorang asisten medis yang ahli dalam memberikan rekomandasi obat.
        Berdasarkan informasi-informasi obat yang telah disediakan berikan rekomendasi obat untuk pertanyaan yang diberikan.
        Rekomendasi obat yang diberikan harus berisi informasi terkait nama obat, deskripsi kegunaan obat, dosis, dan efek samping obat.
        Berikan rekomendasi dengan maksimum 2 paragraf.
        Jika kamu tidak mengetahui terkait informasi obat yang diberikan, maka kamu cukup bilang tidak mengetahuinya.
        Terakhir, tolong kasih tahu ke pengguna bahwa jika dalam waktu 3 hari pengguna masih mengalami gejala yang dialami, maka segera konsultasi ke dokter.

        Konteks: {context}
        """
    else:
        docs = []
        template = """Anda adalah seorang asisten medis yang ahli dalam memberikan rekomandasi obat.
        Rekomendasi obat yang diberikan harus berisi informasi terkait nama obat, deskripsi kegunaan obat, dosis, dan efek samping obat.
        Berikan rekomendasi dengan maksimum 2 paragraf.
        Jika kamu tidak mengetahui terkait informasi obat yang diberikan, maka kamu cukup bilang tidak mengetahuinya.
        Terakhir, tolong kasih tahu ke pengguna bahwa jika dalam waktu 3 hari pengguna masih mengalami gejala yang dialami, maka segera konsultasi ke dokter.

        Konteks: {context}
        """

    llm = LLMPipelineSingleton.get_pipeline()
    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="history"),
        ("user", MessagesPlaceholder(variable_name="query"))
    ])

    generator = Generator(llm, prompt)
    conn_str = current_app.config["MONGODB_URI"]
    db_name = current_app.config["MONGODB_DBNAME"]

    response = generator.generate(query, docs, session_id, conn_str, db_name)
    if isinstance(response, dict):
        response["session_id"] = session_id
    else:
        response = {"response": response, "session_id": session_id}

    return response