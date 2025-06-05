from flask import Blueprint, request, current_app
from app.chatbot import Retriever, Generator
from app.db import VectorStoreSingleton, MongoDBClientSingleton
from app.llm import LLMPipelineSingleton, CrossRerankerSingleton
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_tavily import TavilySearch
from uuid import uuid4
from markupsafe import escape

main = Blueprint('main', __name__)

@main.route("/", methods=["GET"])
def home():
    return "ini response"

@main.route("/health", methods=["GET"])
def health_check():
    return {"status" : "ok"}

@main.route("/chat/histories", methods=["GET"])
def chat_sessions():
    db_name = current_app.config["MONGODB_DBNAME"]
    collection_name = "chat_history"

    try:
        client = MongoDBClientSingleton.get_instance().get_client()
        db = client[db_name]
        collection = db[collection_name]
        session_ids = collection.distinct("SessionId")
        
        return {"session_ids": session_ids}
    except Exception as e:
        current_app.logger.error(f"Error fetching chat sessions: {e}")
        return {"error": "Failed to retrieve chat sessions"}, 500
    
@main.route("/chat/histories/<session_id>", methods=["GET"])
def chat_get(session_id):
    session_id = escape(session_id)
    if not session_id:
        return {"error": "Session ID is required"}, 400
    
    try:
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
    except Exception as e:
        current_app.logger.error(f"Error fetching chat history: {e}")
        return {"error": "Failed to retrieve chat history"}, 500

@main.route("/chat/histories/<session_id>", methods=["DELETE"])
def delete_chat_session(session_id):
    session_id = escape(session_id)
    if not session_id:
        return {"error": "Session ID is required"}, 400
    
    try:
        
        client = MongoDBClientSingleton.get_instance().get_client()
        db = client[current_app.config["MONGODB_DBNAME"]]
        collection = db["chat_history"]
        collection.delete_one({"SessionId": session_id})
        return {"message": "Chat session deleted successfully"}, 200
    
    except Exception as e:
        current_app.logger.error(f"Error deleting chat session: {e}")
        return {"error": "Failed to delete chat session"}, 500

@main.route("/chat", methods=["POST"])
def chat():
    try:
        query = request.get_json().get("query")
        session_id = request.get_json().get("session_id", str(uuid4()))
        is_using_rag = request.args.get("is_using_rag", "false").lower() == "true"

        if not query:
            return {"error" : "Query is required"}, 400 

        if is_using_rag:
            # Get pre-initialized instances
            vector_store = VectorStoreSingleton.get_instance().get_vector_store()
            reranker = CrossRerankerSingleton.get_instance().get_reranker()
            retriever = Retriever(vector_store, 10, reranker)

            current_app.logger.info(f"Successfully initialized RAG pipeline")  

            docs = retriever.get_relevant_docs(query)
            template = """Anda adalah seorang asisten medis yang ahli dalam memberikan rekomandasi obat. 
            Berikut adalah informasi tentang obat yang perlu direkomendasikan:
            
            {context}

            Berikan jawaban dalam bentuk paragraf yang mengandung informasi berikut:

            1. Nama obat yang direkomendasikan
            2. Deskripsi dan kegunaan obat tersebut
            3. Dosis yang disarankan
            4. Efek samping yang mungkin terjadi

            Dengan tools yang tersedia, cari informasi lebih lanjut tentang obat di Indonesia.
            Berikan jawaban dalam maksimal 2 paragraf dan akhiri dengan peringatan: "Jika dalam waktu 3 hari gejala masih berlanjut, segera konsultasi ke dokter."
            """
        else:
            docs = []
            template = """Anda adalah seorang asisten medis yang ahli dalam memberikan rekomandasi obat.
            Berikan jawaban dalam bentuk paragraf yang mengandung informasi berikut:

            1. Nama obat yang direkomendasikan
            2. Deskripsi dan kegunaan obat tersebut
            3. Dosis yang disarankan
            4. Efek samping yang mungkin terjadi

            Dengan tools yang tersedia, cari informasi lebih lanjut tentang obat di Indonesia
            Berikan jawaban dalam maksimal 2 paragraf dan akhiri dengan peringatan: "Jika dalam waktu 3 hari gejala masih berlanjut, segera konsultasi ke dokter."

            """

        # Get pre-initialized LLM
        llm = LLMPipelineSingleton.get_instance().get_pipeline()

        prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            MessagesPlaceholder(variable_name="history"),
            ("user", "{query}")
        ])

        search_tool = TavilySearch(max_results=5, topic="general")

        generator = Generator(llm, prompt)
        conn_str = current_app.config["MONGODB_URI"]
        db_name = current_app.config["MONGODB_DBNAME"]

        response_from_generator = generator.generate(query, docs, session_id, conn_str, db_name, tools=[search_tool])
        current_app.logger.info(f"Response from generator: {response_from_generator}")

        # The response_from_generator is expected to be {"answer": AIMessage_object}
        final_response = {}
        if isinstance(response_from_generator, dict) and "answer" in response_from_generator:
            ai_message = response_from_generator["answer"] # This is the AIMessage object
            
            # Extract content and any tool calls you might want to send
            final_response["text_content"] = ai_message.content
            final_response["tool_calls"] = ai_message.tool_calls if hasattr(ai_message, 'tool_calls') else []
            # You can add other relevant AIMessage attributes here if needed
        else:
            # Fallback if the structure is not as expected (should ideally not happen with current chatbot.py)
            final_response["text_content"] = "Error: Unexpected response structure from AI."
            final_response["tool_calls"] = []

        final_response["sessionId"] = session_id

        current_app.logger.info(f"Final response: {final_response}")

        return final_response

    except Exception as e:
        current_app.logger.error(f"Error in chat endpoint: {e}")
        return {
            "error": "An error occurred while processing your request",
            "text_content": "Maaf, terjadi kesalahan dalam memproses permintaan Anda.",
            "sessionId": session_id if 'session_id' in locals() else str(uuid4()),
            "tool_calls": []
        }, 500