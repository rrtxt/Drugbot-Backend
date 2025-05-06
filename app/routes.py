from flask import Blueprint, request
from app.chatbot import Retriever, Generator
from app.db import VectorStoreSingleton
from app.llm import LLMPipelineSingleton, CrossRerankerSingleton

main = Blueprint('main', __name__)

@main.route("/", methods=["GET"])
def home():
    return "ini response"

@main.route("/health", methods=["GET"])
def health_check():
    return {"status" : "ok"}

@main.route("/chat", methods=["POST"])
def chat():
    query = request.form["query"] 
    chroma = VectorStoreSingleton.get_vector_store()
    reranker = CrossRerankerSingleton.get_reranker()
    retriever = Retriever(chroma, 10, reranker) 

    docs = retriever.get_relevant_docs(query)

    llm = LLMPipelineSingleton.get_pipeline()
    template = """Anda adalah seorang asisten medis yang ahli dalam memberikan rekomandasi obat.
    Berdasarkan informasi-informasi obat yang telah disediakan berikan rekomendasi obat untuk pertanyaan yang diberikan.
    Rekomendasi obat yang diberikan harus berisi informasi terkait nama obat, deskripsi kegunaan obat, dosis, dan efek samping obat.
    Berikan rekomendasi dengan maksimum 2 paragraf.
    Jika kamu tidak mengetahui terkait informasi obat yang diberikan, maka kamu cukup bilang tidak mengetahuinya.
    Terakhir, tolong kasih tahu ke pengguna bahwa jika dalam waktu 3 hari pengguna masih mengalami gejala yang dialami, maka segera konsultasi ke dokter.

    Konteks: {context}

    Pertanyaan: {query}
    """
    
    generator = Generator(llm, template)
    response = generator.generate(query, docs)

    return response