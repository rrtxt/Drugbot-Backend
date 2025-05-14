from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings

class VectorStoreSingleton:
    _instance = None  # Stores the single instance

    def __new__(cls, host, port, model_name, cache_dir = None):
        """Ensures only one instance of the vector store is created."""
        if cls._instance is None:
            cls._instance = super(VectorStoreSingleton, cls).__new__(cls)

            # Initialize Hugging Face Embeddings
            cls._instance.embeddings = HuggingFaceEmbeddings(
                model_name=model_name, 
                model_kwargs={"device": "cuda"},
                cache_folder=cache_dir
            )

            # Initialize Chroma Vector Store
            cls._instance.vector_store = Chroma(
                collection_name="drugs",
                embedding_function=cls._instance.embeddings,
                persist_directory="./chroma_db",
                client_settings=Settings(
                    chroma_api_impl="chromadb.api.fastapi.FastAPI",
                    chroma_server_host=host,
                    chroma_server_http_port=port
                )
            )

        return cls._instance  # Return the existing instance

    def get_vector_store(self):
        """Provides access to the vector store."""
        return self.vector_store
