from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings

class VectorStoreSingleton:
    _instance = None  # Stores the single instance
    _config = None   # Stores the configuration

    def __new__(cls, host=None, port=None, model_name=None, cache_dir=None):
        """Ensures only one instance of the vector store is created."""
        if cls._instance is None:
            if host is None or port is None or model_name is None:
                raise ValueError("host, port, and model_name must be provided when creating the first instance")
            
            cls._instance = super(VectorStoreSingleton, cls).__new__(cls)
            cls._config = {
                "host": host,
                "port": port,
                "model_name": model_name,
                "cache_dir": cache_dir
            }

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
                client_settings=Settings(
                    chroma_api_impl="chromadb.api.fastapi.FastAPI",
                    chroma_server_host=host,
                    chroma_server_http_port=port
                )
            )

        return cls._instance

    @classmethod
    def get_instance(cls, host=None, port=None, model_name=None, cache_dir=None):
        """Class method to get or create the instance."""
        if cls._instance is None and (host is None or port is None or model_name is None):
            raise ValueError("host, port, and model_name must be provided when creating the first instance")
        return cls(host, port, model_name, cache_dir)

    def get_vector_store(self):
        """Provides access to the vector store."""
        return self.vector_store
