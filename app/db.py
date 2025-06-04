from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings
from pymongo import MongoClient, errors as pymongo_errors

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

class MongoDBClientSingleton:
    _instance = None
    _client = None
    _config = None

    def __new__(cls, connection_string=None, default_database_name=None):
        if cls._instance is None:
            if connection_string is None or default_database_name is None:
                raise ValueError("connection_string and default_database_name must be provided "
                                 "when creating the first instance of MongoDBClientSingleton")
            
            cls._instance = super(MongoDBClientSingleton, cls).__new__(cls)
            cls._config = {
                "connection_string": connection_string,
                "default_database_name": default_database_name
            }
            try:
                cls._client = MongoClient(connection_string)
                # You can add a server ping here to verify connection if needed
                # cls._client.admin.command('ping') 
                print("Successfully connected to MongoDB.")
            except pymongo_errors.ConnectionFailure as e:
                print(f"Could not connect to MongoDB: {e}")
                # Depending on your application's needs, you might want to raise an error here
                # or handle it differently. For now, we'll allow the app to start
                # but the client will be None.
                cls._client = None 
            except Exception as e:
                print(f"An unexpected error occurred during MongoDB connection: {e}")
                cls._client = None


        return cls._instance

    @classmethod
    def get_instance(cls, connection_string=None, default_database_name=None):
        if cls._instance is None:
            if connection_string is None or default_database_name is None:
                # This ensures that if called without args after initialization, it still works
                # but the first call *must* have args.
                if cls._config is None: # Truly the first call attempt without args
                    raise ValueError("connection_string and default_database_name must be provided "
                                     "for the first call to get_instance.")
                # If _config exists, it means it was initialized before, re-use config
                connection_string = cls._config["connection_string"]
                default_database_name = cls._config["default_database_name"]
            
        return cls(connection_string, default_database_name)

    def get_client(self):
        """Provides access to the MongoClient instance."""
        if self._client is None:
            # Attempt to reconnect if the client is None (e.g., due to initial connection failure)
            # This is a simple retry, more sophisticated retry logic might be needed for production
            try:
                print("Attempting to reconnect to MongoDB...")
                self._client = MongoClient(self._config["connection_string"])
                # self._client.admin.command('ping') # Optional: verify connection
                print("Successfully reconnected to MongoDB.")
            except Exception as e:
                print(f"Failed to reconnect to MongoDB: {e}")
                return None # Or raise an error
        return self._client

    def get_database(self, db_name=None):
        """Provides access to a specific database.
        
        Args:
            db_name (str, optional): The name of the database to access. 
                                     Defaults to the default_database_name if None.
        
        Returns:
            pymongo.database.Database or None: The database object, or None if client is not available.
        """
        client = self.get_client()
        if client:
            target_db_name = db_name if db_name else self._config["default_database_name"]
            return client[target_db_name]
        return None
