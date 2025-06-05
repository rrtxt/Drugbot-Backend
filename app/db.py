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
    _client = None  # This will be an instance-level attribute, not class-level for client
    _config = None

    def __new__(cls, connection_string=None, default_database_name=None):
        if cls._instance is None:
            if connection_string is None or default_database_name is None:
                raise ValueError("connection_string and default_database_name must be provided "
                                 "when creating the first instance of MongoDBClientSingleton")
            
            instance = super(MongoDBClientSingleton, cls).__new__(cls)
            # Store config on the class for potential re-use if get_instance is called without args later
            # (though for lazy loading, client should ideally always be re-established if None)
            cls._config = {
                "connection_string": connection_string,
                "default_database_name": default_database_name
            }
            # Initialize _client to None for each new instance.
            # The actual connection will be made in get_client().
            instance._client = None 
            cls._instance = instance
            print("MongoDBClientSingleton instance created, client will connect on first use.")

        return cls._instance

    @classmethod
    def get_instance(cls, connection_string=None, default_database_name=None):
        if cls._instance is None:
            if connection_string is None or default_database_name is None:
                if cls._config is None: 
                    raise ValueError("connection_string and default_database_name must be provided "
                                     "for the first call to get_instance if not already initialized.")
                # This case is if create_app initialized it, and a worker calls get_instance without args
                # We'll rely on the stored _config.
                connection_string = cls._config["connection_string"]
                default_database_name = cls._config["default_database_name"]
            
        return cls(connection_string, default_database_name)

    def get_client(self):
        """Provides access to the MongoClient instance, connecting if necessary."""
        if self._client is None: # Check instance's client
            # Ensure _config is available on the instance, falling back to class _config
            # This handles the case where get_instance was called by a worker without args after preload
            config_to_use = self._config if hasattr(self, '_config') and self._config else MongoDBClientSingleton._config

            if config_to_use is None:
                # This should ideally not happen if initialization logic is correct
                raise RuntimeError("MongoDBClientSingleton configuration is missing. Cannot create client.")

            try:
                print(f"Attempting to connect to MongoDB using: {config_to_use['connection_string'][:30]}...") # Log part of URI
                self._client = MongoClient(config_to_use["connection_string"])
                # Optional: Ping to verify connection. Gunicorn workers might do this implicitly.
                self._client.admin.command('ping')
                print("Successfully connected to MongoDB in worker.")
            except pymongo_errors.ConnectionFailure as e:
                print(f"Could not connect to MongoDB in worker: {e}")
                self._client = None # Ensure client remains None on failure
                # Depending on your app's needs, you might re-raise or return None
                raise # Re-raise the exception to make the failure visible
            except Exception as e:
                print(f"An unexpected error occurred during MongoDB connection in worker: {e}")
                self._client = None # Ensure client remains None on failure
                raise # Re-raise
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
