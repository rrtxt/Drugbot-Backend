from langchain_huggingface import HuggingFacePipeline
from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from transformers import pipeline, BitsAndBytesConfig
import torch

class LLMPipelineSingleton:
    _instance = None  # Holds the single instance
    _model_id = None  # Store the model ID
    _quantization_config = None # Store quantization config

    def __new__(cls, model_id=None, quantization_config=None):
        """Ensures only one instance is created (Singleton)."""
        if cls._instance is None:
            if model_id is None:
                raise ValueError("model_id must be provided when creating the first instance")
            cls._instance = super(LLMPipelineSingleton, cls).__new__(cls)
            cls._model_id = model_id
            cls._quantization_config = quantization_config
            cls._instance._initialize_pipeline(model_id, quantization_config)
        return cls._instance

    @classmethod
    def get_instance(cls, model_id=None, quantization_config=None):
        """Class method to get or create the instance."""
        if cls._instance is None and model_id is None:
            raise ValueError("model_id must be provided when creating the first instance")
        # If instance exists, model_id and quantization_config are ignored
        # To reconfigure, a new mechanism would be needed (e.g. a reset method)
        if cls._instance is not None:
            return cls._instance
        return cls(model_id, quantization_config)

    def _initialize_pipeline(self, model_id, quantization_config):
        """Loads the model only once.""" 
        model_kwargs = {'cache_dir' : './model_cache'}
        if quantization_config:
            model_kwargs['quantization_config'] = quantization_config
            
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            model_kwargs=model_kwargs,
            device_map="auto",
            max_new_tokens=1000,
            temperature=0.7
        )

    def get_pipeline(self):
        """Returns the initialized Hugging Face pipeline."""
        return HuggingFacePipeline(pipeline=self.pipe)

class CrossRerankerSingleton:
    _instance = None  # Class variable to hold the single instance
    _reranker_name = None  # Store the reranker name

    def __new__(cls, reranker_name=None):
        """Ensures only one instance of CrossEncoderReranker exists."""
        if cls._instance is None:
            if reranker_name is None:
                raise ValueError("reranker_name must be provided when creating the first instance")
            cls._instance = super(CrossRerankerSingleton, cls).__new__(cls)
            cls._reranker_name = reranker_name
            model_kwargs = {'device' : 'cuda', 'cache_dir' : './model_cache'}
            cls._instance.cross_encoder = HuggingFaceCrossEncoder(model_name=reranker_name, model_kwargs=model_kwargs)
            cls._instance._initialize_reranker(cls._instance.cross_encoder)
        return cls._instance

    @classmethod
    def get_instance(cls, reranker_name=None):
        """Class method to get or create the instance."""
        if cls._instance is None and reranker_name is None:
            raise ValueError("reranker_name must be provided when creating the first instance")
        return cls(reranker_name)

    def _initialize_reranker(self, reranker_model):
        """Loads the CrossEncoderReranker only once."""
        self.reranker = CrossEncoderReranker(
            model=reranker_model,
            top_n=5
        )

    def get_reranker(self):
        """Returns the initialized CrossEncoderReranker."""
        return self.reranker
 

