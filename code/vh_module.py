# text_embedding_service/target_object.py
class TargetObject:
    def __init__(self, id, text, current_embedding, current_embedding_model, metadata={}):
        self.id = id
        self.text = text
        self.current_embedding = current_embedding
        self.current_embedding_model = current_embedding_model
        self.metadata = metadata

# text_embedding_service/embedding_function.py
import spacy
spacy.prefer_gpu()
#nlp = spacy.load("en_core_web_sm")

from chromadb import EmbeddingFunction
from typing import Union, List

Embeddable = Union[str, List[str]]
Embedding = List[float]
Embeddings = Union[Embedding, List[Embedding]]

class SpacyWord2VecEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")

    def __call__(self, input: Embeddable) -> Embeddings:
        if isinstance(input, str):
            return self._embed_single(input)
        elif isinstance(input, list):
            return self._embed_batch(input)
        else:
            raise TypeError(f"Input must be a string or list of strings, got {type(input)}")

    def _embed_single(self, text: str) -> Embedding:
        doc = self.nlp(text)
        return doc.vector.tolist()

    def _embed_batch(self, texts: List[str]) -> List[Embedding]:
        return [self._embed_single(text) for text in texts]

# text_embedding_service/text_embedding_service.py
import chromadb
from chromadb.config import Settings
from typing import Union

class TextEmbeddingService:
    def __init__(self, data_set_name, embedding_model, embedding_function=None, metric="cosine", persist_path=None, remote_host=None):
        """
        Initializes the TextEmbeddingService with the specified embedding models and functions.
        
        Args:
        - embedding_model (str): The name of the text embedding model to use.
        - embedding_function (AbstractEmbeddingFunction): AbstractEmbeddingFunction instance.
        """
        # https://docs.trychroma.com/guides#creating,-inspecting,-and-deleting-collections:~:text=Chroma%20will%20use%20sentence%20transformer%20as%20a%20default
        self.embedding_model = embedding_model
        self.embedding_function = embedding_function() if embedding_function else None #Chroma will use sentence transformer as a default 
        self.entries = []
        
        if persist_path:
            self.client = chromadb.PersistentClient(path=persist_path)
        elif remote_host:
            # remote_host["host"] = 'localhost'
            # remote_host["port"] = 8000
            self.client = chromadb.HttpClient(host=remote_host["host"], port=remote_host["port"])
        else:
            # https://docs.trychroma.com/reference/py-client#get_max_batch_size:~:text=for%20this%20client.-,Client%23,-Copy%20Code
            self.client = chromadb.Client(Settings()) # Uses default tenant and database.

        self.metric = metric
        self.metadata = self._set_metric_metadata(metric)
        self.collection_name = data_set_name + "_" + embedding_model
        self.collection = self.client.get_or_create_collection(name=self.collection_name, metadata=self.metadata, embedding_function=self.embedding_function)

    def _set_metric_metadata(self, metric):
        # https://docs.trychroma.com/guides#:~:text=Changing%20the%20distance,hnsw%3Aspace.
        # Other Distance Measures : https://medium.com/@jodancker/a-brief-introduction-to-distance-measures-ac89cbd2298
        if metric == "cosine":
            self.metadata={"hnsw:space": "cosine"}
            self.metric_details = "Distance = 1 - ∑(Ai x Bi) / Dot(Sqrt(∑(Ai^2),Sqrt(∑(Bi^2))))"
            self.metric_details_domain = "The Cosine distance, which substracts the cosine similarity from 1, lies between 0 (similar values) and 1 (different values)."
            self.metric_notes = "The main disadvantage of the Cosine distance is that is does not consider the magnitude but only the direction of vectors. Hence, the differences in values is not fully taken into account."
        elif metric == "l2":
            self.metadata={"hnsw:space": "l2"}
            self.metric_details = "Euclidian or l2 Distance = ∑(Ai - Bi)^2"
            self.metric_details_domain = "The Euclidean distance measures the shortest distance between two real-valued vectors."
            self.metric_notes = "The Euclidean distance has two major disadvantages. First, the distance measure does not work well for higher dimensional data than 2D or 3D space. Second, if we do not normalize and/or standardize our features, the distance might be skewed due to different units."
        elif metric == "ip":
            # Inner Product Space: https://web.auburn.edu/holmerr/2660/Textbook/innerproduct-print.pdf
            # Cauchy-Schwarz theorem. For all v,w ∈ V, |v, w| ≤ v w.
            # The inner product is a length projection of one vector onto the other by the angle between the two vectors.
            self.metadata={"hnsw:space": "ip"}
            self.metric_details = "Distance = 1 - ∑(Ai x Bi)"
            self.metric_details_domain = "The inner product distance... Depends if the vector is normalized..."
            self.metric_notes = "May have some unique properties of interest as it may incorporate the length?"
        else:
            self.metadata=None # l2 is the default distance function in "hnsw:space"
        return self.metadata
    def purge_current_collection(self):
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name, metadata=self.metadata, embedding_function=self.embedding_function)
        
    def purge_all_collections(self):
        active_collections = self.client.list_collections()
        for item in active_collections:
            print("Deleting Collection: ", item)
            self.client.delete_collection(name=item.name)

    def show_client_stats(self):
        chroma_version = self.client.get_version()
        print(f"Chroma Client Version: {chroma_version}")
        max_batch_size = self.client.get_max_batch_size()
        print(f"Chroma Client Max Batch Size: {max_batch_size}")
        active_collections = self.client.list_collections()
        print(f"Chroma Client Active Collections: {active_collections}")

        chroma_client_settings = self.client.get_settings()
        print(f"Chroma Client Settings:")
        print(chroma_client_settings)

    def show_collection_stats(self):
        embedding_count = self.collection.count()
        print("Collection Count: ", embedding_count)

    def add_entry(self, entry):
        self.entries.append(entry)
        self.collection.add(ids=[entry.id], metadatas=[entry.metadata], documents=[entry.text])

    def text_search(self, query_text, n=5):
        results = self.collection.query(query_texts=[query_text], n_results=n, include=['embeddings', 'documents', 'distances', 'metadatas'])
        return results