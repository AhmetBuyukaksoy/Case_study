import string
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter

# Qdrant parameters
host = "localhost"
port = 6333
prefer_grpc = False
collection_name = "Text Data Collection"


client = QdrantClient(host=host, port=port, prefer_grpc=prefer_grpc)


def preprocess_text(text):
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(embedding_model)


query_text = input("Please enter your query: ")

processed_query = preprocess_text(query_text)

query_embedding = model.encode(processed_query).tolist()

search_results = client.search(
    collection_name=collection_name,
    query_vector=query_embedding,
    # filter=query_filter,
    limit=5,  # Number of results to return
    score_threshold=0.7,  # Minimum score for results
)

# Display search results
for result in search_results:
    print(f"ID: {result.id}, Score: {result.score}, Payload: {result.payload}")
