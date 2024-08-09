import logging
from datasets import load_dataset
import pandas as pd
import string
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


logging.basicConfig(level=logging.INFO)

dataset = load_dataset("yashika0998/Sentiment-Analysis-on-appReviews")

# Qdrant parameters
host = "localhost"
port = 6333
prefer_grpc = True

# Ingest the training texts only
# How to extract the text from the dataset
print(dataset["train"][0]["text"])

dataset["train"].to_parquet("data_pq")
# Create dataframe to manipulate the data
dataframe = pd.read_parquet("data_pq")


# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


dataframe["text"] = dataframe["text"].apply(preprocess_text)
# print(dataframe.head(10)["text"])

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def get_embeddings(texts):
    embeddings = model.encode(texts)
    return embeddings


texts = dataframe["text"].tolist()
embeddings = get_embeddings(texts)
vector_size = embeddings.shape[1]

print(vector_size)
print(embeddings[0])

client = QdrantClient(host=host, port=port, prefer_grpc=prefer_grpc)

print("Qdrant client has been succesfully created")

collection_name = "Text Data Collection"
client.create_collection(collection_name)

if collection_name not in client.get_collections():
    client.create_collection(
        collection_name=collection_name, vector_size=vector_size, distance="Cosine"
    )
    
    
