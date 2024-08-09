import logging
from datasets import load_dataset
import string
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import pyarrow.parquet as pq
from qdrant_client.http.models import PointStruct, VectorParams, Distance


logging.basicConfig(level=logging.INFO)

dataset = load_dataset("yashika0998/Sentiment-Analysis-on-appReviews")

# Qdrant parameters
host = "localhost"
port = 6333
prefer_grpc = False

# Ingest the training texts only
# How to extract the text from the dataset
parquet_file = "data_pq"
batch_size = 256

dataset["train"].to_parquet(parquet_file)
# Create dataframe to manipulate the data


# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(embedding_model)


def get_embeddings(texts):
    embeddings = model.encode(texts)
    return embeddings


client = QdrantClient(host=host, port=port, prefer_grpc=prefer_grpc)

print("Qdrant client has been succesfully created")

collection_name = "Text Data Collection"
print(client.get_collections())

if collection_name not in [col.name for col in client.get_collections().collections]:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=model.get_sentence_embedding_dimension(), distance=Distance.COSINE
        ),
    )


# Ingest the data
def read_parquet_in_batches(parquet_file, batch_size):
    parquet_file = pq.ParquetFile(parquet_file)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()
        df["text"] = df["text"].apply(preprocess_text)
        yield df


def upsert_data_in_batches(parquet_file, batch_size):
    for batch_index, batch_df in enumerate(
        read_parquet_in_batches(parquet_file, batch_size)
    ):
        texts = batch_df["text"].tolist()
        labels = batch_df["labels"].tolist()
        embeddings = get_embeddings(texts)

        # Prepare points for insertion
        points = [
            PointStruct(
                id=batch_index * batch_size + i,
                vector=embedding.tolist(),
                payload={"text": text, "label": label},
            )
            for i, (embedding, text, label) in enumerate(zip(embeddings, texts, labels))
        ]

        # Upsert points into Qdrant
        client.upsert(collection_name=collection_name, points=points)


# Call the function to start the upsert process
upsert_data_in_batches(parquet_file, batch_size)

print("Data ingested into Qdrant.")
