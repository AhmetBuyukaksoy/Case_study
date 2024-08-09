import logging
from datasets import load_dataset
import pandas as pd 

logging.basicConfig(level=logging.INFO)

dataset = load_dataset("yashika0998/Sentiment-Analysis-on-appReviews")

# Ingest the training texts only
# How to extract the text from the dataset
print(dataset["train"][0]["text"])

dataset["train"].to_parquet("data_pq")
# Create dataframe to manipulate the data
dataframe = pd.read_parquet("data_pq")

