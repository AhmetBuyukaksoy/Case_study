import logging
from datasets import load_dataset
import pandas as pd
import string

logging.basicConfig(level=logging.INFO)

dataset = load_dataset("yashika0998/Sentiment-Analysis-on-appReviews")

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
#print(dataframe.head(10)["text"])


