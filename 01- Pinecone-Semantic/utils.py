# Import Libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
from fastapi import HTTPException

os.chdir("/home/mango/Coding/Projects/ Mastering Vector Databases for AI Applications | Arabic")

# Load the dotenv file
_ = load_dotenv(override = True)         # return true if .env exist

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
HF_ACCESS_TOKEN = os.getenv('HF_ACCESS_TOKEN')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Connect to Pinecone DB
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = 'semantic-search'
index = pc.Index(name = index_name)

# Load Model For embeddings
model = SentenceTransformer(
    model_name_or_path = 'sentence-transformers/all-MiniLM-L6-v2',
    device="cuda",
    cache_folder = os.getcwd()
)


# Translate Query into English using Helsinki-NLP/opus-mt-ar-en
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ar-en")

def translate_to_english(user_prompt: str):
    ''' 
    This Function takes the input text and translates it to English using a free Hugging Face model.

    Args:
        (user_prompt: str) --> The input text that we want to translate to English.
    
    Returns:
        (translated_text: str) --> The translation of the input text to English Language.
    '''

    # # Intialize a system prompt (mimicking ChatCompletion style)
    # system_prompt = f"""You will be provided with the following information.
    # 1. An arbitrary input text. The text is delimited with triple backticks. 

    # Perform the following tasks:
    # 1. Translate the following text to English.
    # 2. Return only the translation. Do not provide any additional information.

    # Input text: ```{user_prompt}```
    # """

    # # Prepare messages
    # messages = [
    #     {"role": "system", "content": system_prompt},
    #     {"role": "user", "content": user_prompt}
    # ]

    # # Concatenate messages into a single input string
    # input_text = "\n".join([msg["content"] for msg in messages])

    # Call HuggingFace translation model
    result = translator(user_prompt, max_length=200)

    # Extract translation
    translated_text = result[0]["translation_text"]

    # Validation
    if not translated_text:
        raise ValueError("Failed to translate the text.")

    return translated_text


# Getting Similar IDs using pinecone
def search_vectDB(query_text: str,
                  top_k: int,
                  threshold: float = None):
    ''' 
    This Function is to use the pinecone index to make a query and retrieve similar records.

    Args:
        (query_text: str) --> The query text to get similar records to it.
        (top_k: int) --> The number required of similar records in descending order.
        (threshold: float) --> The threshold to filter the retrieved IDs based on it.
    
    Returns:
        (similar_records: List[dict]) --> A List of dicts containing IDs, scores, and class metadata for similar records.
    '''

    try:
        # 1. Translate the query for better results
        query_translated = translate_to_english(user_prompt=query_text)

        # 2. Get embeddings of the input query
        query_embedding = model.encode(query_translated).tolist()

        # 3. Search in Pinecone (no class filter)
        results = index.query(
            vector=[query_embedding],
            top_k=top_k,
            namespace=index_name,
            # include_metadata=True,
            # include_values=True,
            # filter=
        )

        matches = results["matches"]

        # 4. Filter by threshold if provided
        if threshold:
            similar_records = [
                {
                    "id": int(record["id"]),
                    "score": float(record["score"]),
                }
                for record in matches if float(record["score"]) > threshold
            ]
        else:
            similar_records = [
                {
                    "id": int(record["id"]),
                    "score": float(record["score"]),
                }
                for record in matches
            ]

        return similar_records

    except Exception as e:
        raise HTTPException(status_code=500, detail='Failed to get similar records' + str(e))


# Upsert New Data to Pinecone
def insert_vectorDB(text: str, text_id: int):

    try:
        # Get Embeddings using HuggingFace model
        embeds_new = model.encode(text).tolist()

        # Prepare data to upsert to pinecone
        to_upsert = [(str(text_id), embeds_new, {})]

        # Insert to pinecone
        _ = index.upsert(vectors=to_upsert, namespace=index_name)

        # Get the count of vectors after upserting
        count_after = index.describe_index_stats()['total_vector_count']

        return {f'Upserting Done: Vectors Count Now is: {count_after} ..'}

    except Exception as e:
        raise HTTPException(status_code=500, detail='Failed to upsert to pinecone vector DB.')


# Delete Vectors form Pinecone
def delete_vectorDB(text_id: int):
  
    try:
        # Delete from Vector DB
        _ = index.delete(ids=[str(text_id)], namespace=index_name)

        # Get the count of vectors after deleting
        count_after = index.describe_index_stats()['total_vector_count']

        return {f'Deleting Done: Vectors Count Now is: {count_after} ..'}

    except Exception as e:
        raise HTTPException(status_code=500, detail='Failed to delete from pinecone vector DB.')
