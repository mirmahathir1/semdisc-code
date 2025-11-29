from sentence_transformers import SentenceTransformer
import torch
import platform
import os
from custom_lib import utils, console
import faiss
import numpy as np

# Load the pre-trained Sentence Transformer model
encoder_model = None
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

model_name = 'paraphrase-distilroberta-base-v1'

def get_output_dim(model):
    return model[1].word_embedding_dimension

def encode(string_array=None, model=None):
    if model is None:
        model = get_sentence_encoder()
    return model.encode(string_array)

def create_sentence_encoder():
    new_model = SentenceTransformer(model_name)
    if torch.cuda.is_available():
        # console.log("text encoder loaidng: CUDA available")
        new_model = new_model.to('cuda')

    return new_model

def load_sentence_encodder(cpu_only=False):
    global encoder_model
    # console.log("text encoder loading: started")
    if platform.system() == 'Darwin':
        console.log("MacOS detected. Text encoder not loaded")
        return
    encoder_model = SentenceTransformer(model_name)
    if cpu_only == False and torch.cuda.is_available():
        # console.log("text encoder loading: CUDA available")
        encoder_model = encoder_model.to('cuda')
    # console.log("text encoder loading: complete")

def reset_sentence_encoder():
    global encoder_model
    encoder_model = None

def get_sentence_encoder(cpu_only=False):
    if encoder_model is None:
        load_sentence_encodder(cpu_only=False)
    return encoder_model

def get_top_k_similar_string_indexes(target_string, k, list_of_strings=None, list_of_string_embeddings=None, list_IndexFlatL2 = None, model=None):
    if model is None:
        model = get_sentence_encoder()

    if list_IndexFlatL2 is None:
        list_IndexFlatL2 = faiss.IndexFlatL2(get_output_dim(model))
        if list_of_string_embeddings is None:
            if list_of_strings is None:
                utils.crash_code("get_top_k_similar_string_indexes received all None arguments")
            list_of_string_embeddings = encode(model, list_of_strings)
        list_IndexFlatL2.add(list_of_string_embeddings)

    target_string_embedding = encode(model, [target_string])[0]
    distances, indices = list_IndexFlatL2.search(target_string_embedding[np.newaxis], k)
    top_indices = [int(index) for index in indices[0]]
    return top_indices

def get_index_flat_l2_for_list_of_strings(list_of_strings, model=None):
    if model is None:
        model = get_sentence_encoder()
    embeddings = encode(model, list_of_strings)
    list_IndexFlatL2 = faiss.IndexFlatL2(get_output_dim(model))
    list_IndexFlatL2.add(embeddings)
    return list_IndexFlatL2

from sentence_transformers import util
import torch

def top_k_cosine_similarity(embeddings_list1, embeddings_list2, k):
    """
    For each embedding in embeddings_list1, find the top k most similar embeddings from embeddings_list2 based on cosine similarity.
    
    Args:
        embeddings_list1 (list): List of embeddings (e.g., user embeddings).
        embeddings_list2 (list): List of embeddings (e.g., dataframe column embeddings).
        k (int): Number of top similar embeddings to retrieve.
        
    Returns:
        tuple: A tuple containing:
            - results (list): A list of dictionaries, where each dictionary contains:
                - 'top_k_indexes': Indexes of the top k most similar embeddings from embeddings_list2.
                - 'mean_cosine_similarity': Mean cosine similarity of the top k embeddings.
            - overall_mean_cosine_similarity (float): The mean of all 'mean_cosine_similarity' values.
    """
    results = []
    mean_cosine_similarities = []
    
    for embedding in embeddings_list1:
        cos_scores = util.cos_sim(embedding, embeddings_list2).squeeze(0)
        
        top_k_values, top_k_indexes = torch.topk(cos_scores, min(k, len(cos_scores)))
        
        mean_cosine_similarity = torch.mean(top_k_values).item()
        mean_cosine_similarities.append(mean_cosine_similarity)
        
        results.append({
            'top_k_indexes': top_k_indexes.tolist(),
            'mean_cosine_similarity': mean_cosine_similarity
        })
    
    overall_mean_cosine_similarity = sum(mean_cosine_similarities) / len(mean_cosine_similarities)
    
    return results, overall_mean_cosine_similarity

    # Example usage:
    # embeddings_list1 = [torch.tensor([[...]]), torch.tensor([[...]])]  # Replace with actual embeddings
    # embeddings_list2 = [torch.tensor([[...]]), torch.tensor([[...]])]
    # k = 3
    # output = top_k_cosine_similarity(embeddings_list1, embeddings_list2, k)
    # printbroken(output)


