from custom_lib import utils, console

from sentence_transformers import SentenceTransformer
import torch
import platform
import os
import faiss
import numpy as np
from sentence_transformers import util
import torch
import gc

# Load the pre-trained Sentence Transformer model
encoder_model = None
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

model_name = 'paraphrase-distilroberta-base-v1'

def get_output_dim(model=None):
    model = get_sentence_encoder()
    return model[1].word_embedding_dimension

def encode(string_array=None, normalize = True, model=None):
    if len(string_array) == 0:
        utils.crash_code("encode received an empty string_array")
    if model is None:
        model = get_sentence_encoder()
    embeddings = model.encode(string_array)
    if normalize:
        faiss.normalize_L2(embeddings)
    return embeddings

def create_sentence_encoder(cpu_only=False):
    # console.log("Trying to create sentence encoder")
    new_model = SentenceTransformer(model_name)
    if cpu_only == False and torch.cuda.is_available():
        # console.log("text encoder loaidng: CUDA available")
        new_model = new_model.to('cuda')
    # print("COMPLETED!!")
    return new_model

def reset_sentence_encoder():
    console.log("Reset sentence encoder")
    global encoder_model
    del encoder_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    encoder_model = None

def get_sentence_encoder(cpu_only=False):
    global encoder_model
    if encoder_model is None:
        encoder_model = create_sentence_encoder(cpu_only=False)
    return encoder_model

import numpy as np

def filter_by_cosine_similarity(list_a, list_b, threshold=0.9):
    """
    Return the strings in `list_b` whose cosine similarity with **any**
    string in `list_a` exceeds `threshold`.

    Parameters
    ----------
    list_a : list[str]
        Reference strings.
    list_b : list[str]
        Candidate strings to keep or discard.
    threshold : float, optional (default=0.9)
        Cosine‑similarity cutoff.
    encoder : callable, optional (default=text_encoder)
        Any encoder with a `.encode()` method that:
            * accepts a list[str]
            * returns an **L2‑normalized** NumPy array (shape = (n, d))

    Returns
    -------
    list[str]
        Subset of `list_b` whose similarity to at least one string
        in `list_a` is > `threshold`.
    """
    # Short‑circuit if either list is empty
    if not list_a or not list_b:
        return []

    # Encode and get L2‑normalized embeddings
    emb_a = encode(list_a)            # shape (m, d)
    emb_b = encode(list_b)            # shape (n, d)

    # Cosine similarity = dot product because embeddings are L2‑normalized
    sim_matrix = emb_b @ emb_a.T             # shape (n, m)

    # Boolean mask: True if max similarity for a row exceeds threshold
    keep_mask = sim_matrix.max(axis=1) > threshold

    # Filter list_b by the mask
    return [s for s, keep in zip(list_b, keep_mask) if keep]


def get_top_k_similar_string_indexes(target_string, k, list_of_strings=None, list_of_string_embeddings=None, list_IndexFlatL2 = None, model=None):
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

def get_index_hnsw_flat_for_list_of_strings(list_of_strings):
    embeddings = encode(string_array=list_of_strings)
    hnsw_index = faiss.IndexHNSWFlat(get_output_dim(get_sentence_encoder()), 32)
    hnsw_index.add(embeddings)
    return hnsw_index

def get_index_flat_for_list_of_strings(list_of_strings):
    embeddings = encode(string_array=list_of_strings)
    flat_index = faiss.IndexFlatL2(get_output_dim(get_sentence_encoder()))
    flat_index.add(embeddings)
    return flat_index

def get_top_k_closest_strings_using_hnsw(hnsw_index, list_of_objects, target_embedding, K):
    distances, indices = hnsw_index.search(target_embedding.reshape(1, -1), K)
    top_strings = [list_of_objects[index] for index in indices[0]]
    return top_strings

def get_top_k_closest_strings_using_index_flat(index_flat_index, list_of_objects, target_embedding, K):
    distances, indices = index_flat_index.search(target_embedding.reshape(1, -1), K)
    top_strings = [list_of_objects[index] for index in indices[0]]
    return top_strings

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

