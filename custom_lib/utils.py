import os
import codecs
import numpy as np
import re
import itertools
from collections import defaultdict
import json
from custom_lib import console
import multiprocessing
import math
import argparse
from flask import has_request_context
import hashlib
import uuid
import random
import shutil
import pickle
import subprocess
import os
import subprocess
import textwrap
import signal
import dill
from collections import Counter
import difflib
import psutil
from collections import defaultdict
from datetime import datetime
import time
from itertools import zip_longest

random.seed(0)

def file_load_nested_dict(file_path):
    if ".json" not in file_path:
        crash_code("nested dict must be a json")
    if not exists(file_path):
        return nested_dict()
    return convert_to_nested_dict(file_load(file_path))

def file_dump_nested_dict(nested_dictionary, file_path):
    if ".json" not in file_path:
        crash_code("nested dict must be a json")

    file_dump(convert_to_regular_dict(nested_dictionary), file_path)

def hash_dicts(dicts):
    json_dicts = json.dumps(dicts, sort_keys=True)
    hasher = hashlib.md5()
    hasher.update(json_dicts.encode('utf-8'))
    return hasher.hexdigest()

def select_one_element_randomly_from_list(list):
    return random.choice(list)

def get_random_variable():
    return random.Random(0)

def ordered_sample(lst, k):
    """
    Returns a sample of k elements from the list, maintaining their original order.
    
    Args:
        lst: Input list
        k: Number of elements to sample
        
    Returns:
        A new list containing the sampled elements in their original order
        
    Raises:
        ValueError: If k is larger than the list length
    """
    if k > len(lst):
        raise ValueError("Sample size k cannot be larger than list length")
        
    # Generate sorted random indices
    indices = sorted(random.sample(range(len(lst)), k))
    
    # Return elements corresponding to these indices
    return [lst[i] for i in indices]

def convert_to_tuple_dict(nested_dict):
    """
    Converts a nested dictionary into a dictionary where the keys are tuples
    representing the path to the value in the original nested dictionary.
    
    Args:
        nested_dict: A dictionary where values may also be dictionaries
        
    Returns:
        A new dictionary with tuple keys and non-dictionary values
    """
    result = {}
    for outer_key, inner_dict in nested_dict.items():
        if isinstance(inner_dict, dict):
            for inner_key, value in inner_dict.items():
                result[(outer_key, inner_key)] = value
        else:
            crash_code(f"Expected a dictionary for key '{outer_key}', but got {type(inner_dict)}")
    return result

def get_size_in_bytes_using_pickle(obj):
    """Returns size of object in bytes using pickle serialization"""
    return len(pickle.dumps(obj))

def get_current_time():
    """
    Returns the current time in seconds since the epoch.
    """
    return time.time()

def get_formatted_time():
    """
    Returns a string representing the current time in the format YYYY-MM-DD-HH-MM-SS.
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d-%H-%M-%S")

def jaccard_similarity(list1, list2):
    """
    Compute the Jaccard similarity between two lists.

    Jaccard(list1, list2) = |intersection(list1, list2)| / |union(list1, list2)|

    Duplicates are ignored (the comparison is set‑based).

    Parameters
    ----------
    list1, list2 : list
        Input lists (they can contain any hashable, comparable items).

    Returns
    -------
    float
        Jaccard similarity in the range [0, 1].
    """
    # print(list1, list2)
    # exit(0)
    set1, set2 = set(list1), set(list2)
    if not set1 and not set2:          # both empty → define similarity as 1
        return 1.0

    intersection_size = len(set1 & set2)
    union_size = len(set1 | set2)
    return intersection_size / union_size

def measure_memory_usage_GB():
    """
    Returns the current process's memory usage in GB.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024 * 1024) 

def get_available_memory_gb():
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    return available_gb

def best_match_difflib(strings, query):
    """
    Returns the string from 'strings' that has the highest similarity
    to 'query', using difflib's built-in ratio.
    """
    best_score = 0
    best_string = None
    for s in strings:
        # Create a SequenceMatcher and get a similarity ratio (0 to 1)
        score = difflib.SequenceMatcher(None, s, query).ratio()
        if score > best_score:
            best_score = score
            best_string = s
    return best_string

def calculate_entropy(data):
    """
    Calculate the Shannon entropy (in bits) of a list of discrete values.
    
    Parameters:
    -----------
    data : list
        The list of values for which we want to compute the entropy.
    
    Returns:
    --------
    float
        The Shannon entropy in bits.
    """
    if not data:
        return 0.0
    
    # Count how many times each unique value appears
    data = [str(item) for item in data]
    counter = Counter(data)
    total_count = len(data)
    
    entropy = 0.0
    for count in counter.values():
        # Probability of this value
        p = count / total_count
        # Accumulate the entropy
        entropy -= p * math.log2(p)
    
    return entropy


# Custom exception for timeouts
class TimeoutException(Exception):
    pass

# Timeout handler
def timeout_handler(signum, frame):
    raise TimeoutException("Function execution timed out")

# Function wrapper to run with a timeout
FUNCTION_TIME_OUT = "Function timed out"
def run_with_timeout(func, timeout, *args, **kwargs):
    # Set the signal handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)  # Set the timeout

    try:
        result = func(*args, **kwargs)  # Call the function with arguments
    except TimeoutException:
        return FUNCTION_TIME_OUT
    finally:
        signal.alarm(0)  # Disable the alarm
    return result

def randomize_dictionary_keys(original_dict, n):
    if n > len(original_dict):
        raise ValueError("n cannot be greater than the number of keys in the dictionary")
    
    selected_keys = random.sample(list(original_dict.keys()), n)
    new_dict = {key: original_dict[key] for key in selected_keys}
    return new_dict

def remove_dulicates_in_string_list(string_list):
    return list(set(string_list))

def extract_json_from_string(input_string):
    # Find the starting index of the first '['
    start_index = input_string.find('[')
    if start_index == -1:
        raise ValueError("No '[' found in the input string.")

    # Find the ending index of the last ']'
    end_index = input_string.rfind(']')
    if end_index == -1:
        raise ValueError("No ']' found in the input string.")
    
    # Extract the substring that should be valid JSON
    json_substring = input_string[start_index:end_index + 1]

    # Parse the JSON substring
    try:
        data = json.loads(json_substring)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON data: {e}")

    return data

def get_cosine_similarity(vec_a, vec_b, *, eps: float = 1e-8) -> float:
    """
    Cosine similarity between two 1-D embeddings.

    Parameters
    ----------
    vec_a, vec_b : array-like (same length)
        Embedding vectors (list, tuple, or NumPy array).
    eps : float, optional
        Small value to avoid division by zero when a norm is zero.

    Returns
    -------
    float
        Similarity in the range [-1, 1].
    """
    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)

    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("Inputs must be 1-D arrays of the same length.")

    denom = max(np.linalg.norm(a) * np.linalg.norm(b), eps)
    return float(np.dot(a, b) / denom)

def get_combinations(list_of_items, c):
    return list(itertools.combinations(list_of_items, c))

import random
from typing import Dict, Hashable, Any

def sample_dict(src: Dict[Hashable, Any], n: int) -> Dict[Hashable, Any]:
    """
    Return a new dictionary containing `n` randomly selected key–value pairs
    from `src`.  Sampling is **without replacement**.

    Parameters
    ----------
    src   : dict
        Source dictionary to sample from.
    n     : int
        Number of keys to sample. Must be ≤ len(src).
    seed  : int | None, optional
        Random-seed for reproducibility.

    Raises
    ------
    ValueError
        If `n` is greater than the number of keys in `src`.

    Returns
    -------
    dict
        Dictionary with `n` sampled key–value pairs.
    """
    if n > len(src):
        return src

    random.seed(0)

    keys = random.sample(src.keys(), n)
    return {k: src[k] for k in keys}


def run_command_in_dir(directory: str, command: str) -> str:
    """
    Run a system command in the specified directory, streaming output 
    to stdout as the command runs, and return the full output upon completion.
    """
    output_lines = []

    # Open the subprocess with the specified command and directory
    with subprocess.Popen(
        command,
        cwd=directory,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True  # For Python 3.7+, ensures we read strings instead of bytes
    ) as process:

        # Stream output line by line
        for line in process.stdout:
            console.log(line, end='')  # Stream to the console
            output_lines.append(line)

        # Wait for the process to complete
        process.wait()

    final_output = "".join(output_lines)
    has_error = "Traceback" in final_output
    # Return the collected output as a single string
    return {"final_output": final_output, "has_error": has_error}


# Example usage:
# execute_command_in_directory("/path/to/directory", "ls -la")


def copy_file_or_directory(source: str, destination: str) -> bool:
    """
    Copies a file or directory from the source path to the destination path.

    Parameters:
        source (str): The path to the source file or directory.
        destination (str): The path to the destination file or directory.

    Returns:
        bool: True if the file or directory is copied successfully, False otherwise.
    """
    try:
        if os.path.isdir(source):
            # Copy the directory and its contents
            shutil.copytree(source, destination)
        elif os.path.isfile(source):
            # Copy the file
            shutil.copy(source, destination)
        else:
            console.log(f"Source path is neither a file nor a directory: {source}")
            crash_code(f"Traceback Source path is neither a file nor a directory: {source}")
            return False
        return True
    except FileNotFoundError:
        crash_code(f"Path not found: {source}, {destination}")
    except PermissionError:
        crash_code("Permission denied.")
    except FileExistsError:
        crash_code(f"Destination already exists: {destination}")
    except Exception as e:
        crash_code(f"An error occurred: {e}")
    
    return False

def file_load(path):
    # spinner = console.Spinner(f"loading from file {path}")
    console.log(f"loading from file started: {path}")
    extension = path.split(".")[-1]
    file_object = None
    if extension == 'pickle':
        file_object = pickle.load(open(path, 'rb'))
    elif extension == 'json':
        file_object = json.load(open(path, 'r'))
    elif extension == 'dill':
        file_object = dill.load(open(path, 'rb'))
    else:
        crash_code(f"file extension not valid. path: {path}, extension found: {extension}")
    # spinner.stop()
    console.log(f"loading from file ended: {path}")
    return file_object

def file_dump(object, path):
    # spinner = console.Spinner(f"dumping to file {path}")
    console.log(f"dumping to file started: {path}")
    extension = path.split(".")[-1]
    if extension == 'pickle':
        pickle.dump(object, open(path, 'wb'))
    elif extension == 'json':
        json.dump(obj=object, fp=open(path, 'w'), indent=4)
    elif extension == 'dill':
        dill.dump(object, open(path, 'wb'))
    else:
        crash_code(f"file extension not valid. path: {path}, extension found: {extension}")
    # spinner.stop()
    console.log(f"dumping to file ended: {path}")

def text_dump(content: str, file_path: str, wrap: bool = False, wrap_width: int = 150):
    try:
        if wrap:
            content = "\n".join(textwrap.fill(line, width=wrap_width) for line in content.splitlines())

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
    except Exception as e:
        console.log(f"An error occurred while writing to the file: {e}")

def listdir(directory, substring=None):
    all_files_and_folders = os.listdir(directory)
    if substring is not None:
        return [file_folder_name for file_folder_name in all_files_and_folders if substring in file_folder_name]
    return all_files_and_folders

def exists(path):
    return os.path.exists(path)

def reset_directory(directory_path):
    if exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path)

def create_directory(directory_path):
    os.makedirs(directory_path, exist_ok=True)

def delete(path):
    if exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.unlink(path)

def empty_function():
    return None

def nested_dict():
    return defaultdict(nested_dict)

def nested_dict_factory(leaf_type: type):
    def factory():
        return defaultdict(nested_dict_factory(leaf_type))
    return factory

def typed_nested_dict(leaf_type: type):
    return defaultdict(nested_dict_factory(leaf_type))

def convert_to_regular_dict(d):
    if isinstance(d, defaultdict):
        d = {k: convert_to_regular_dict(v) for k, v in d.items()}
    return d

def convert_to_nested_dict(d):
    """
    Recursively turn a regular (possibly deeply-nested) dict into a
    `defaultdict` whose children are also `defaultdict`s created by
    `nested_dict()`.  
    Non-mapping values are copied as-is.

    Parameters
    ----------
    d : dict | defaultdict
        The input mapping to convert.

    Returns
    -------
    defaultdict
        A structure identical in content to *d* but with auto-expanding
        defaultdict nodes, so you can assign to deeper keys without
        first checking/creating intermediate levels.
    """
    if isinstance(d, defaultdict):          # already in the desired form
        return d

    nd = nested_dict()
    for k, v in d.items():
        nd[k] = convert_to_nested_dict(v) if isinstance(v, dict) else v
    return nd

def top_k_items(d, k):
    # Sort the dictionary items by value in descending order and take the top k items
    top_k = dict(sorted(d.items(), key=lambda item: item[1], reverse=True)[:k])
    return top_k

def shuffle_list(original_list):
    new_list = original_list[:]
    random.shuffle(new_list)
    return new_list

def get_slurm_job_id():
    if not 'SLURM_JOB_ID' in os.environ:
        return None
    return os.environ['SLURM_JOB_ID']

def get_available_job_memory_for_slurm_job_in_gigabytes():
    if not 'SLURM_MEM_PER_NODE' in os.environ:
        return None
    return int(os.environ['SLURM_MEM_PER_NODE'])/1024

def get_avaiable_cpu_core_for_slurm_job():
    if not 'SLURM_CPUS_PER_TASK' in os.environ:
        return None
    return int(os.environ['SLURM_CPUS_PER_TASK'])

def is_function_running_under_flask():
    return has_request_context()

def parse_arg(arguments):
    """
    Details of the arguments:
    - name: The name of the argument (without the leading '--').
    - type: The type of the argument (e.g., int, float, str).
    - default: The default value for the argument.
    """
    parser = argparse.ArgumentParser()
    for argument in arguments:
        parser.add_argument(f'--{argument["name"]}', type=argument['type'], default=argument['default'])
    args, unknown = parser.parse_known_args()
    args = vars(args)
    return args

def divide_list(lst, n):
    length = len(lst)
    size = length // n
    remainder = length % n
    start = 0
    result = []
    for i in range(n):
        end = start + size
        if remainder:
            end += 1
            remainder -= 1
        if len(lst[start:end]) == 0:
            continue
        result.append(lst[start:end])
        start = end
    return result

def remove_non_utf8_lines(filename):
    with codecs.open(filename, 'r', encoding='utf-8', errors='ignore') as f_in:
        lines = f_in.readlines()

    with codecs.open(filename, 'w', encoding='utf-8') as f_out:
        for line in lines:
            try:
                line.encode('utf-8')
                f_out.write(line)
            except UnicodeEncodeError:
                continue

def get_filename_without_extension(filename_with_extension):
    return ".".join(filename_with_extension.split('.')[:-1])

def os_saveable_filename(input_string):
    invalid_chars_pattern = r'[<>:"/\\|?*\n\r\t\v\f]'
    return re.sub(invalid_chars_pattern, '__', input_string).strip()

def crash_code(error_message, write_to_file=True):
    console.log(f"ERROR: {error_message}. Code will now exit")
    os._exit(1)

def get_random_hash():
    return hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()

def list_product(list_of_lists):
    return list(itertools.product(*list_of_lists))

def hash_tuples(input_tuple):
    # Convert the tuple to a string representation
    string_representation = str(input_tuple)
    # Use hashlib to create a hash (e.g., SHA-256)
    hash_object = hashlib.sha256(string_representation.encode())
    return hash_object.hexdigest()

def interleave_lists(list_of_lists):
    interleaved = []
    for elements in zip_longest(*list_of_lists):
        interleaved.extend([el for el in elements if el is not None])
    return interleaved

def get_sublist_between_markers(full_list, markers):
    """
    Returns the sublist that starts and ends with elements from the markers list.
    
    Args:
        full_list: The main list to search in
        markers: List of elements that can start/end the sublist
    
    Returns:
        A sublist of full_list or empty list if no valid sublist found
    """
    # Find all indices where elements are in markers
    indices = [i for i, x in enumerate(full_list) if x in markers]
    
    # If no markers found, return empty list
    if not indices:
        return []
    
    # Return the slice from first to last marker (inclusive)
    start_idx = indices[0]
    end_idx = indices[-1]
    return full_list[start_idx:end_idx+1]

def sort_product_by_index_sum(lists):
    # Precompute item-to-index mappings for each list
    list_mappings = []
    for lst in lists:
        # Create a dictionary mapping each item to its index
        # Using enumerate to track positions
        mapping = {id(item): idx for idx, item in enumerate(lst)}
        list_mappings.append(mapping)
    
    # Compute the Cartesian product
    products = list(itertools.product(*lists))
    
    # Calculate index sum for each product tuple
    products_with_index_sum = []
    for tuple_item in products:
        index_sum = 0
        for i in range(len(tuple_item)):
            item = tuple_item[i]
            # Get the index from the precomputed mapping
            index_sum += list_mappings[i][id(item)]
        products_with_index_sum.append((index_sum, tuple_item))
    
    # Sort by index sum
    products_with_index_sum.sort(key=lambda x: x[0])
    
    # Extract sorted tuples
    sorted_products = [item for (_, item) in products_with_index_sum]
    
    return sorted_products

import heapq
from typing import Iterable, List, Sequence, Tuple

def stream_product_by_index_sum(lists: Sequence[Sequence]) -> Iterable[Tuple]:
    """
    Lazily yield tuples from the Cartesian product of *lists* ordered by the sum
    of their element indices.

    Example
    -------
    >>> a, b, c = ['A0', 'A1'], ['B0', 'B1', 'B2'], ['C0']
    >>> for combo in stream_product_by_index_sum([a, b, c]):
    ...     print(combo)
    ('A0', 'B0', 'C0')   # 0+0+0 = 0
    ('A1', 'B0', 'C0')   # 1+0+0 = 1
    ('A0', 'B1', 'C0')   # 0+1+0 = 1
    ('A1', 'B1', 'C0')   # 1+1+0 = 2
    ('A0', 'B2', 'C0')   # 0+2+0 = 2
    ('A1', 'B2', 'C0')   # 1+2+0 = 3
    """
    n = len(lists)
    if n == 0 or any(len(lst) == 0 for lst in lists):
        return  # nothing to produce

    # Min‑heap stores (index_sum, indices_tuple)
    start = tuple(0 for _ in lists)
    heap: List[Tuple[int, Tuple[int, ...]]] = [(0, start)]
    seen = {start}

    lengths = [len(lst) for lst in lists]

    while heap:
        index_sum, indices = heapq.heappop(heap)
        # Yield the actual items corresponding to the current index tuple
        yield tuple(lists[i][idx] for i, idx in enumerate(indices))

        # Expand neighbours by advancing one coordinate at a time
        for dim in range(n):
            if indices[dim] + 1 < lengths[dim]:
                nxt = list(indices)
                nxt[dim] += 1
                nxt = tuple(nxt)
                if nxt not in seen:
                    # New sum = old sum - old_idx + new_idx
                    new_sum = index_sum - indices[dim] + nxt[dim]
                    heapq.heappush(heap, (new_sum, nxt))
                    seen.add(nxt)



def sample_list(original_list, sample_spec):
    """
    Returns a new list containing a random sample from the original list.

    Parameters:
        original_list (list): The list to sample from.
        sample_spec (Number): Either:
            - A float between 0 and 1 indicating the portion of the list to sample, or
            - An integer indicating the exact number of items to sample

    Returns:
        list: A list with randomly sampled elements.

    Raises:
        ValueError: If sample_spec is invalid or would result in invalid sample size.
    """
    if isinstance(sample_spec, float):
        if not 0 <= sample_spec <= 1:
            raise ValueError("When portion is a float, it must be between 0 and 1.")
        sample_size = int(len(original_list) * sample_spec)
    elif isinstance(sample_spec, int):
        if sample_spec < 0:
            raise ValueError("Sample size cannot be negative.")
        sample_size = sample_spec
    else:
        raise ValueError("sample_spec must be either a float (portion) or integer (fixed size).")

    if sample_size > len(original_list):
        return original_list

    return random.sample(original_list, sample_size)

def random_float_between(min_value: float, max_value: float) -> float:
    """
    Returns a random float between min_value and max_value.

    Parameters:
    min_value (float): The lower bound of the range.
    max_value (float): The upper bound of the range.

    Returns:
    float: A random float between min_value and max_value.
    """
    return random.uniform(min_value, max_value)

def find_indexes(lst, value):
    """
    Returns a list of indexes where the given value matches elements in the list.

    Parameters:
    lst (list): The list to search.
    value (any): The value to find.

    Returns:
    list: A list of indexes where the value matches.
    """
    return [index for index, element in enumerate(lst) if element == value]

def list_set_subtraction(list1, list2):
    """
    Returns the set subtraction of two lists: elements in list1 but not in list2.
    """
    return list(set(list1) - set(list2))

def deep_copy(object):
    return json.loads(json.dumps(object))