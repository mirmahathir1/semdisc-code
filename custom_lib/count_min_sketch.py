from probables import CountMinSketch
import numpy as np
import hashlib
import random
import string
from collections import Counter
import pandas as pd
import time

def build_count_min_sketch(values, width=1000, depth=5):
    """
    Builds a Count-Min Sketch from a list of values.

    :param values: The list (or iterable) of values to be inserted into the CMS.
    :param width:  The width (number of columns) of the CMS table.
    :param depth:  The depth (number of hash functions/rows) in the CMS.
    :return:       A CountMinSketch object with all values added.
    """
    # Create a CountMinSketch with the specified width and depth
    cms = CountMinSketch(width=width, depth=depth)

    # Insert values into the sketch
    for val in values:
        cms.add(val)

    return cms

def add_to_count_min_sketch(cms, value, frequency):
    cms.add(key = value, num_els = frequency)

def build_count_min_sketch_manual(values, width=1000, depth=5):
    cms = CountMinSketchManual(width=width, depth=depth)

    for val in values:
        cms.add(val)

    return cms

def get_frequency(cms, value):
    return cms.check(value)

class CountMinSketchManual:
    def __init__(self, width=1000, depth=5):
        """
        Initializes a Count-Min Sketch.
        :param width: Number of columns in the sketch table (determines accuracy)
        :param depth: Number of hash functions (determines probability of overestimation)
        """
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=np.int32)

    def _hash(self, item, i):
        """Hashes the item using a combination of SHA-256 and an index."""
        hash_val = hashlib.sha256(f"{item}-{i}".encode()).hexdigest()
        return int(hash_val, 16) % self.width

    def add(self, item, count=1):
        """Increments the count of the given item by the specified count."""
        for i in range(self.depth):
            index = self._hash(item, i)
            self.table[i, index] += count

    def check(self, item):
        """Returns the estimated count of the given item."""
        return min(self.table[i, self._hash(item, i)] for i in range(self.depth))

    def merge(self, other):
        """
        Merges another Count-Min Sketch with the same dimensions.
        :param other: Another CountMinSketch instance with the same width and depth
        """
        if self.width != other.width or self.depth != other.depth:
            raise ValueError("Cannot merge sketches of different dimensions")
        self.table += other.table

    def __repr__(self):
        return f"CountMinSketchManual(width={self.width}, depth={self.depth})"



# -----------------------------
# Example usage:
if __name__ == "__main__":
    data = ["apple", "banana", "apple", "orange", "banana", "apple", "kiwi"]

    # Build a CMS with default width=1000, depth=5
    cmsketch = build_count_min_sketch_manual(data)

    # Query the frequencies of various items
    console.log("Frequency of 'apple' :", get_frequency(cmsketch, "apple"))
    console.log("Frequency of 'banana':", get_frequency(cmsketch, "banana"))
    console.log("Frequency of 'orange':", get_frequency(cmsketch, "orange"))
    console.log("Frequency of 'kiwi'  :", get_frequency(cmsketch, "kiwi"))
    console.log("Frequency of 'mango' :", get_frequency(cmsketch, "mango"))

    # Generate 1 million random strings (length 10)
    num_strings = 1_000_000
    unique_strings = ["".join(random.choices(string.ascii_letters + string.digits, k=10)) for _ in range(100_000)]
    random_strings = [random.choice(unique_strings) for _ in range(num_strings)]
    cms_manual = build_count_min_sketch_manual(random_strings)
    cms_lib = build_count_min_sketch(random_strings)

    df = pd.DataFrame(random_strings, columns=['values'])
    pandas_counts = df['values'].value_counts()

    # Pick a random string to check frequency
    random_choice = random.choice(random_strings)

    # Measure query time for Count-Min Sketch
    start_query_cmslib = time.time()
    estimated_frequency_lib = get_frequency(cms_lib, random_choice)
    cmslib_query_time = time.time() - start_query_cmslib

    start_query_cmsmanual = time.time()
    estimated_frequency_manual = get_frequency(cms_manual, random_choice)
    cmsmanual_query_time = time.time() - start_query_cmsmanual

    # Measure query time for Pandas
    start_query_pandas = time.time()
    true_frequency = pandas_counts[random_choice]
    pandas_query_time = time.time() - start_query_pandas

    speedup_query_lib = pandas_query_time / cmslib_query_time

    speedup_query_manual = pandas_query_time / cmsmanual_query_time

    # Display results
    console.log(f"true freq: ", true_frequency)
    console.log(f"speedup cms lib: ", speedup_query_lib)
    console.log(f"speedup cms manual: ", speedup_query_manual)
    console.log(f"cms lib freq: ", estimated_frequency_lib)
    console.log(f"cms manual freq: ", estimated_frequency_manual)
