import time
import functools
import random

from datasketch import MinHash, MinHashLSH
import pandas as pd


def timer(func):
    """A decorator that prints the execution time of the function."""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"{func.__name__!r} took {elapsed_time:.4f} seconds")
        return value

    return wrapper_timer


def create_minhash(text, num_perm=128):
    """
    Create a MinHash object for a given piece of text.
    `num_perm` specifies the number of permutation functions used in MinHash algorithm.
    """
    m = MinHash(num_perm=num_perm)
    for d in text.split(" "):  # Assuming whitespace tokenization is sufficient
        m.update(d.encode("utf8"))
    return m


@timer
def find_duplicates(texts, num_perm=128, threshold=0.95):
    """
    Find and group near-duplicate texts using MinHash and LSH.
    Returns a list of lists, where each sublist contains the indices of near-duplicate texts.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = {}

    # Create MinHash objects and insert them into the LSH index
    for i, text in enumerate(texts):
        m = create_minhash(text, num_perm=num_perm)
        lsh.insert(f"text_{i}", m)
        minhashes[f"text_{i}"] = m

    # Find near-duplicate groups using the LSH query
    duplicate_groups = []
    processed_keys = (
        set()
    )  # Keep track of processed keys to avoid duplicates in the output
    for key, minhash in minhashes.items():
        if key in processed_keys:
            continue
        result = lsh.query(minhash)
        if len(result) > 1:  # Ensure group has more than one item
            duplicate_groups.append(result)
            processed_keys.update(result)

    # Convert keys in groups to original indices
    groups_indices = [
        [int(k.split("_")[1]) for k in group] for group in duplicate_groups
    ]

    return groups_indices


def remove_duplicates(file_path: str = "output/synthetic/train-00000-of-00001.parquet"):
    main = pd.read_parquet(file_path, engine="pyarrow")
    texts = main["tweet"].astype(str).tolist()

    duplicate_groups = find_duplicates(texts)

    # Initialize an empty set to keep track of indices to drop
    indices_to_drop = set()

    # Iterate through each group, randomly select an index to keep, and add the rest to the indices_to_drop set
    for group in duplicate_groups:
        keep_index = random.choice(group)
        indices_to_drop.update(i for i in group if i != keep_index)

    print("Dropping:", len(indices_to_drop))

    # Drop the rows from the main DataFrame in one operation
    # Note: Use .iloc for position-based indexing if 'group' indices are position-based from the texts list
    # If 'group' indices correspond to DataFrame's index, use .loc or .drop with the index directly
    main_filtered = main.drop(main.index[indices_to_drop])

    # If needed, reset index
    main_filtered.reset_index(drop=True, inplace=True)

    # Save the filtered DataFrame to a new parquet file
    main_filtered.to_parquet(
        f"{file_path.split('.')[0]}_filtered.parquet", engine="pyarrow"
    )


def compare_to_og():
    # Download the original dataset
    og = pd.read_parquet("data/finetune/finetuning_data.parquet", engine="pyarrow")
    og_texts = og["tweet"].astype(str).tolist()

    # Get the filtered dataset
    main = pd.read_parquet(
        "output/synthetic/train-00000-of-00001_filtered.parquet", engine="pyarrow"
    )
    filtered_texts = main["tweet"].astype(str).tolist()

    # Merge into a single list
    all_texts = og_texts + filtered_texts

    # Find duplicates in the merged list
    duplicate_groups = find_duplicates(all_texts)

    indices_to_drop = set()

    for group in duplicate_groups:
        # Adjust indices for the filtered dataset part
        indices_to_drop.update(i - len(og_texts) for i in group if i >= len(og_texts))
        for i in group:
            print(all_texts[i])

    print("Dropping:", len(indices_to_drop))

    # Convert indices_to_drop to boolean mask for iloc
    mask = [i not in indices_to_drop for i in range(len(filtered_texts))]

    # Apply mask to filter out duplicates
    main_filtered = main.iloc[mask]

    # Reset index if needed
    main_filtered.reset_index(drop=True, inplace=True)

    # Save the filtered DataFrame to a new parquet file
    main_filtered.to_parquet(
        "output/synthetic/train-00000-of-00001_filtered2.parquet", engine="pyarrow"
    )
