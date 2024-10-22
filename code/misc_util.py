DEFAULT_DELIMITER = "_"

def set_global_delimiter(new_delimiter: str):
    global DEFAULT_DELIMITER
    DEFAULT_DELIMITER = new_delimiter

def string_join(*args, delimiter: str = None) -> str:
    if delimiter is None:
        delimiter = DEFAULT_DELIMITER
    return delimiter.join(args)

def misc_util_examples():
    # Example usage:
    base_dataset = "mydataset"
    base_dataset_proc_lvl = "lvl1"
    embedding_function_tag = "emb_v1"

    data_set_name = string_join(base_dataset, base_dataset_proc_lvl, embedding_function_tag)
    print(data_set_name)  # Output: mydataset_lvl1_emb_v1

    # Changing global delimiter
    set_global_delimiter("-")
    data_set_name_with_dash = string_join(base_dataset, base_dataset_proc_lvl, embedding_function_tag)
    print(data_set_name_with_dash)  # Output: mydataset-lvl1-emb_v1


# text_embedding_service/target_object.py
class TargetObject:
    def __init__(self, id, text, current_embedding, current_embedding_model, metadata={}):
        self.id = id
        self.text = text
        self.current_embedding = current_embedding
        self.current_embedding_model = current_embedding_model
        self.metadata = metadata

import os
import json
import gzip
import pickle


def detailed_display_query_results(target_object, top_similar_entries, text_embedding_service):
    print("By Query Results")
    print(f"Current Query id: {target_object.id}")
    print(f"Current Query Text: {target_object.text}")
    print(f"Distance Metric: {text_embedding_service.metric}")
    for index, entry_id in enumerate(top_similar_entries['ids'][0]):
        entry_text = top_similar_entries['documents'][0][index]
        truncate_len = 50
        delta = truncate_len - len(entry_text)
        truncated_entry_text = entry_text[:50] if delta <= 0 else entry_text + (' ' * delta)
        entry_distance = top_similar_entries['distances'][0][index]
        entry_embedding = top_similar_entries['embeddings'][0][index]
        print(f"\tIndex: {index} Distance: {entry_distance:.3f} ID: {entry_id} Text: {truncated_entry_text}\tEmbedding: {entry_embedding[:5]}")


# Function to load a document by its ID from the corpus
def get_document_from_corpus(doc_id, corpus_directory):
    """
    Retrieves the document content for a given document ID from the MS MARCO corpus.
    
    Args:
    - doc_id (str): The ID of the document to retrieve.
    - corpus_directory (str): The path to the directory containing the corpus.
    
    Returns:
    - str: The content of the document if found, else an empty string.
    """
    try:
        # Parse document ID to locate specific file and position
        (string1, string2, bundlenum, position) = doc_id.split('_')
        assert string1 == 'msmarco' and string2 == 'doc'
        
        # Locate the appropriate gzipped JSONL file in the corpus directory
        corpus_file = os.path.join(corpus_directory, f'msmarco_doc_{bundlenum}.gz')
        
        if os.path.exists(corpus_file):
            with gzip.open(corpus_file, 'rb') as in_fh:  # Read the gzipped file in binary mode
                in_fh.seek(int(position))
                json_string = in_fh.readline().decode('utf-8', errors='replace')  # Decode with error handling
                document = json.loads(json_string)
                if document['docid'] == doc_id:
                    return document['body']
    except Exception as e:
        print(f"Error retrieving document {doc_id}: {e}")
    
    return ""

def load_or_create_pickle(pickle_path, function, *args, **kwargs):
    """
    Checks if a pickle file exists at the specified path. If it exists, load and return the pickled object.
    Otherwise, call the provided function with the provided arguments, pickle the result, and return it.

    :param pickle_path: str, the path to the pickle file
    :param function: callable, the function to call if the pickle file does not exist
    :param args: arguments for the function
    :param kwargs: keyword arguments for the function
    :return: the loaded or newly created object
    """
    # Check if the pickle file exists
    if os.path.exists(pickle_path):
        # Load and return the object from the pickle file
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    else:
        # Call the function to create the object
        result = function(*args, **kwargs)
        
        # Pickle the result for future use
        with open(pickle_path, 'wb') as f:
            pickle.dump(result, f)
        
        # Return the result
        return result