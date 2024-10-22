
# ch_module.py
# Corpus Hander Code...
import os
import re
import gzip
import json

# Early stopping constants for controlling processing
EARLY_FILE_STOP = None  # Set to an integer to stop after parsing this many files
EARLY_DOC_STOP = None  # Set to an integer to stop after parsing this many documents per file

corpus_dir = "./data/msmarco/corpus/msmarco_v2_doc"  # Update with your local path

# Function to return the currently set early stopping values
def get_early_stopping_values():
    return {
        "EARLY_FILE_STOP": EARLY_FILE_STOP,
        "EARLY_DOC_STOP": EARLY_DOC_STOP
    }

# Function to get filenames from a directory
def get_filenames_from_directory(path):
    try:
        filenames = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
        return filenames
    except FileNotFoundError:
        print(f"Error: The directory '{path}' does not exist.")
        return []
    except PermissionError:
        print(f"Error: You do not have permission to access '{path}'.")
        return []

# Function to sort filenames by bundle number
def sort_filenames_by_bundlenum(filenames):
    sorted_filenames = sorted(filenames, key=lambda filename: int(re.search(r'\d+', filename).group()))
    return sorted_filenames

# Generator function to yield filenames in sorted order
def get_next_sorted_filename(path):
    filenames = get_filenames_from_directory(path)
    sorted_filenames = sort_filenames_by_bundlenum(filenames)
    for idx, filename in enumerate(sorted_filenames):
        if EARLY_FILE_STOP is not None and idx >= EARLY_FILE_STOP:
            break
        yield filename

# Function to parse documents from a corpus file
def parse_documents_from_corpus(corpus_file):
    documents = []
    try:
        with gzip.open(corpus_file, 'rb') as in_fh:
            for idx, json_string in enumerate(in_fh):
                if EARLY_DOC_STOP is not None and idx >= EARLY_DOC_STOP:
                    break
                document = json.loads(json_string.decode('utf-8', errors='replace'))
                documents.append(document)
    except FileNotFoundError:
        print(f"Error: The file '{corpus_file}' does not exist.")
    except OSError as e:
        print(f"Error: An error occurred while trying to read the file '{corpus_file}'. Details: {e}")
    return documents

# Generator to yield documents from a corpus file
def get_next_document_fm_corpusfile(corpus_file):
    document_list = parse_documents_from_corpus(corpus_file)
    for document in document_list:
        yield document

# Function to continuously load documents from all files in a directory
def continuous_doc_data_fm_corpus(corpus_directory):
    next_filename_generator = get_next_sorted_filename(corpus_directory)
    for next_filename in next_filename_generator:
        corpus_file = os.path.join(corpus_directory, next_filename)
        next_doc_generator = get_next_document_fm_corpusfile(corpus_file)
        for next_doc in next_doc_generator:
            print(next_doc['docid'])

# Function to continuously load documents from all files in a directory and apply a given function to each document
def continuous_doc_data_apply_func(corpus_directory, func, *args, **kwargs):
    next_filename_generator = get_next_sorted_filename(corpus_directory)
    for next_filename in next_filename_generator:
        corpus_file = os.path.join(corpus_directory, next_filename)
        next_doc_generator = get_next_document_fm_corpusfile(corpus_file)
        for next_doc in next_doc_generator:
            func(msmarco_doc=next_doc, *args, **kwargs)

if __name__ == "__main__":
    # Example usage with early stopping
    EARLY_FILE_STOP = 2  # Stop after processing 2 files
    EARLY_DOC_STOP = 5   # Stop after processing 5 documents per file
    
    # Example usage of get_early_stopping_values
    print(get_early_stopping_values())

    # Example function to use with continuous_doc_data_apply_func
    def example_function(msmarco_doc, additional_info=None):
        print(f"DocID: {msmarco_doc['docid']}, Additional Info: {additional_info}")
    
    continuous_doc_data_apply_func(corpus_dir, example_function, additional_info="Example")
