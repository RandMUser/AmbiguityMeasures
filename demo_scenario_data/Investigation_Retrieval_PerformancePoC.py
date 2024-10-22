# %% [markdown]
# The goal of the Dense Retriever environment is to demonstrate the full lifecycle of a search system implemented as dense retriever with a single static text embedding model to pre-embed all candidate texts from the corpus, and calculate/return the nearest-k documents (embeddings) to a query that is embedded on-demand (real-time user input/output).

# %% [markdown]
# Step 1. Configure environment to load raw text from the available corpus. msmarco...

# %%
# Define paths (update these paths to the correct local paths)
corpus_dir = "./data/msmarco/corpus/msmarco_v2_passage"

import ch_module
GLOBAL_TEST_EARLY_STOP_FILES = None
GLOBAL_TEST_EARLY_STOP_DOCUMENTS = None
ch_module.EARLY_DOC_STOP = GLOBAL_TEST_EARLY_STOP_DOCUMENTS
ch_module.EARLY_FILE_STOP = GLOBAL_TEST_EARLY_STOP_FILES

# %% [markdown]
# Step 2. Configure the vector database with embedding model distance metric and setup experiment varibles.

# %%
import misc_util as mu
from vh_module import TextEmbeddingService, TargetObject, SpacyWord2VecEmbeddingFunction

# Setup vector store with the embedding model and distance metric for retrieval.

# Configuration elements for settung up and persisting data for / from the experiment.
# 1
chroma_db_experiment_content = 'corpus'
# 2
base_dataset = "msmarco"  # Replace with your dataset name
# 3
base_dataset_proc_lvl = "passages" # could also be passages, or other passage / sentence level processing like done with pysereni...
# 4
embedding_function = SpacyWord2VecEmbeddingFunction
#embedding_model_tag = "spaCy" # used as a key to a dictionary somewhere else in Stremlined Demo Environment...??? Probably need to merge with embedding_function_tag below to be more precise and to streamline the number of redundant variables...
embedding_function_tag = 'spaCyW2V'
vectordb_distance_metric="cosine"
# 5
corpus_earlystopping_scope = ch_module.get_early_stopping_values()
# 6
mu.set_global_delimiter("_")
chroma_data_set_name = mu.string_join(chroma_db_experiment_content, base_dataset, base_dataset_proc_lvl, embedding_function_tag, vectordb_distance_metric, str(corpus_earlystopping_scope['EARLY_FILE_STOP']), str(corpus_earlystopping_scope['EARLY_DOC_STOP']))
# 7 
# Final file paths and filenames.
vectordb_persist_path = "./persisted_vector_store/" + chroma_data_set_name + ".chromadb"
calc_persist_path = "./persisted_vector_store/" + chroma_data_set_name + "_persisted_calculated_data/pickle_"
# 8
# Create the object holding the configured vector databse.
corpus_embedding_service = TextEmbeddingService(data_set_name=chroma_data_set_name, embedding_model=embedding_function_tag, embedding_function=embedding_function, metric=vectordb_distance_metric, persist_path=vectordb_persist_path)

# %% [markdown]
# Step 2. Process the text into a vector database that supports retrieval with a distance metric of choice...

# %%
# I need to create a function or use one from the text embedding service package to pass into ch_module.continuous_doc_data_apply_func which will load the text and generate embeddings for the vector database.
def embed_and_insert(msmarco_doc, corpus_embedding_service):
    # Need to convert the msmarco_doc into text_embedding_service.target_object if i use add_entry --     corpus_embedding_service.add_entry()
    # Accessing the chromadb directly to save this step...
    msmarco_passage = msmarco_doc
    source = mu.string_join(base_dataset, base_dataset_proc_lvl)
    corpus_embedding_service.collection.add(ids=[msmarco_passage['pid']], metadatas=[{'text_source':source}], documents=[msmarco_passage['passage']])

# May need to modify something in corpus_embedding_service --> text_embedding_service to allow for saving only the vectors in the database for a corpus. 
# This may be necessary since it will be ~30GB for all docs (because that's how big the full .gz corpus files are?) plus the embedings and any metadata included with the vectors.
# I could possibly create a "Master db" to quickly reference the plain text if needed...
ch_module.continuous_doc_data_apply_func(corpus_dir, embed_and_insert, corpus_embedding_service = corpus_embedding_service)

# %% [markdown]
# Step 3. Enter user queries for retrieval and save the top-k documents returned by the dense retriever (search system).

# %%
import pandas as pd
# Define dataset paths (update these paths to the correct local paths)
base_dataset = "msmarco"  # Replace with your dataset name
data_split_pfn = "passv2_dev2_"
query_data_static_pfn= "queries.tsv"
local_data_path = "./data/"
queries_path = local_data_path + base_dataset + "/" + data_split_pfn + query_data_static_pfn
# Load queries data
queries = pd.read_csv(queries_path, sep='\t', names=["qid", "query"], dtype={"qid": str, "query": str})
queries.head()

# Loop through the queries and collect the top-k for scoring...

# Run file conversion script: https://github.com/castorini/pyserini/blob/master/pyserini/eval/convert_msmarco_run_to_trec_run.py
# MARCO run file format?
# TREC run file format?
# DPR retrieval result json

# Kilt Retrieval ? https://github.com/facebookresearch/KILT/tree/9bcb119a7ed5fda88826058b062d0e45c726c676


# %% [markdown]
# Step 4. Score the top-k documents returned by the dense retriever (search system) using the qrel data (relevance labels for at least one relevant document in the corpus).

# %%
def save_trec_run_file(top_k_search_results_by_qid, trec_run_filepath):
    """
    This function processes the dictionary 'top_k_search_results_by_qid' 
    and writes it to a file in the TREC format. The output file is compatible 
    with the trec_eval tool.
    
    TREC Format: 
    qid Q0 doc_id rank score run_name
    
    Args:
        top_k_search_results_by_qid (dict): A dictionary containing search results,
                                            where the keys are query IDs (qid), 
                                            and the values are dictionaries with:
                                                - 'ids': List of document IDs.
                                                - 'distances': List of scores.
        trec_run_filepath (str): The file path to save the TREC-formatted run file.
    """
    try:
        with open(trec_run_filepath, 'w') as f:
            for qid, results in top_k_search_results_by_qid.items():
                doc_ids = results['ids'][0]
                scores = results['distances'][0]

                for rank, (doc_id, score) in enumerate(zip(doc_ids, scores)):
                    # Writing in TREC format: qid Q0 doc_id rank score run_name
                    run_name = "text_embedding_run"  # You can modify this to reflect your experiment name
                    line = f"{qid} Q0 {doc_id} {rank + 1} {score:.4f} {run_name}\n"
                    f.write(line)
        print(f"TREC run file saved at: {trec_run_filepath}")
    except Exception as e:
        print(f"An error occurred while saving the TREC run file: {e}")

def generate_searcher_run_data(queries, searchsystem, trec_run_filepath):
    corpus_embedding_service = searchsystem
    top_k_search_results_by_qid = {}
    for item in queries.index:
        qid = queries["qid"][item]
        q_text = queries["query"][item]
        target_query = TargetObject(qid, q_text, current_embedding=None, current_embedding_model=None, 
                                    metadata={"doc_v2_devQueries_StreamlinedDemo"})
        
        top_k_search_results = corpus_embedding_service.text_search(q_text)
        mu.detailed_display_query_results(target_object=target_query, top_similar_entries=top_k_search_results, 
                                       text_embedding_service=corpus_embedding_service)
        top_k_search_results_by_qid[qid] = top_k_search_results
    
    save_trec_run_file(top_k_search_results_by_qid, trec_run_filepath)
    return top_k_search_results_by_qid
'''
queries = pd.DataFrame({
    "qid": [1, 2],
    "query": ["example query 1", "example query 2"]
})
'''

searchsystem = corpus_embedding_service  # Assuming this is your search system instance
top_k_search_results_by_qid = generate_searcher_run_data(queries, searchsystem, "results.trec")


# %%
import subprocess

def run_trec_eval_and_summary(qrel_filepath, run_filepath):
    """
    Runs the trec_eval shell script and prints a summary of the results.

    Args:
        qrel_filepath (str): Path to the qrel file (ground truth relevance judgments).
        run_filepath (str): Path to the run file generated from the search system.
    """
    try:
        # Execute the shell script and capture output
        result = subprocess.run(
            ['./run_trec_eval.sh', qrel_filepath, run_filepath],
            capture_output=True, text=True, check=True
        )
        
        # Print the full output for reference
        print("===== Full TREC Evaluation Output =====")
        print(result.stdout)

        # Parse and display key metrics (already handled in the shell script)
        print("===== Summary of Results =====")
        print(result.stdout.splitlines()[-10:])  # Last few lines for quick review

    except subprocess.CalledProcessError as e:
        print(f"Error executing trec_eval: {e.stderr}")

    except FileNotFoundError:
        print("Error: Shell script 'run_trec_eval.sh' not found or not executable.")

# Example usage
# Ensure you provide valid file paths for the qrel and run files
#qrel_filepath = "path/to/qrels.txt"

run_filepath = "results.trec"
qrel_data_static_pfn= "qrels.tsv"
qrel_path = local_data_path + base_dataset + "/" + data_split_pfn + qrel_data_static_pfn
qrel_filepath = qrel_path
# Load qrels data
qrels_data = pd.read_csv(qrel_path, sep='\t', names=["qid", "unknown", "docid", "rel"], dtype={"qid": str, "docid": str, "rel": int})


run_trec_eval_and_summary(qrel_filepath, run_filepath)


# %%
import pandas as pd
import numpy as np

def dcg(scores, k):
    """
    Compute Discounted Cumulative Gain (DCG) up to rank k.
    Args:
        scores (list): List of relevance scores in ranked order.
        k (int): Rank position up to which DCG is calculated.
    Returns:
        float: DCG score.
    """
    scores = np.array(scores)[:k]
    return np.sum((2 ** scores - 1) / np.log2(np.arange(2, scores.size + 2)))

def ndcg_at_k(retrieved_docs, relevant_docs, k):
    """
    Compute the Normalized Discounted Cumulative Gain (NDCG) at rank k.
    Args:
        retrieved_docs (list): List of retrieved document IDs.
        relevant_docs (dict): Dictionary of relevant document IDs and their relevance scores.
        k (int): Rank position for the NDCG calculation.
    Returns:
        float: NDCG score at rank k.
    """
    # Create the list of relevance scores in the order of retrieved_docs
    scores = [relevant_docs.get(doc, 0) for doc in retrieved_docs]
    
    # Calculate DCG for the given ranking
    dcg_score = dcg(scores, k)

    # Calculate ideal DCG (IDCG) with the perfect ranking
    ideal_scores = sorted(relevant_docs.values(), reverse=True)
    idcg_score = dcg(ideal_scores, k)

    # Avoid division by zero
    return dcg_score / idcg_score if idcg_score > 0 else 0.0

def calculate_ndcg_for_queries(top_k_search_results_by_qid, qrels_data, k=10):
    """
    Calculate NDCG@k for each query in the search results.
    Args:
        top_k_search_results_by_qid (dict): Search results with query IDs as keys.
        qrels_data (pd.DataFrame): DataFrame containing relevance judgments (qrels).
        k (int): Rank position for the NDCG calculation.
    Returns:
        dict: Dictionary with query IDs as keys and NDCG scores as values.
    """
    ndcg_scores = {}

    # Convert qrels data into a dictionary with qid -> {docid: relevance_score}
    qrels_dict = qrels_data.groupby("qid").apply(
        lambda x: dict(zip(x["docid"], x["rel"]))
    ).to_dict()

    # Calculate NDCG@k for each query
    for qid, search_result in top_k_search_results_by_qid.items():
        retrieved_docs = search_result["ids"][0]  # List of retrieved doc IDs
        relevant_docs = qrels_dict.get(qid, {})   # Relevant docs with scores

        # Compute NDCG for this query
        ndcg_score = ndcg_at_k(retrieved_docs, relevant_docs, k)
        ndcg_scores[qid] = ndcg_score

    return ndcg_scores

# Example Usage:
qrels_data = pd.read_csv(qrel_path, sep='\t', names=["qid", "unknown", "docid", "rel"],
                         dtype={"qid": str, "docid": str, "rel": int})

# Assuming 'top_k_search_results_by_qid' is already available from your search results
ndcg_scores = calculate_ndcg_for_queries(top_k_search_results_by_qid, qrels_data, k=10)

# Display NDCG scores
for qid, score in ndcg_scores.items():
    print(f"Query ID: {qid}, NDCG@10: {score:.4f}")


# %% [markdown]
# Step 5. ** Compare dense retriever performance (by query) with the calculated ambiguity measures for each query (see StreamlinedDemoEnvironment.ipynb). **

# %% [markdown]
# Consider the following example for building a script to execute the full experiment run:
# https://github.com/facebookresearch/KILT/blob/9bcb119a7ed5fda88826058b062d0e45c726c676/scripts/execute_retrieval.py
# 
# https://github.com/facebookresearch/KILT/blob/main/kilt/retrieval.py


