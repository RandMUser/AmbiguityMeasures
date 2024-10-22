# %%
from misc_util import load_or_create_pickle

# %% [markdown]
# Step 1. Load raw queries to form the initial query pool.

# %%
import pandas as pd
# Define dataset paths (update these paths to the correct local paths)
base_dataset = "msmarco"  # Replace with your dataset name
data_split_pfn = "docv2_dev2_"
query_data_static_pfn= "queries.tsv"
local_data_path = "./data/"
queries_path = local_data_path + base_dataset + "/" + data_split_pfn + query_data_static_pfn
# Load queries data
queries = pd.read_csv(queries_path, sep='\t', names=["qid", "query"], dtype={"qid": str, "query": str})
queries.head()

# %% [markdown]
# Step 2. Setup the pre-trained embedding model of interest / for evlauation.
# -- Factors:
# ---- Frozen / pre-trained text embedding model (Fixed parameters trained using an objective and specific training observations/data)
# ---- Metric selected for calculating distance between vectors.

# %%
from vh_module import TextEmbeddingService, TargetObject, SpacyWord2VecEmbeddingFunction
# Setup vector store with the embedding model and distance metric for retrieval.
embedding_function = SpacyWord2VecEmbeddingFunction
data_set_name = data_split_pfn + query_data_static_pfn
vectordb_persist_path = "./persisted_vector_store/" + data_split_pfn + query_data_static_pfn + ".chromadb"
calc_persist_path = "./persisted_vector_store/" + "persisted_calculated_data/pickle_"

query_embedding_service = TextEmbeddingService(data_set_name=data_set_name, embedding_model="spaCy", embedding_function=embedding_function, metric="cosine", persist_path=vectordb_persist_path)

# %% [markdown]
# Step 3. Calculate text embeddings for all queries in the initial query pool.

# %%
# TODO: Add check to see if the vector databse in query_embedding_service contains all of the embeddings already since it is persisted... Do not loop through queries.index and 'query_embedding_service.add_entry(entry)' if the vector database is already populated.
# iterate through each row and select qid and query for processing into the vector database
def populate_query_embedding_service_vectordb(query_embedding_service, queries):
    current_collection_count = query_embedding_service.collection.count()
    if (current_collection_count < len(queries)) or (query_embedding_service.entries == []):
        query_embedding_service.purge_current_collection()
        query_embedding_service.entries = []
        for ind in queries.index:
            entry = TargetObject(id=queries['qid'][ind], text=queries['query'][ind], current_embedding=None, current_embedding_model="spaCy", metadata={"source": data_split_pfn + query_data_static_pfn})
            query_embedding_service.add_entry(entry)
    else:
        pass# The vector db should be populated from the persisted databse at the provided path.
    return query_embedding_service.entries
#populate_query_embedding_service_vectordb(query_embedding_service, queries)

pickle_file_path = vectordb_persist_path + "_query_embedding_service.entries"
result = load_or_create_pickle(pickle_file_path, populate_query_embedding_service_vectordb, query_embedding_service, queries)
query_embedding_service.entries = result


# %% [markdown]
# Step 4. Estimate the semantic region around each query by identifying 'similar' queries from the pool.
# -- Factors:
# ---- Method of selecting 'similar' queries that represent the likely semantic region for each query.
# ------ Opt 1. For each query, locate the nearest k queries in the pool of all queries. (This may not yield 'symmetric'? 'isolated'? 'non-overlapping'? sets for each query or group of queries.)
# ------ Opt 1a. For each query, locate the nearest k queries in the pool of all queries. Filter out 'distant' queries with a threshold on the distance measure. This may addres a situation where the embedded query text is unique in the pool of queries being embedded. In this case, the populated embedding space around the query is sparse for the given data (query pool). (This may not yield 'symmetric'? 'isolated'? 'non-overlapping'? sets for each query or group of queries. Specifically, when an excluded query may be near other queries in the nearest k but just outside the selected threshold value.)
# ------ Opt 2. A range of clustering techniques... Assign labels to queries based upon the solution produced with an unsupervised vector clustering algorithm. All queries and associated vectors are then associated with a unique and non-overlapping QSR value.
# 
# ---- Method of assigning the identified semantic region and associated measures to each query.
# ------ Opt 1. Each query potentially has a unique qsr based upon it's nearest k...
# ------ Opt 2. Each query potentially has a unique qsr based upon it's nearest k with threshold...
# ------ Opt 3. Somehow assign a single qsr to multiple related queries, where each query is only considerd once in the final set of QSRs...

# %%
# TODO: Add check to see if the query_embedding_service.entries[index].metadata["querypool_nearest_k_results"] is populated already to avoid running 'query_embedding_service.text_search(entry.text, n=query_nearest_k)' again...
query_nearest_k = 10

def gather_all_nearest_k(query_embedding_service):
    for index, entry in enumerate(query_embedding_service.entries):
        if "querypool_nearest_k_results" not in query_embedding_service.entries[index].metadata:
            #print(entry.id, entry.text)
            #TODO: Maybe I should minimize the redundant data brought back from query_embedding_service.text_search()?
            # querypool_nearest_k_results will contain the vectors, text, IDs, etc, which is redundant... The ID and distance is probably all that is needed here. The text or vector can be pulled back with the ID if needed later also.
            # for 5000 queries the list will now contain the original 5000 full texts with 10 each full texts plus embedding, plus++ in results if k = 10...
            # Execution time was not too bad for 5k though ~3min
            querypool_nearest_k_results = query_embedding_service.text_search(entry.text, n=query_nearest_k)
            query_embedding_service.entries[index].metadata["querypool_nearest_k_results"] = querypool_nearest_k_results
        else:
            pass #The nearest k data is alredy saved for this entry.
    return query_embedding_service.entries

def populate_query_embedding_service_entries_with_nearest_k(query_embedding_service):    
    if len(query_embedding_service.entries) > 0:
        if "querypool_nearest_k_results" not in query_embedding_service.entries[0].metadata:        
            #Try to load a pickeled object for the vectordb in this query_embedding_service...
            pickle_file_path = vectordb_persist_path + "_query_embedding_service.entries_nearest_k"
            result = load_or_create_pickle(pickle_file_path, gather_all_nearest_k, query_embedding_service)
            query_embedding_service.entries = result
        else:
            assert(0==1)# Something may be wrong. Why would we call populate_query_embedding_service_entries_with_nearest_k() if we have values already in query_embedding_service.entries?
    else:
        assert(0==1)# Something is wrong. There should be entries already...


populate_query_embedding_service_entries_with_nearest_k(query_embedding_service)

# %%
from misc_util import detailed_display_query_results

# %%
for index, entry in enumerate(query_embedding_service.entries):
    querypool_nearest_k_results = query_embedding_service.entries[index].metadata["querypool_nearest_k_results"]
    detailed_display_query_results(entry,querypool_nearest_k_results,query_embedding_service)
    if index > 2:
        break

# %% [markdown]
# 
# Step 4x. Assess the QSR with some measurements?
# 

# %% [markdown]
# Step 5. Estimate the document space semantic region coresponding to the QSR selected for each query. We choose only one qrel (relevant) labeled document (the first identified) for each query in the QSR to populate the DSR.

# %%
qrel_data_static_pfn= "qrels.tsv"
qrel_path = local_data_path + base_dataset + "/" + data_split_pfn + qrel_data_static_pfn
# Load qrels data
qrels_data = pd.read_csv(qrel_path, sep='\t', names=["qid", "unknown", "docid", "rel"], dtype={"qid": str, "docid": str, "rel": int})

# Function to get relevant document IDs for a given query ID using qrels data
def get_qid_qrel_doc_ids(qid):
    """
    Retrieves the set of document IDs that are relevant to a given query ID based on the qrels data.
    
    Args:
    - qid (str): The query ID to find relevant documents for.
    - qrels_data (DataFrame): The qrels DataFrame containing relevance information.
    
    Returns:
    - list: A list of relevant document IDs.
    """
    qid_qrel_docids = qrels_data[qrels_data['qid'] == qid]['docid'].tolist()
    qid_qrel_rel_labels = qrels_data[qrels_data['qid'] == qid]['rel'].tolist()
    qid_qrel_docid_tuples = list(zip(qid_qrel_docids,qid_qrel_rel_labels))
    return qid_qrel_docid_tuples

def get_all_relevant_doc_ids(qid):
    rel_docids = []
    qid_qrel_docid_tuples = get_qid_qrel_doc_ids(qid)
    for element in qid_qrel_docid_tuples:
        if element[1] == 1:
            rel_docids.append(element[0])
    return rel_docids

def get_first_relevant_doc_id(qid):
    rel_docids = get_all_relevant_doc_ids(qid)
    if len(rel_docids) == 0:
        return rel_docids #same as return []
    elif len(rel_docids) == 1:
        return rel_docids #same as return [rel_docids[0]] which will be list of length 1 with the docid string in it.
    else:
        return [rel_docids[0]] # the return value will be a list of length 1 with the docid string in it.

# %%
#TODO: Look at the effenciency of this code. ~14 Minutes to process Step 5 for 10 queries.
from misc_util import get_document_from_corpus
from chromadb.utils.distance_functions import cosine as chroma_cosine_distance

# Requires local_data_path defined above
corpus_dir = "./data/msmarco/corpus/msmarco_v2_doc" # Update with your local path to the massive corpus data files (~30GB)

def gather_DSR_doc_ids(querypool_nearest_k_results):
    # Get each qid that forms the QSR for the current query from querypool_nearest_k_results["ids"][0]
    DSR_doc_ids_by_QSR_qid = {} #Key will be the qid, value will be the doc id; this is the QSR neaearest k qid, not the entry qid...

    for qid in querypool_nearest_k_results["ids"][0]:
        # Get a relevant doc for the QSR forming qid.
        relevant_doc_id_list = get_first_relevant_doc_id(qid)
        #TODO: Maybe store this relevant_doc_id somewhere for future reference analysis when validating the results? For example, we may want to know if qids in the QSR share the same qrel doc. This could help explain something observed in the resluting calculated Ambiguity Measure.
        print(relevant_doc_id_list)
        assert(len(relevant_doc_id_list)==1) # We want at least 1 rel document per qid/E(query) in the QSR we are estimating.
        DSR_doc_ids_by_QSR_qid[qid] = relevant_doc_id_list
    return DSR_doc_ids_by_QSR_qid

def collapse_validate_DSR_doc_ids(DSR_doc_ids_by_QSR_qid):
    DSR_doc_ids = []
    for qid in DSR_doc_ids_by_QSR_qid:
        relevant_doc_id_list = DSR_doc_ids_by_QSR_qid[qid]
        # We will now collapse the list (relevent_doc_id_list) into DSR_doc_ids so we don't have a list of lists, but a list of docunent Id strings.
        for item in relevant_doc_id_list:
            DSR_doc_ids.append(item)
    print(len(DSR_doc_ids))
    assert(len(DSR_doc_ids)==query_nearest_k) # check that we have 1 qrel document per query to estimate the DSR. Need to change query_nearest_k if we use another option to select queries/number of queries in the QSR. 
    # Need to process this list of doc ids 'QSR_doc_ids' by getting the doc text, and embedding the text so we can make measurements within the embedding space for the DSR.
    return DSR_doc_ids

def create_DSR_data_in_QES_entries(query_embedding_service):
    # Get the qrel docs for each query - QSR...
    # entry contains the details of a query that we have QSR estimating data stored/persisted in query_embedding_service.entries[index].metadata["querypool_nearest_k_results"].
    for index, entry in enumerate(query_embedding_service.entries):
        #TODO: remove this early break after code is complete to allow for a full data run.
        if index > 2:
            break
        # TODO: I need to preserve, maybe pickle then restore, maybe stuff in the vector database, the contents of query_embedding_service.entries to suppport this functionality without having to re-process the queries back through the query_embedding_service if the ipynb environment is reset.
        querypool_nearest_k_results = query_embedding_service.entries[index].metadata["querypool_nearest_k_results"]
        #What is the type of querypool_nearest_k_results? If it is a dict, then we can add a new key/keys to hold the qrel docids, and correspinding doc embedding vectors we are getting to estimate the DSR.
        DSR_unique_doc_texts = {} # Key will be the document id, value will be the document text
        DSR_unique_doc_embeddings = {} # Key will be the document id, value will be the document embedding vector
        DSR_unique_doc_ids = []
        DSR_doc_ids = []
        DSR_doc_ids_by_QSR_qid = {} #Key will be the qid, value will be the doc id; this is the QSR neaearest k qid, not the entry qid...


        # Get each qid that forms the QSR for the current query from querypool_nearest_k_results["ids"][0]
        """for qid in querypool_nearest_k_results["ids"][0]:
            # Get a relevant doc for the QSR forming qid.
            relevant_doc_id_list = get_first_relevant_doc_id(qid)
            #TODO: Maybe store this relevant_doc_id somewhere for future reference analysis when validating the results? For example, we may want to know if qids in the QSR share the same qrel doc. This could help explain something observed in the resluting calculated Ambiguity Measure.
            print(relevant_doc_id_list)
            assert(len(relevant_doc_id_list)==1) # We want at least 1 rel document per qid/E(query) in the QSR we are estimating.
            DSR_doc_ids_by_QSR_qid[qid]=relevant_doc_id_list
            # We will now collapse the list (relevent_doc_id) into DSR_doc_ids so we don't have a list of lists, but a list of docunent Id strings.
            for item in relevant_doc_id_list:
                DSR_doc_ids.append(item)
            # Need to process this list of doc ids 'QSR_doc_ids' by getting the doc text, and embedding the text so we can make measurements within the embedding space for the DSR. """

        DSR_doc_ids_by_QSR_qid = gather_DSR_doc_ids(querypool_nearest_k_results)
        DSR_doc_ids = collapse_validate_DSR_doc_ids(DSR_doc_ids_by_QSR_qid)

        # Make relevant_doc_ids into a set to drop duplicates...
        # We can see how many unique docs there are... If multiple queries are related by qrel to a single document then it would suggest they are certianly "related" by that fact which supports the hypothesis that they have this other relationship based on the notion of QSR.
        DSR_unique_doc_ids = list(set(DSR_doc_ids))
        assert(len(DSR_doc_ids)==len(DSR_unique_doc_ids)) # Need to think about the implications of this condition if we see it in practice... I will craft an example of this in the demo case data.

        # Now collect the document text and manually compute the corresponding embedding vector with the embedding function for our the fixed embedding model being inspected.
        for item in DSR_unique_doc_ids:
            doc_text = get_document_from_corpus(item, corpus_dir)
            #print(type(doc_text),doc_text)
            #TODO: probably need to use the MSMARCO passage data or similar since the documents are full of "junk text" formatting, etc from basic parsing of a raw webpage instead of a focused content paragraph that couold more easialy be found to be relevant or not-relevant to the given query.

            DSR_unique_doc_texts[item] = doc_text
            doc_embedding = query_embedding_service.embedding_function(doc_text)
            DSR_unique_doc_embeddings[item] = doc_embedding
        
        
        detailed_display_query_results(entry,querypool_nearest_k_results,query_embedding_service)

        # Create the DSR_Data dictionary
        DSR_Data = {
            'DSR_unique_doc_texts': DSR_unique_doc_texts,
            'DSR_unique_doc_embeddings': DSR_unique_doc_embeddings,
            'DSR_unique_doc_ids': DSR_unique_doc_ids,
            'DSR_doc_ids': DSR_doc_ids,
            'DSR_doc_ids_by_QSR_qid': DSR_doc_ids_by_QSR_qid,
        }
        query_embedding_service.entries[index].metadata["DSR_Data"] = DSR_Data

    return query_embedding_service.entries

def populate_query_embedding_service_entries_with_DSR_data(query_embedding_service):
    if len(query_embedding_service.entries) > 0:
        if "DSR_Data" not in query_embedding_service.entries[0].metadata:        
            #Try to load a pickeled object for the vectordb in this query_embedding_service...
            pickle_file_path = vectordb_persist_path + "_query_embedding_service.entries_DSR"
            result = load_or_create_pickle(pickle_file_path, create_DSR_data_in_QES_entries, query_embedding_service)
            query_embedding_service.entries = result
        else:
            # Something may be wrong. Why would we call populate_query_embedding_service_entries_with_nearest_k() if we have values already in query_embedding_service.entries?
            pass
    else:
        assert(0==1)# Something is wrong. There should be entries already...
        pass

populate_query_embedding_service_entries_with_DSR_data(query_embedding_service)


# %%
def display_DSR_Data(query_embedding_service):
    for index, entry in enumerate(query_embedding_service.entries):
        #TODO: remove this early break after code is complete to allow for a full data run.
        if index > 2:
            break
        querypool_nearest_k_results = query_embedding_service.entries[index].metadata["querypool_nearest_k_results"]
        DSR_Data = query_embedding_service.entries[index].metadata["DSR_Data"]

        #What is the type of querypool_nearest_k_results? If it is a dict, then we can add a new key/keys to hold the qrel docids, and correspinding doc embedding vectors we are getting to estimate the DSR.
        DSR_unique_doc_texts = {} # Key will be the document id, value will be the document text
        DSR_unique_doc_embeddings = {} # Key will be the document id, value will be the document embedding vector
        DSR_doc_ids_by_QSR_qid = {} #Key will be the qid, value will be the doc id; this is the QSR neaearest k qid, not the entry qid...

        DSR_unique_doc_texts = DSR_Data.get('DSR_unique_doc_texts', {})
        DSR_unique_doc_embeddings = DSR_Data.get('DSR_unique_doc_embeddings', {})
        DSR_doc_ids_by_QSR_qid = DSR_Data.get('DSR_doc_ids_by_QSR_qid', {})

        #TODO: remove this early break after code is complete to allow for a full data run.
        if index > 2:
            break
        # Maybe I can display this like I do with the top-k search results?
        # build a std_data_display_obj from the DSR data like querypool_nearest_k_results
        #detailed_display_query_results(entry,querypool_nearest_k_results,query_embedding_service)
        std_data_display_obj = {'ids':[[]],'documents':[[]],'distances':[[]],'embeddings':[[]]}
        for qid in querypool_nearest_k_results["ids"][0]:
            qrel_doc_ids = DSR_doc_ids_by_QSR_qid[qid]
            for qrel_doc_id in qrel_doc_ids:
                std_data_display_obj['ids'][0].append(qrel_doc_id)

                qrel_doc_text = DSR_unique_doc_texts[qrel_doc_id]
                std_data_display_obj['documents'][0].append(qrel_doc_text)

                tqid_db_data = query_embedding_service.collection.get(ids=[entry.id],include=['embeddings'])
                tqid_vector = tqid_db_data['embeddings'][0]
                qrel_doc_embedding = DSR_unique_doc_embeddings[qrel_doc_id]

                tqid_vector_DSR_doc_distance = chroma_cosine_distance(qrel_doc_embedding, tqid_vector)
                std_data_display_obj['distances'][0].append(float(tqid_vector_DSR_doc_distance))
                

                doc_embedding = DSR_unique_doc_embeddings[qrel_doc_id]
                std_data_display_obj['embeddings'][0].append(doc_embedding[0])
        
        print("Here are the DSR Documents for the given query:")
        detailed_display_query_results(entry,std_data_display_obj,query_embedding_service)
        
display_DSR_Data(query_embedding_service)





# %% [markdown]
# Step 6. With the DSR data organized for each query, we can experiment with a range of ideas to calculate new measures of "Ambiguity" within the embedding space that contains vector representations of queries and documents.

# %% [markdown]
# We hope to identify new measures that will predict the retrieval performance of a search system, or search system component that relies on the analyzed text embedding model as a "Dense Retriever". If we can predict "Dense Retriever" performance with a new measure, then we may be able to improve the embedding model and search system with this knowledge. By incorporating the intuition, or data generated with the performance predicting measure into a new training scheme, we would expect to improve future embedding models for the task of acting as a "Dense Retriever".
# 
# If no additional performance is gained from future embedding models trained with insight or data from a new performance predicting measure for the task of acting as a "Dense Retriever", the measure could still be useful as a measure of 'query ambiguity', or 'query difficulty'. In this case, the query and document data representing information in the search domain cannot be further 'disambiguated', or 'better arranged' in the embedding space due to the inherant ambiguity in the labeled data. For example, if the search data contains a set of query-document objects paired by known qrel relevant labels that meet the following criteria:
# q1 == q2 (the text in q1 is identical to the text in q2); E(q1) == E(q2) (the equality of E(q1) and E(q2) follows from the definition of the embedding function...); R(q1)=d1 and R(q2)=d2 (d1, d2 are known to be relavant to q1,q2 respectively by the labeled data 'qrels' shown as the function R.) If DISTANCE(E(d1),E(d2)) is large then there may be an issue with the query being highly ambiguous (high ambiguity in the sense that the semantic region for identical queries q1 and q2 is "wide", thus documents with a range of irrelevant topics may be found in this space...) Perhaps more precisely, if DISTANCE(E(d1),E(d2)) > (DISTANCE(E(q1),E(d1)) or DISTANCE(E(q2),E(d2)) or DISTANCE(E(q1),E(d2)) or DISTANCE(E(q2),E(d1)))?? Perhaps more precisely if d3 is known to be irrelevant to q1, q2, or both, and DISTANCE(E(q1),E(d3)) < DISTANCE(E(q1),E(d1))... Maybe try again with this with the manufactured demo data or some new synthetic wordnet data.

# %% [markdown]
# Step 6. 
# Measure 1 Ambiguity by Query Semantic Region (QSR) Relevant Document Mean Deviation (RDMD): A-QSR-RDMD...
# Measure 1 Ambiguity by Query Semantic Region (QSR) Relevant Document Max? Deviation (RDMD): A-QSR-RDMD... This query (q1) has higher ambiguity than the remaining QSR related queries if the distance between it's qrel document if further from the DSR centroid than all other documents in the DSR...???
# 
# What about pairwise distances between all documets d in the DSR?
# What about variance for all documents d in the DSR?
# How does DSR "size" realate to variance in the DSR?
# -- How can I efficiently calculate the volume or area bounded by the points given by the DSR vectors?
# 

# %% [markdown]
# In high-dimensional spaces, the Gram determinant (or square root of it) provides a way to measure the volume spanned by a set of vectors. While this concept generalizes the idea of "area" or "volume" from 2D and 3D, it is crucial for tasks like evaluating linear independence, diversity, or coverage of embeddings in machine learning and search systems.

# %%





# %% [markdown]
# Steps to Compute the Volume Efficiently:
# Form a Matrix 𝑉: If you have 8 vectors, each with 300 dimensions, create a matrix
# 𝑉 of size 8 × 300. Each row corresponds to one of your vectors.
# 
# Compute the Gram Matrix 
# 𝐺 = 𝑉𝑉^𝑇: 𝐺 will be an 8×8 matrix. 
# 
# This matrix encapsulates the dot products between all pairs of your original vectors, summarizing their relationships.
# 
# Compute the Determinant of the Gram Matrix:
# Calculate det(𝐺). 
# If the determinant is positive, take the square root to get the volume.
# 
# Handle Degeneracies: If your vectors are not linearly independent (i.e., they lie in a lower-dimensional subspace), the determinant will be zero, indicating that the volume in the higher-dimensional space is zero.

# %%
import numpy as np

def compute_gram_matrix(V):
    # Assume V is your matrix of 8 vectors, each with 300 dimensions
    #V = np.random.rand(8, 300)  # Replace with your actual data

    # Compute the Gram matrix
    G = np.dot(V, V.T)
    return G

def k_dim_parallelepiped_volume(G):
    # Compute the determinant of the Gram matrix
    det_G = np.linalg.det(G)

    # Compute the volume (if the determinant is non-negative)
    volume = np.sqrt(det_G) if det_G >= 0 else 0

    print(f"Volume of the parallelepiped: {volume}")
    return volume

def compute_centroid(embeddings):
    """
    Computes the centroid (average vector) of a list of embeddings.
    
    Args:
    - embeddings (list of numpy.ndarray): List of embedding vectors.
    
    Returns:
    - numpy.ndarray: The centroid vector.
    """
    return np.mean(embeddings, axis=0)

# %%
import statistics

calc_persist_path = "./persisted_vector_store/" + "persisted_calculated_data/pickle_"


Ambiguity_Measures = ['DSR_Components', 'QA_QSR_DSR_RDMeD', 'QA_QSR_DSR_RDVo', 'QA_QSR_DSR_RDVMD'] 
Ambiguity_Measures_Data = {} # Keys are ambiguity measure acronyms/names, values are dicts of the calculated measure for each query in the analyzed data 
#----- dicts of the calculated measure --------- dict keys are qids (str), dict values are the scalar ambiguity values from each qid associated DSR data (float).
#----- dict of the DSR components used for measures --------- dict keys are qids (str), dict values are dicts containing the DSR components for the indexed qid, by component name.
QA_QSR_DSR_RDMeD = {} # dict keys are qids (str), dict values are the scalar ambiguity values from each qid associated DSR data (float).
#----- measure quantity / value for the qid key

DSR_Components = ['deviations_list', 'mean_deviation_sc', 'centroid_vec', 'q1_distance_list', 'qx_distance_list', 'DSR_gram_mat'] 
#Gram matrix of a vector V composed of all d and q in QSR and DSR will give the dot products between all pairs of original vectors...
#Gram matrix of a vector v composed of all d in DSR... --? for volume...
DSR_Component_Data = {} #dict keys are qids (str), dict values are dicts containing the DSR components for the indexed qid, by component name.
#---------------- values are either a single scalar, vector, etc for the DSR, or a list of scalar, vector, etc, one for each document linked to a QSR entry in the DSR.
#---------------- dict Keys are the component names...
# QA_QSR_DSR_RDMeD -- Relevant document mean deviation...
# QA_QSR_DSR_RDMaD -- Relevant document max mean deviation... ratio of this query qrel doc deviation to max deviation in DSR? The range of this measure would be 1 for ambiguous, ~0 for some other doc in DSR is way out there... Say 3 docs hae a deviation of 1 from centroid. 1/1 = 1...? Says ambiguous but they are all the same...
#-------------------------------------------------------- I think instead of max deviation orr ratio of this docs over max would be some idea of the distrubition / variance of the deviations. See the 3 docs each 1 unit deviation above.
# QA_QSR_DSR_RDVMD  -- Relevant document variance in mean deviation...
# QA_QSR_DSR_RDVo -- Relevant document volume...




def calculate_DSR_Components(query_embedding_service):
    DSR_Component_Data = {}
    for index, entry in enumerate(query_embedding_service.entries):
        DSR_Component_Data_Element = {}
        #TODO: remove this early break after code is complete to allow for a full data run.
        if index > 2:
            break
        querypool_nearest_k_results = query_embedding_service.entries[index].metadata["querypool_nearest_k_results"]

        DSR_Data = {
            'DSR_unique_doc_texts': [],# DSR_unique_doc_texts,
            'DSR_unique_doc_embeddings': [],# DSR_unique_doc_embeddings,
            'DSR_unique_doc_ids': [],# DSR_unique_doc_ids,
            'DSR_doc_ids': [],# DSR_doc_ids,
            'DSR_doc_ids_by_QSR_qid': [],# DSR_doc_ids_by_QSR_qid,
        }
        
        DSR_Data = query_embedding_service.entries[index].metadata["DSR_Data"]

        #What is the type of querypool_nearest_k_results? If it is a dict, then we can add a new key/keys to hold the qrel docids, and correspinding doc embedding vectors we are getting to estimate the DSR.
        DSR_unique_doc_texts = {} # Key will be the document id, value will be the document text
        DSR_unique_doc_embeddings = {} # Key will be the document id, value will be the document embedding vector
        DSR_doc_ids_by_QSR_qid = {} #Key will be the qid, value will be the doc id; this is the QSR neaearest k qid, not the entry qid...

        DSR_unique_doc_texts = DSR_Data.get('DSR_unique_doc_texts', {})
        DSR_unique_doc_embeddings = DSR_Data.get('DSR_unique_doc_embeddings', {})
        DSR_unique_doc_ids = DSR_Data.get('DSR_unique_doc_ids', {})
        DSR_doc_ids = DSR_Data.get('DSR_doc_ids', {})
        DSR_doc_ids_by_QSR_qid = DSR_Data.get('DSR_doc_ids_by_QSR_qid', {})

        # Calculate deviations of each document vector from the document region centroid
        deviations_list = [] #[np.linalg.norm(doc_embedding - doc_region_centroid) for doc_embedding in relevant_doc_embeddings]
        mean_deviation_sc = float(0)
        centroid_vec = None
        q1_distance_list = [] # The distance between the DSR indexed qrel document and the query specified in the current entry that the DSR describes.
        qx_distance_list = [] # The distance between the DSR indexed qrel document and the QSR query linking it to the DSR.
        DSR_gram_mat = None


        # Assuming DSR_unique_doc_embeddings.values() is a list of arrays
        values_list = [doc_embedding_entry[0] for doc_embedding_entry in list(DSR_unique_doc_embeddings.values())]
        centroid_vec = compute_centroid(values_list)
        doc_region_centroid = centroid_vec

        # Convert the list of arrays into a NumPy ndarray
        ndarray = np.array(values_list)
        DSR_gram_mat = compute_gram_matrix(ndarray)

        std_data_display_obj = {'ids':[[]],'documents':[[]],'distances':[[]],'embeddings':[[]]}
        for index, qid in enumerate(querypool_nearest_k_results["ids"][0]):
            qrel_doc_ids = DSR_doc_ids_by_QSR_qid[qid]
            for qrel_doc_id in qrel_doc_ids:
                std_data_display_obj['ids'][0].append(qrel_doc_id)

                qrel_doc_text = DSR_unique_doc_texts[qrel_doc_id]
                std_data_display_obj['documents'][0].append(qrel_doc_text)

                tqid_db_data = query_embedding_service.collection.get(ids=[entry.id],include=['embeddings'])
                tqid_vector = tqid_db_data['embeddings'][0]
                qrel_doc_embedding = DSR_unique_doc_embeddings[qrel_doc_id]

                tqid_vector_DSR_doc_distance = chroma_cosine_distance(qrel_doc_embedding, tqid_vector)
                q1_distance_list.append(float(tqid_vector_DSR_doc_distance))
                std_data_display_obj['distances'][0].append(float(tqid_vector_DSR_doc_distance))
                
                QSRqid_vector = querypool_nearest_k_results["embeddings"][0][index]
                QSRqid_vector_DSR_doc_distance = chroma_cosine_distance(qrel_doc_embedding, QSRqid_vector)
                qx_distance_list.append(QSRqid_vector_DSR_doc_distance)

                doc_embedding = DSR_unique_doc_embeddings[qrel_doc_id]
                std_data_display_obj['embeddings'][0].append(doc_embedding[0])
                deviation = np.linalg.norm(doc_embedding - doc_region_centroid)
                deviations_list.append(float(deviation))
        
        # Calculate the mean deviation
        mean_deviation_sc = np.mean(deviations_list)
        DSR_Component_Data_Element['mean_deviation_sc'] = mean_deviation_sc
        DSR_Component_Data_Element['deviations_list'] = deviations_list
        DSR_Component_Data_Element['centroid_vec'] = centroid_vec
        DSR_Component_Data_Element['q1_distance_list'] = q1_distance_list
        DSR_Component_Data_Element['qx_distance_list'] = qx_distance_list
        DSR_Component_Data_Element['DSR_gram_mat'] = DSR_gram_mat 

        DSR_Component_Data[entry.id] = DSR_Component_Data_Element
        print("Here are the DSR Documents for the given query:")
        detailed_display_query_results(entry,std_data_display_obj,query_embedding_service)
    return DSR_Component_Data

def calculate_QA_QSR_DSR_RDMeD(query_embedding_service, DSR_Component_Data):
    QA_QSR_DSR_RDMeD_Data = {}
    for index, entry in enumerate(query_embedding_service.entries):
        #TODO: remove this early break after code is complete to allow for a full data run.
        if index > 2:
            break
        #querypool_nearest_k_results = query_embedding_service.entries[index].metadata["querypool_nearest_k_results"]
        #for index, qid in enumerate(querypool_nearest_k_results["ids"][0]):
        #   pass
        QA_QSR_DSR_RDMeD = DSR_Component_Data[entry.id]['deviations_list'][0]
        QA_QSR_DSR_RDMeD_Data_Element = QA_QSR_DSR_RDMeD
        QA_QSR_DSR_RDMeD_Data[entry.id] = QA_QSR_DSR_RDMeD_Data_Element
    return QA_QSR_DSR_RDMeD_Data

def calculate_QA_QSR_DSR_RDVo(query_embedding_service, DSR_Component_Data):
    QA_QSR_DSR_RDVo_Data = {}
    for index, entry in enumerate(query_embedding_service.entries):
        #TODO: remove this early break after code is complete to allow for a full data run.
        if index > 2:
            break
        #querypool_nearest_k_results = query_embedding_service.entries[index].metadata["querypool_nearest_k_results"]
        #for index, qid in enumerate(querypool_nearest_k_results["ids"][0]):
        #   pass       
        volume = k_dim_parallelepiped_volume(DSR_Component_Data[entry.id]['DSR_gram_mat'])
        
        QA_QSR_DSR_RDVo_Data_Element = volume
        QA_QSR_DSR_RDVo_Data[entry.id] = QA_QSR_DSR_RDVo_Data_Element
    return QA_QSR_DSR_RDVo_Data

# QA_QSR_DSR_RDVMD  -- Relevant document variance in mean deviation...
def calculate_QA_QSR_DSR_RDVMD(query_embedding_service, DSR_Component_Data):
    QA_QSR_DSR_RDVMD_Data = {}
    for index, entry in enumerate(query_embedding_service.entries):
        #TODO: remove this early break after code is complete to allow for a full data run.
        if index > 2:
            break
        #querypool_nearest_k_results = query_embedding_service.entries[index].metadata["querypool_nearest_k_results"]
        #for index, qid in enumerate(querypool_nearest_k_results["ids"][0]):
        #   pass       
        sample = DSR_Component_Data[entry.id]['deviations_list']
        variance_in_mean_deviation = statistics.variance(sample)
        
        QA_QSR_DSR_RDVMD_Data_Element = variance_in_mean_deviation
        QA_QSR_DSR_RDVMD_Data[entry.id] = QA_QSR_DSR_RDVMD_Data_Element
    return QA_QSR_DSR_RDVMD_Data

def calculate_ambiguity_measures(query_embedding_service, Ambiguity_Measures_Data):
    if len(query_embedding_service.entries) > 0:
        if "DSR_Data" in query_embedding_service.entries[0].metadata:
            if "DSR_Components" not in Ambiguity_Measures_Data:
                #Try to load a pickeled object for data, recompute if not available
                pickle_file_path = calc_persist_path + "DSR_Components"
                result = load_or_create_pickle(pickle_file_path, calculate_DSR_Components, query_embedding_service)
                Ambiguity_Measures_Data["DSR_Components"] = result

            for tag in Ambiguity_Measures:
                if (tag not in Ambiguity_Measures_Data):
                    pickle_file_path = calc_persist_path + tag
                    if tag == 'QA_QSR_DSR_RDMeD':
                        result = load_or_create_pickle(pickle_file_path, calculate_QA_QSR_DSR_RDMeD, query_embedding_service, Ambiguity_Measures_Data["DSR_Components"])
                        pass
                    if tag == 'QA_QSR_DSR_RDVo':
                        result = load_or_create_pickle(pickle_file_path, calculate_QA_QSR_DSR_RDVo, query_embedding_service, Ambiguity_Measures_Data["DSR_Components"])
                        pass
                    if tag == 'QA_QSR_DSR_RDVMD':
                        result = load_or_create_pickle(pickle_file_path, calculate_QA_QSR_DSR_RDVMD, query_embedding_service, Ambiguity_Measures_Data["DSR_Components"])
                        pass

                    Ambiguity_Measures_Data[tag] = result
        else:
            assert(0==1)# Something is wrong. We need DSR_Data...
            pass
    else:
        assert(0==1)# Something is wrong. There should be entries already...
        pass
    return Ambiguity_Measures_Data

Ambiguity_Measures_Data = calculate_ambiguity_measures(query_embedding_service, Ambiguity_Measures_Data)
Ambiguity_Measures_Data

# %%

# Calculate query embeddings using spaCy Word2Vec model
#queries['embedding'] = queries['query'].apply(lambda x: get_spacy_embedding(x, nlp))


