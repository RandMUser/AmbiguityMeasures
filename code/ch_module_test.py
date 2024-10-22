# ch_module.py
# ch_module_test.py
import ch_module
GLOBAL_TEST_EARLY_STOP_DOCUMENTS = 5
GLOBAL_TEST_EARLY_STOP_FILES = 3


def test_get_filenames_from_directory():
    # Test valid directory
    valid_path = "./data/msmarco/corpus/msmarco_v2_doc"
    filenames = ch_module.get_filenames_from_directory(valid_path)
    assert isinstance(filenames, list), "The result should be a list."
    print("test_get_filenames_from_directory: Passed")

def test_sort_filenames_by_bundlenum():
    filenames = ['msmarco_doc_08.gz', 'msmarco_doc_01.gz', 'msmarco_doc_03.gz']
    sorted_filenames = ch_module.sort_filenames_by_bundlenum(filenames)
    assert sorted_filenames == ['msmarco_doc_01.gz', 'msmarco_doc_03.gz', 'msmarco_doc_08.gz'], "Sorting did not work as expected."
    print("test_sort_filenames_by_bundlenum: Passed")

def test_get_next_sorted_filename():
    path = "./data/msmarco/corpus/msmarco_v2_doc"
    filename_generator = ch_module.get_next_sorted_filename(path)
    filenames = list(filename_generator)
    assert len(filenames) > 0, "The generator did not yield any filenames."
    print("test_get_next_sorted_filename: Passed")

def test_parse_documents_from_corpus():
    saved_EARLY_DOC_STOP = ch_module.EARLY_DOC_STOP
    ch_module.EARLY_DOC_STOP = GLOBAL_TEST_EARLY_STOP_DOCUMENTS
    corpus_file = "./data/msmarco/corpus/msmarco_v2_doc/msmarco_doc_00.gz"
    documents = ch_module.parse_documents_from_corpus(corpus_file)
    assert isinstance(documents, list), "The result should be a list of documents."
    assert len(documents) > 0, "No documents were parsed."
    print("test_parse_documents_from_corpus: Passed")
    ch_module.EARLY_DOC_STOP = saved_EARLY_DOC_STOP

def test_get_next_document_fm_corpusfile():
    saved_EARLY_DOC_STOP = ch_module.EARLY_DOC_STOP
    print(ch_module.get_early_stopping_values())
    ch_module.EARLY_DOC_STOP = GLOBAL_TEST_EARLY_STOP_DOCUMENTS
    print(ch_module.get_early_stopping_values())
    corpus_file = "./data/msmarco/corpus/msmarco_v2_doc/msmarco_doc_00.gz"
    document_generator = ch_module.get_next_document_fm_corpusfile(corpus_file)
    documents = list(document_generator)
    assert len(documents) > 0, "The generator did not yield any documents."
    print("test_get_next_document_fm_corpusfile: Passed")
    ch_module.EARLY_DOC_STOP = saved_EARLY_DOC_STOP

def test_continuous_doc_data_fm_corpus():
    ch_module.EARLY_FILE_STOP = 1  # Test early stopping after 1 file
    ch_module.EARLY_DOC_STOP = 3   # Test early stopping after 3 documents per file
    corpus_directory = "./data/msmarco/corpus/msmarco_v2_doc"
    try:
        ch_module.continuous_doc_data_fm_corpus(corpus_directory)
        print("test_continuous_doc_data_fm_corpus: Passed")
    except Exception as e:
        assert False, f"An error occurred: {e}"

if __name__ == "__main__":
    test_get_filenames_from_directory()
    test_sort_filenames_by_bundlenum()
    test_get_next_sorted_filename()
    test_parse_documents_from_corpus()
    test_get_next_document_fm_corpusfile()
    test_continuous_doc_data_fm_corpus()


    

