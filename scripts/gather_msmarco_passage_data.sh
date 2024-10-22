#!/bin/bash

DATA_DIR="./data/msmarco"
CORPUS_DIR="$DATA_DIR/corpus"
# Create directories if they do not exist
mkdir -p "$DATA_DIR"
mkdir -p "$CORPUS_DIR"

# Function to download a file if it does not exist or resume if interrupted
download_if_not_exists() {
  local url=$1
  local output_path=$2
  if [ -f "$output_path" ]; then
    # Check if the file is incomplete by attempting to resume the download
    echo "File $output_path already exists. Checking if download is complete..."
    if ! curl -L -C - -o "$output_path" "$url"; then
      echo "Error: Failed to resume download for $url"
      exit 1
    else
      echo "File $output_path download resumed and completed successfully."
    fi
  else
    echo "Downloading $output_path..."
    if ! curl -L -C - -o "$output_path" "$url"; then
      echo "Error: Failed to download $url"
      exit 1
    fi
  fi
}

# Download MSMARCO Corpus
echo "Downloading MSMARCO Corpus..."
download_if_not_exists "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2_passage.tar" "$DATA_DIR/msmarco_v2_passage.tar"

# Extract MSMARCO Corpus if not already extracted
if [ -d "$CORPUS_DIR/msmarco_passage_00.jsonl.gz" ]; then
  echo "Corpus already extracted. Skipping extraction."
else
  echo "Extracting MSMARCO Corpus..."
  if ! tar -xvf "$DATA_DIR/msmarco_v2_passage.tar" -C "$CORPUS_DIR"; then
    echo "Error: Failed to extract MSMARCO Corpus."
    exit 1
  fi
fi

# Download Queries and Qrels
echo "Downloading Train Queries..."
download_if_not_exists "https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_train_queries.tsv" "$DATA_DIR/passv2_train_queries.tsv"

echo "Downloading Train Top 100..."
download_if_not_exists "https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_train_top100.txt.gz" "$DATA_DIR/passv2_train_top100.txt.gz"

echo "Downloading Train Qrels..."
download_if_not_exists "https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_train_qrels.tsv" "$DATA_DIR/passv2_train_qrels.tsv"

# Download Development Set 1
echo "Downloading Dev 1 Queries..."
download_if_not_exists "https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev_queries.tsv" "$DATA_DIR/passv2_dev_queries.tsv"

echo "Downloading Dev 1 Top 100..."
download_if_not_exists "https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev_top100.txt.gz" "$DATA_DIR/passv2_dev_top100.txt.gz"

echo "Downloading Dev 1 Qrels..."
download_if_not_exists "https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev_qrels.tsv" "$DATA_DIR/passv2_dev_qrels.tsv"

# Download Development Set 2
echo "Downloading Dev 2 Queries..."
download_if_not_exists "https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev2_queries.tsv" "$DATA_DIR/passv2_dev2_queries.tsv"

echo "Downloading Dev 2 Top 100..."
download_if_not_exists "https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev2_top100.txt.gz" "$DATA_DIR/passv2_dev2_top100.txt.gz"

echo "Downloading Dev 2 Qrels..."
download_if_not_exists "https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev2_qrels.tsv" "$DATA_DIR/passv2_dev2_qrels.tsv"

# Download Validation Sets
echo "Downloading Validation 1 (TREC 2021) Queries..."
download_if_not_exists "https://msmarco.z22.web.core.windows.net/msmarcoranking/2021_queries.tsv" "$DATA_DIR/2021_queries.tsv"

echo "Downloading Validation 1 Top 100..."
download_if_not_exists "https://msmarco.z22.web.core.windows.net/msmarcoranking/2021_passage_top100.txt.gz" "$DATA_DIR/2021_passage_top100.txt.gz"

echo "Downloading Validation 1 Qrels..."
download_if_not_exists "https://trec.nist.gov/data/deep/2021.qrels.pass.final.txt" "$DATA_DIR/2021.qrels.pass.final.txt"

echo "Downloading Validation 2 (TREC 2022) Queries..."
download_if_not_exists "https://msmarco.z22.web.core.windows.net/msmarcoranking/2022_queries.tsv" "$DATA_DIR/2022_queries.tsv"

echo "Downloading Validation 2 Top 100..."
download_if_not_exists "https://msmarco.z22.web.core.windows.net/msmarcoranking/2022_passage_top100.txt.gz" "$DATA_DIR/2022_passage_top100.txt.gz"

echo "Downloading Validation 2 Qrels..."
download_if_not_exists "https://trec.nist.gov/data/deep/2022.qrels.pass.withDupes.txt" "$DATA_DIR/2022.qrels.pass.withDupes.txt"

# Download Test Set (TREC 2023)
echo "Downloading Test Queries (TREC 2023)..."
download_if_not_exists "https://msmarco.z22.web.core.windows.net/msmarcoranking/2023_queries.tsv" "$DATA_DIR/2023_queries.tsv"

echo "Downloading Test Top 100..."
download_if_not_exists "https://msmarco.z22.web.core.windows.net/msmarcoranking/2023_passage_top100.txt.gz" "$DATA_DIR/2023_passage_top100.txt.gz"

echo "Downloading Test Qrels..."
download_if_not_exists "https://trec.nist.gov/data/deep/2023.qrels.pass.withDupes.txt" "$DATA_DIR/2023.qrels.pass.withDupes.txt"


echo "Downloading Completed."
