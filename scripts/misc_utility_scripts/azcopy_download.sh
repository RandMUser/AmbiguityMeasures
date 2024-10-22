#!/bin/bash
#./azcopy_download.sh -s "https://example.blob.core.windows.net/container/file" -d "/local/path/file" --dry-run --verify-hash
#./azcopy_download.sh -s "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2_passage.tar" -d "./az_msmarco_v2_passage.tar" --dry-run --verify-hash
#download_if_not_exists "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2_passage.tar" "$DATA_DIR/msmarco_v2_passage.tar"

#azcopy copy "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2_passage.tar" "./az_msmarco_v2_passage.tar" --recursive --dry-run --check-md5 FailIfDifferentOrMissing --from-to BlobLocal

#azcopy copy "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2_passage.tar" "./az_msmarco_v2_passage.tar" --recursive --dry-run --check-md5 FailIfDifferentOrMissing

#!/bin/bash
#!/bin/bash

# Usage function to display help
usage() {
    echo "Usage: $0 -s <source-url> -d <destination-path> [-src-sas <source-sas-token>] [-dest-sas <destination-sas-token>] [--dry-run] [--verify-hash]"
    exit 1
}

# Variables
DRY_RUN=false
VERIFY_HASH=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--source-url)
            SOURCE_URL="$2"
            shift 2
            ;;
        -d|--destination-path)
            DESTINATION_PATH="$2"
            shift 2
            ;;
        -src-sas|--source-sas-token)
            SOURCE_SAS="$2"
            shift 2
            ;;
        -dest-sas|--destination-sas-token)
            DEST_SAS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift 1
            ;;
        --verify-hash)
            VERIFY_HASH=true
            shift 1
            ;;
        *)
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$SOURCE_URL" || -z "$DESTINATION_PATH" ]]; then
    echo "Error: Source URL and destination path are required."
    usage
fi

# Function to list metadata in dry-run mode
list_metadata() {
    echo "Listing metadata for: $SOURCE_URL"
    azcopy list "$SOURCE_URL" --properties "ContentMD5;LastModifiedTime;BlobType;ContentType;VersionId" --output-type json --location Blob
    if [ $? -eq 0 ]; then
        echo "Metadata listed successfully."
    else
        echo "Failed to list metadata. Check the logs for more details."
        exit 1
    fi
}

# Function to resume interrupted jobs
resume_jobs() {
    echo "Checking for interrupted jobs..."
    JOB_IDS=$(azcopy jobs list --with-status=Failed | grep 'JobID:' | awk '{print $2}')

    if [ -z "$JOB_IDS" ]; then
        echo "No interrupted jobs found to resume."
        return 1  # No jobs resumed
    else
        echo "Found interrupted job(s). Attempting to resume..."
        for JOB_ID in $JOB_IDS; do
            echo "Resuming job: $JOB_ID"
            azcopy jobs resume "$JOB_ID" --source-sas="$SOURCE_SAS" --destination-sas="$DEST_SAS"
            if [ $? -eq 0 ]; then
                echo "Job $JOB_ID resumed and completed successfully."
                return 0  # Job resumed successfully
            else
                echo "Failed to resume job $JOB_ID. Check the logs for details."
            fi
        done
        return 1  # If all resume attempts failed
    fi
}

# Function to perform a dry run
dry_run() {
    echo "Performing a dry run..."
    list_metadata
    azcopy copy "$SOURCE_URL" "$DESTINATION_PATH" --recursive --from-to BlobLocal  --dry-run
    if [ $? -eq 0 ]; then
        echo "Dry run completed successfully."
    else
        echo "Dry run failed. Check the logs for more details."
        exit 1
    fi
    exit 0  # Exit after a dry run
}

# Function to calculate MD5 hash of the downloaded file
verify_hash() {
    if command -v md5sum &> /dev/null; then
        HASH=$(md5sum "$DESTINATION_PATH" | awk '{print $1}')
        echo "MD5 hash of the downloaded file: $HASH"
    else
        echo "md5sum command not found. Skipping hash verification."
    fi
}

# Check if dry run is requested
if [ "$DRY_RUN" = true ]; then
    dry_run
fi

# Start download or resume previous one if available
echo "Initiating download from $SOURCE_URL to $DESTINATION_PATH..."

# Attempt to resume jobs
resume_jobs
RESUME_RESULT=$?

if [ $RESUME_RESULT -ne 0 ]; then
    # Start a new download only if no jobs were resumed successfully
    echo "Starting new download..."
    azcopy copy "$SOURCE_URL" "$DESTINATION_PATH" --recursive --from-to BlobLocal 

    # Verify if the download completed successfully
    if [ $? -eq 0 ]; then
        echo "Download completed successfully."
    else
        echo "Download failed. Check the logs for more details."
        exit 1
    fi
else
    echo "Download was completed with resumed job(s). No new download needed."
fi

# Verify hash if requested
if [ "$VERIFY_HASH" = true ]; then
    verify_hash
fi
