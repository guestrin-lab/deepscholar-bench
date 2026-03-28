#!/bin/bash

# Batch filter multiple output folders
# Usage: ./batch_filter_papers.sh [min_citations] [min_rw_length] [max_rw_length]

# Set default parameters
MIN_CITATIONS=${1:-5}
MIN_RW_LENGTH=${2:-200}
MAX_RW_LENGTH=${3:-10000}

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUTS_DIR="$SCRIPT_DIR/outputs"

echo "========================================"
echo "Batch Paper Quality Filter"
echo "========================================"
echo "Minimum citations: $MIN_CITATIONS"
echo "Min related works length: $MIN_RW_LENGTH"
echo "Max related works length: $MAX_RW_LENGTH"
echo ""

# Counter for tracking
TOTAL=0
SUCCESS=0
FAILED=0

# Find all directories in outputs/ that contain paper_content.csv
for folder in "$OUTPUTS_DIR"/*/; do
    if [ -f "$folder/paper_content.csv" ]; then
        TOTAL=$((TOTAL + 1))
        folder_name=$(basename "$folder")
        
        echo "----------------------------------------"
        echo "Processing: $folder_name"
        echo "----------------------------------------"
        
        # Run the filter script
        if python "$SCRIPT_DIR/filter_quality_papers.py" \
            --input-folder "$folder" \
            --min-citations "$MIN_CITATIONS" \
            --min-rw-length "$MIN_RW_LENGTH" \
            --max-rw-length "$MAX_RW_LENGTH"; then
            SUCCESS=$((SUCCESS + 1))
            echo "✅ Successfully filtered $folder_name"
        else
            FAILED=$((FAILED + 1))
            echo "❌ Failed to filter $folder_name"
        fi
        echo ""
    fi
done

# Print summary
echo "========================================"
echo "BATCH PROCESSING SUMMARY"
echo "========================================"
echo "Total folders processed: $TOTAL"
echo "Successfully filtered: $SUCCESS"
echo "Failed: $FAILED"
echo "========================================"

if [ $TOTAL -eq 0 ]; then
    echo "⚠️  No folders with paper_content.csv found in $OUTPUTS_DIR"
    exit 1
fi

if [ $FAILED -gt 0 ]; then
    exit 1
fi

echo "✅ All folders processed successfully!"
exit 0



