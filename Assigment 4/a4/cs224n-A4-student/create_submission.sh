#!/bin/bash

# CS224n Assignment 4 - Student Submission Packager
# This script packages your implementation for Gradescope submission

echo "Creating student submission zip..."

# Required files for submission
FILES_TO_INCLUDE=(
    "gsm8k.py"
    "llm_judge.py"
    "redteam.py"
)

# Check if required files exist
missing_files=()
for file in "${FILES_TO_INCLUDE[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "ERROR: Missing required files:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    echo "Please ensure all required files are present before creating submission."
    exit 1
fi

# Create submission zip
zip_name="submission.zip"
rm -f "$zip_name"

# Add required files only
for file in "${FILES_TO_INCLUDE[@]}"; do
    if [ -f "$file" ]; then
        echo "Adding: $file"
        zip -q "$zip_name" "$file"
    fi
done

echo ""
echo "✓ Submission created: $zip_name"
echo "✓ Ready to upload to Gradescope!"
echo ""
echo "Files included:"
unzip -l "$zip_name"