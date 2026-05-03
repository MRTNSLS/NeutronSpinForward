#!/bin/bash

# --- Configuration ---
# You can change the destination path here
DEST_BASE="0:/home/user"
DEST_DIR="$DEST_BASE"

echo "🚀 Syncing GPU-optimized pipeline to $DEST_DIR..."

# Ensure we are in the correct directory
cd "$(dirname "$0")"

# 1. Sync the core package (reproduce_neutron)
# We assume tnr scp -r works for recursive directory upload
echo "📦 Uploading library: reproduce_neutron/ ..."
tnr scp reproduce_neutron "$DEST_DIR/"

# 2. Sync root-level python scripts and config
echo "📄 Uploading scripts, configuration, and requirements..."
if [ -f "../requirements.txt" ]; then
    echo "   -> requirements.txt (from root)"
    tnr scp "../requirements.txt" "$DEST_DIR/"
fi

for file in *.py config*.json README.md; do
    if [ -f "$file" ]; then
        echo "   -> $file"
        tnr scp "$file" "$DEST_DIR/"
    fi
done

echo "----------------------------------------------------"
echo "✅ Sync complete!"
echo "Skipped: data/, models/, results/, and __pycache__"
echo "----------------------------------------------------"
