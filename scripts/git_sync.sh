#!/bin/bash

echo "🔄 Syncing with remote repository..."

# First, fetch the latest changes
git fetch origin

# Check if there are any conflicts
echo "📥 Pulling remote changes..."
git pull origin main

# If pull was successful, push your changes
if [ $? -eq 0 ]; then
    echo "📤 Pushing local changes..."
    git push origin main
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully synced with remote!"
    else
        echo "❌ Failed to push changes"
    fi
else
    echo "⚠️  Merge conflicts detected. Please resolve manually."
    echo "Run 'git status' to see conflicted files"
fi
