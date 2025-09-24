#!/bin/bash

# Quick Jekyll Server - Minimal setup
# Run this for quick development without full setup

set -e

echo "🚀 Starting Jekyll server..."
echo "📍 Server: http://127.0.0.1:4000/"
echo "⏹️  Press Ctrl+C to stop"
echo ""

# Kill existing Jekyll processes
pkill -f "jekyll serve" 2>/dev/null || true

# Start server
bundle exec jekyll serve --livereload --incremental
