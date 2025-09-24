#!/bin/bash

# Jekyll Local Development Server Script
# This script sets up and runs Jekyll locally for the Intro to ML course website

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "_config.yml" ]; then
    print_error "Not in Jekyll website directory. Please run this script from the website/ folder."
    exit 1
fi

print_status "Starting Jekyll local development server..."

# Check if Ruby and Bundler are installed
if ! command -v ruby &> /dev/null; then
    print_error "Ruby is not installed. Please install Ruby first."
    exit 1
fi

if ! command -v bundle &> /dev/null; then
    print_error "Bundler is not installed. Please install Bundler first."
    exit 1
fi

# Install dependencies if needed
if [ ! -d "vendor/bundle" ]; then
    print_status "Installing Jekyll dependencies..."
    bundle config set --local path 'vendor/bundle'
    bundle install
    print_success "Dependencies installed successfully"
else
    print_status "Dependencies already installed"
fi

# Kill any existing Jekyll processes
print_status "Checking for existing Jekyll processes..."
if pgrep -f "jekyll serve" > /dev/null; then
    print_warning "Found existing Jekyll server. Stopping it..."
    pkill -f "jekyll serve"
    sleep 2
fi

# Sync materials before starting server
print_status "Syncing lecture materials..."
cd ..
if [ -f "sync_materials.py" ]; then
    python3 sync_materials.py --verbose
    print_success "Materials synced successfully"
else
    print_warning "sync_materials.py not found. Skipping material sync."
fi
cd website

# Start Jekyll server
print_status "Starting Jekyll server with live reload..."
print_status "Server will be available at: http://127.0.0.1:4000/"
print_status "Press Ctrl+C to stop the server"
echo ""

# Start the server with live reload and incremental builds
bundle exec jekyll serve --livereload --incremental --trace

print_success "Jekyll server stopped"
