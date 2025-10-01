@echo off
REM Jekyll Local Development Server Script for Windows
REM Run this script from the website/ directory

echo [INFO] Starting Jekyll local development server...

REM Check if we're in the right directory
if not exist "_config.yml" (
    echo [ERROR] Not in Jekyll website directory. Please run this script from the website/ folder.
    pause
    exit /b 1
)

REM Kill any existing Jekyll processes
echo [INFO] Checking for existing Jekyll processes...
taskkill /f /im ruby.exe 2>nul || echo No existing Jekyll processes found

REM Install dependencies if needed
if not exist "vendor\bundle" (
    echo [INFO] Installing Jekyll dependencies...
    bundle config set --local path 'vendor/bundle'
    bundle install
    echo [SUCCESS] Dependencies installed successfully
) else (
    echo [INFO] Dependencies already installed
)

REM Sync materials before starting server
echo [INFO] Syncing lecture materials...
cd ..
if exist "sync_materials.py" (
    python sync_materials.py --verbose
    echo [SUCCESS] Materials synced successfully
) else (
    echo [WARNING] sync_materials.py not found. Skipping material sync.
)
cd website

REM Start Jekyll server
echo [INFO] Starting Jekyll server with live reload...
echo [INFO] Server will be available at: http://127.0.0.1:4000/
echo [INFO] Press Ctrl+C to stop the server
echo.

bundle exec jekyll serve --livereload --incremental --trace

echo [SUCCESS] Jekyll server stopped
pause

