#!/bin/bash

# Deploy script for Image-to-Image Editor Application to Databricks Apps
# This script builds the application and deploys it as a Databricks App

set -e

echo "======================================"
echo "Databricks App Deployment Script"
echo "======================================"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to display error and exit
error_exit() {
    echo "ERROR: $1" >&2
    exit 1
}

# Check for required tools
echo ""
echo "Step 1: Checking dependencies..."

if ! command_exists databricks; then
    echo "❌ Databricks CLI not found!"
    echo "Install it with: pip install databricks-cli"
    echo "Or follow: https://docs.databricks.com/dev-tools/cli/index.html"
    error_exit "Databricks CLI is required"
fi
echo "✓ Databricks CLI found"

if ! command_exists node; then
    echo "❌ Node.js not found!"
    echo "Install it from: https://nodejs.org/"
    error_exit "Node.js is required for building the frontend"
fi
echo "✓ Node.js found"

if ! command_exists npm; then
    echo "❌ npm not found!"
    error_exit "npm is required for building the frontend"
fi
echo "✓ npm found"

if ! command_exists python3; then
    echo "❌ Python 3 not found!"
    error_exit "Python 3 is required"
fi
echo "✓ Python 3 found"

# Load environment variables
echo ""
echo "Step 2: Loading configuration..."

if [ -f ".env.local" ]; then
    echo "Loading environment from .env.local"
    set -a
    source .env.local
    set +a
elif [ -f ".env" ]; then
    echo "Loading environment from .env"
    set -a
    source .env
    set +a
else
    echo "⚠ No .env.local or .env file found. Using environment variables..."
fi

# Validate required environment variables
DATABRICKS_CONFIG_PROFILE=${DATABRICKS_CONFIG_PROFILE:-"DEFAULT"}
WORKSPACE_SOURCE_PATH=${WORKSPACE_SOURCE_PATH:-""}
DATABRICKS_APP_NAME=${DATABRICKS_APP_NAME:-""}
MODEL_ENDPOINT_URL=${MODEL_ENDPOINT_URL:-""}

if [ -z "$WORKSPACE_SOURCE_PATH" ]; then
    echo "⚠ WORKSPACE_SOURCE_PATH not set. Please enter the workspace path:"
    read -p "Workspace path (e.g., /Workspace/Users/user@example.com/imagen-ui-app): " WORKSPACE_SOURCE_PATH
    if [ -z "$WORKSPACE_SOURCE_PATH" ]; then
        error_exit "WORKSPACE_SOURCE_PATH is required"
    fi
fi

if [ -z "$DATABRICKS_APP_NAME" ]; then
    echo "⚠ DATABRICKS_APP_NAME not set. Please enter the app name:"
    read -p "App name (e.g., imagen-ui-app): " DATABRICKS_APP_NAME
    if [ -z "$DATABRICKS_APP_NAME" ]; then
        error_exit "DATABRICKS_APP_NAME is required"
    fi
fi

echo "Configuration:"
echo "  Profile: $DATABRICKS_CONFIG_PROFILE"
echo "  Workspace Path: $WORKSPACE_SOURCE_PATH"
echo "  App Name: $DATABRICKS_APP_NAME"
if [ -n "$MODEL_ENDPOINT_URL" ]; then
    echo "  Model Endpoint: $MODEL_ENDPOINT_URL"
else
    echo "  ⚠ Model Endpoint: Not configured (required at runtime)"
fi

# Authenticate with Databricks
echo ""
echo "Step 3: Authenticating with Databricks..."

if ! databricks auth describe --profile "$DATABRICKS_CONFIG_PROFILE" >/dev/null 2>&1; then
    echo "❌ Authentication failed with profile: $DATABRICKS_CONFIG_PROFILE"
    echo "Please configure authentication with: databricks configure --profile $DATABRICKS_CONFIG_PROFILE"
    error_exit "Databricks authentication required"
fi
echo "✓ Authenticated with Databricks"

# Build the application
echo ""
echo "======================================"
echo "Building Application"
echo "======================================"

# Run the build script
if [ ! -f "build.sh" ]; then
    error_exit "build.sh not found in current directory"
fi

echo "Running build script..."
bash build.sh

# Verify build output
if [ ! -d "static" ] || [ ! -f "static/index.html" ]; then
    error_exit "Build failed: static/ directory or index.html not found"
fi
echo "✓ Build completed successfully"

# Create .databricksignore if it doesn't exist
echo ""
echo "Step 4: Preparing deployment files..."

if [ ! -f ".databricksignore" ]; then
    echo "Creating .databricksignore..."
    cat > .databricksignore << 'EOF'
# Node modules and frontend source
frontend/node_modules/
frontend/src/
frontend/public/
frontend/.vite/
frontend/.eslintrc.cjs
frontend/vite.config.js
frontend/package.json
frontend/package-lock.json

# Python virtual environment
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg-info/
dist/
build/

# Development files
.git/
.gitignore
.env.local
.env
*.log
feedback_log.json

# IDE files
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Documentation (optional - remove if you want to include)
README.md
DEPLOYMENT.md

# Build scripts (optional - remove if you want to include)
build.sh
EOF
    echo "✓ Created .databricksignore"
else
    echo "✓ Using existing .databricksignore"
fi

# Create app.yaml for Databricks Apps
echo ""
echo "Step 5: Creating app configuration..."

cat > app.yaml << EOF
# Databricks App Configuration for Image-to-Image Editor
command:
  - "uvicorn"
  - "main:app"
  - "--host"
  - "0.0.0.0"
  - "--port"
  - "8000"

env:
  - name: ENV
    value: production
  - name: MODEL_ENDPOINT_URL
    value: ${MODEL_ENDPOINT_URL:-""}
  - name: MODEL_ENDPOINT_TOKEN
    secret: model-endpoint-token

EOF

echo "✓ Created app.yaml"

# Sync files to Databricks workspace
echo ""
echo "Step 6: Syncing files to Databricks workspace..."

echo "Syncing to: $WORKSPACE_SOURCE_PATH"

# Use databricks sync command
# This will sync the current directory to the workspace, respecting .databricksignore
# and explicitly including the built static files
databricks sync . "$WORKSPACE_SOURCE_PATH" \
    --exclude-from .databricksignore \
    --include "static/**" \
    --full \
    --profile "$DATABRICKS_CONFIG_PROFILE"

echo "✓ Files synced to workspace"

# Deploy the app
echo ""
echo "Step 7: Deploying Databricks App..."

echo "Deploying app: $DATABRICKS_APP_NAME"

# Deploy using Databricks Apps CLI
databricks apps deploy \
    "$DATABRICKS_APP_NAME" \
    --source-code-path "$WORKSPACE_SOURCE_PATH" \
    --profile "$DATABRICKS_CONFIG_PROFILE"

echo "✓ App deployment initiated"

# Wait a moment for deployment to process
echo ""
echo "Waiting for deployment to process..."
sleep 5

# Get app status
echo ""
echo "Step 8: Verifying deployment..."

databricks apps list --profile "$DATABRICKS_CONFIG_PROFILE" | grep "$DATABRICKS_APP_NAME" || true

echo ""
echo "======================================"
echo "Deployment Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Check app status: databricks apps get $DATABRICKS_APP_NAME --profile $DATABRICKS_CONFIG_PROFILE"
echo "  2. View logs: databricks apps logs $DATABRICKS_APP_NAME --profile $DATABRICKS_CONFIG_PROFILE"
echo "  3. Access your app in Databricks UI under 'Apps'"
echo "  4. Test the image-to-image editor functionality"
echo ""
echo "Important notes:"
echo "  - Ensure MODEL_ENDPOINT_URL is configured in app settings"
echo "  - Set up the 'model-endpoint-token' secret if required"
echo "  - Monitor initial startup logs for any issues"
echo ""
echo "To update the app, run this script again."
echo "======================================"
