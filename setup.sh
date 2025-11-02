#!/bin/bash

# Med-HALT RAG PoC Setup Script

set -e

echo "=========================================="
echo "Med-HALT RAG PoC Setup"
echo "=========================================="
echo ""

# Target Python version
PYTHON_VERSION="3.10.14"

# Check if pyenv is installed
echo "Checking for pyenv..."
if ! command -v pyenv &> /dev/null; then
    echo "❌ pyenv not found!"
    echo "Please install pyenv first:"
    echo "  brew install pyenv  # on macOS"
    echo "  or visit: https://github.com/pyenv/pyenv#installation"
    exit 1
fi

echo "✓ pyenv found: $(pyenv --version)"

# Check if Python 3.10.14 is installed
echo ""
echo "Checking for Python ${PYTHON_VERSION}..."
if ! pyenv versions | grep -q "${PYTHON_VERSION}"; then
    echo "Python ${PYTHON_VERSION} not found. Installing..."
    pyenv install ${PYTHON_VERSION}
else
    echo "✓ Python ${PYTHON_VERSION} already installed"
fi

# Set local Python version for this project
echo ""
echo "Setting local Python version to ${PYTHON_VERSION}..."
pyenv local ${PYTHON_VERSION}

# Verify Python version
echo "Verifying Python version..."
python --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt

# Download spaCy model
echo ""
echo "Downloading spaCy biomedical model..."
python -m spacy download en_core_sci_sm || echo "Note: Install scispacy models manually if this fails"

# Create .env from template
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  IMPORTANT: Edit .env and add your API keys!"
fi

# Create placeholder directories
echo ""
echo "Creating data directories..."
mkdir -p data/raw/medhalt
mkdir -p data/raw/pubmed_baseline
mkdir -p data/raw/medlineplus
mkdir -p data/raw/pubtator
mkdir -p data/chunks
mkdir -p index
mkdir -p graph
mkdir -p results
mkdir -p reports

# Create .gitkeep files
touch data/.gitkeep
touch results/.gitkeep
touch reports/.gitkeep

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env and add your OpenAI API key"
echo "2. (Optional) Start Neo4j: docker-compose up -d"
echo "3. Download datasets: python ingest/download_datasets.py --dest data/raw"
echo "4. See README.md for full pipeline instructions"
echo ""
echo "Quick test:"
echo "  source venv/bin/activate"
echo "  python baseline/baseline_run.py --help"
echo ""
