"""
Setup script for Universal RAG Application
This script helps set up the environment and dependencies.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("Please use Python 3.8 or higher")
        return False

def main():
    """Main setup function."""
    print("🚀 Universal RAG Application Setup")
    print("==================================")
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install Python dependencies
    print("\n📦 Installing Python dependencies...")
    if not run_command("pip install -e .", "Installing project dependencies"):
        print("Alternatively, try: pip install -r requirements.txt")
        if not run_command("pip install -r requirements.txt", "Installing from requirements.txt"):
            return
    
    # Check if Ollama is installed
    print("\n🤖 Checking Ollama installation...")
    if run_command("ollama --version", "Checking Ollama version"):
        print("✅ Ollama is installed")
        
        # Pull required models
        print("\n📥 Pulling required Ollama models...")
        run_command("ollama pull mxbai-embed-large", "Pulling embedding model (mxbai-embed-large)")
        run_command("ollama pull llama3.2:latest", "Pulling language model (llama3.2:latest)")
    else:
        print("❌ Ollama is not installed")
        print("\nPlease install Ollama from: https://ollama.ai/")
        print("Then run the following commands:")
        print("  ollama pull mxbai-embed-large")
        print("  ollama pull llama3.2:latest")
    
    # Check for Tesseract (for OCR)
    print("\n👁️ Checking Tesseract OCR...")
    if not run_command("tesseract --version", "Checking Tesseract installation"):
        print("❌ Tesseract OCR is not installed")
        print("Please install Tesseract for image text extraction:")
        print("  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        print("  macOS: brew install tesseract")
        print("  Linux: sudo apt-get install tesseract-ocr")
    
    # Create necessary directories
    print("\n📁 Creating necessary directories...")
    os.makedirs("data", exist_ok=True)
    os.makedirs("chroma_db", exist_ok=True)
    print("✅ Directories created")
    
    # Final instructions
    print("\n🎉 Setup Complete!")
    print("==================")
    print("\nTo run the application:")
    print("  1. Make sure Ollama is running: ollama serve")
    print("  2. Place your documents in the 'data' folder")
    print("  3. Run: streamlit run main.py")
    print("\nSupported file types: PDF, DOCX, XLSX, JPG, PNG")

if __name__ == "__main__":
    main()
