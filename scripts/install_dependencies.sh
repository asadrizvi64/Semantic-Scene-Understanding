#!/bin/bash
# Script to install system dependencies for Narrative Scene Understanding

# Exit on error
set -e

# Print commands
set -x

# Function to print section headers
print_header() {
    echo "========================================"
    echo "  $1"
    echo "========================================"
}

# Check if script is run as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (use sudo)"
    exit 1
fi

# Update package lists
print_header "Updating package lists"
apt-get update

# Install basic dependencies
print_header "Installing basic dependencies"
apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-venv

# Install OpenCV dependencies
print_header "Installing OpenCV dependencies"
apt-get install -y \
    libopencv-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran

# Install FFmpeg
print_header "Installing FFmpeg"
apt-get install -y ffmpeg

# Install audio processing dependencies
print_header "Installing audio processing dependencies"
apt-get install -y \
    libsndfile1 \
    libasound-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    libjack-jackd2-dev

# Install Tesseract OCR
print_header "Installing Tesseract OCR"
apt-get install -y \
    tesseract-ocr \
    libtesseract-dev

# Install CUDA dependencies (if needed)
read -p "Do you want to install CUDA dependencies? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_header "Installing CUDA dependencies"
    # Add CUDA repository (adjust for your system)
    apt-get install -y software-properties-common
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    dpkg -i cuda-keyring_1.0-1_all.deb
    apt-get update
    
    # Install CUDA toolkit
    apt-get install -y cuda-toolkit-12-1
    
    # Set up CUDA environment variables
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /etc/profile.d/cuda.sh
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> /etc/profile.d/cuda.sh
    chmod +x /etc/profile.d/cuda.sh
    
    # Install cuDNN
    apt-get install -y libcudnn8
fi

# Install GraphViz for visualization
print_header "Installing GraphViz"
apt-get install -y graphviz libgraphviz-dev pkg-config

# Install Headless Chrome dependencies (for interactive visualizations)
print_header "Installing Headless Chrome dependencies"
apt-get install -y \
    gconf-service \
    libasound2 \
    libatk1.0-0 \
    libcairo2 \
    libcups2 \
    libfontconfig1 \
    libgdk-pixbuf2.0-0 \
    libgtk-3-0 \
    libnspr4 \
    libpango-1.0-0 \
    libxss1 \
    fonts-liberation \
    libnss3 \
    lsb-release \
    xdg-utils

# Create virtual environment
print_header "Setting up Python virtual environment"
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Install Python dependencies
print_header "Installing Python dependencies"
source venv/bin/activate
pip install --upgrade pip
pip install wheel setuptools

# Basic requirements
pip install -r requirements.txt

# Print success message
print_header "Installation complete!"
echo "System dependencies have been installed."
echo ""
echo "To complete the setup:"
echo "1. Activate the virtual environment:   source venv/bin/activate"
echo "2. Download the required models:       python scripts/download_models.py"
echo "3. Run the installation test:          python tests/test_installation.py"
echo ""
echo "If you installed CUDA, you may need to restart your system for all changes to take effect."