#!/usr/bin/env python3
"""
Script to convert models to different formats for Narrative Scene Understanding.
Currently supports:
- Converting Llama models from Meta's format to GGUF format for llama.cpp
- Converting PyTorch models to ONNX format for faster inference
"""

import os
import sys
import argparse
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("model-converter")

def convert_llama_to_gguf(input_path, output_path, model_size="7b", quantize="q4_k_m"):
    """
    Convert Llama model from Meta's format to GGUF format for llama.cpp.
    
    Args:
        input_path: Path to the input model directory
        output_path: Path to save the converted model
        model_size: Size of the model (e.g., "7b", "13b", "70b")
        quantize: Quantization method (e.g., "q4_k_m", "q5_k_m", "q8_0")
    
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Converting Llama {model_size} model to GGUF format with {quantize} quantization")
    
    # Validate input path
    input_path = os.path.abspath(os.path.expanduser(input_path))
    if not os.path.isdir(input_path):
        logger.error(f"Input path is not a directory: {input_path}")
        return False
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(os.path.expanduser(output_path)))
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Check if llama.cpp is available in the current path or as a Git submodule
        llama_cpp_dir = None
        
        # Check common locations
        potential_paths = [
            "./llama.cpp",
            "../llama.cpp",
            "./third_party/llama.cpp",
            os.path.join(os.path.dirname(__file__), "../third_party/llama.cpp")
        ]
        
        for path in potential_paths:
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "convert.py")):
                llama_cpp_dir = os.path.abspath(path)
                logger.info(f"Found llama.cpp at {llama_cpp_dir}")
                break
        
        # If not found, clone the repository to a temporary directory
        if llama_cpp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="llama_cpp_")
            logger.info(f"Cloning llama.cpp to {temp_dir}")
            
            subprocess.run(
                ["git", "clone", "https://github.com/ggerganov/llama.cpp.git", temp_dir],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            llama_cpp_dir = temp_dir
        
        # Build llama.cpp if needed
        if not os.path.exists(os.path.join(llama_cpp_dir, "quantize")):
            logger.info("Building llama.cpp...")
            subprocess.run(
                ["make"],
                cwd=llama_cpp_dir,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        
        # First, convert from Meta format to legacy format (if needed)
        # Check if input path has .bin files which indicate it's already in legacy format
        has_bin_files = any(f.endswith(".bin") for f in os.listdir(input_path))
        
        if not has_bin_files:
            logger.info("Converting from Meta format to legacy format...")
            legacy_dir = tempfile.mkdtemp(prefix="llama_legacy_")
            
            subprocess.run(
                [sys.executable, "convert.py", input_path, "--outtype", "f16", "--outfile", os.path.join(legacy_dir, "ggml-model-f16.bin")],
                cwd=llama_cpp_dir,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            input_path = legacy_dir
        
        # Now convert from legacy format to GGUF
        logger.info("Converting to GGUF format...")
        
        # Check if there's a convert-llama-ggml-to-gguf.py script
        if os.path.exists(os.path.join(llama_cpp_dir, "convert-llama-ggml-to-gguf.py")):
            # Find the model file
            model_files = [f for f in os.listdir(input_path) if f.startswith("ggml-model")]
            
            if not model_files:
                logger.error(f"No ggml-model files found in {input_path}")
                return False
            
            model_file = os.path.join(input_path, model_files[0])
            
            # Convert to GGUF
            subprocess.run(
                [sys.executable, "convert-llama-ggml-to-gguf.py", model_file, "--outfile", output_path],
                cwd=llama_cpp_dir,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        else:
            # Newer versions may have a different conversion script
            logger.warning("convert-llama-ggml-to-gguf.py not found, trying newer conversion method")
            
            # Try using convert.py with --outtype gguf
            subprocess.run(
                [sys.executable, "convert.py", input_path, "--outtype", "gguf", "--outfile", output_path],
                cwd=llama_cpp_dir,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        
        # Quantize if needed
        if quantize:
            logger.info(f"Quantizing model with {quantize}...")
            temp_output = output_path + ".temp"
            os.rename(output_path, temp_output)
            
            subprocess.run(
                ["./quantize", temp_output, output_path, quantize],
                cwd=llama_cpp_dir,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            os.remove(temp_output)
        
        logger.info(f"Conversion complete. Model saved to {output_path}")
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Conversion failed: {e}")
        logger.error(f"stderr: {e.stderr.decode() if e.stderr else 'none'}")
        return False
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return False
    finally:
        # Clean up temporary directories
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        if 'legacy_dir' in locals() and os.path.exists(legacy_dir):
            shutil.rmtree(legacy_dir)

def convert_torch_to_onnx(input_path, output_path, input_shape=None):
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        input_path: Path to the input PyTorch model
        output_path: Path to save the converted model
        input_shape: Input shape for the model (e.g., [1, 3, 224, 224])
    
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Converting PyTorch model to ONNX format")
    
    try:
        import torch
        import torch.onnx
        
        # Load the model
        model = torch.load(input_path, map_location=torch.device('cpu'))
        model.eval()
        
        # Create dummy input
        if input_shape is None:
            # Default to image input
            input_shape = [1, 3, 224, 224]
        
        dummy_input = torch.randn(*input_shape)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        logger.info(f"Conversion complete. Model saved to {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert models to different formats")
    parser.add_argument("--input", "-i", required=True, help="Input model path")
    parser.add_argument("--output", "-o", required=True, help="Output model path")
    parser.add_argument("--type", "-t", choices=["llama-to-gguf", "torch-to-onnx"], required=True, 
                      help="Type of conversion to perform")
    parser.add_argument("--model-size", default="7b", help="Size of Llama model (7b, 13b, 70b)")
    parser.add_argument("--quantize", default="q4_k_m", 
                      help="Quantization method for GGUF (q4_k_m, q5_k_m, q8_0)")
    parser.add_argument("--input-shape", nargs="+", type=int, 
                      help="Input shape for ONNX conversion (e.g., 1 3 224 224)")
    
    args = parser.parse_args()
    
    if args.type == "llama-to-gguf":
        success = convert_llama_to_gguf(args.input, args.output, args.model_size, args.quantize)
    elif args.type == "torch-to-onnx":
        input_shape = args.input_shape if args.input_shape else None
        success = convert_torch_to_onnx(args.input, args.output, input_shape)
    else:
        logger.error(f"Unknown conversion type: {args.type}")
        return 1
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())