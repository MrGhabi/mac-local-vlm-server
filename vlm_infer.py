#!/usr/bin/env python3
"""
VLM Inference Script for local-llm-server

This script provides direct inference using mlx-vlm without starting a server.
It outputs results in OpenAI-compatible JSON format.
"""

import argparse
import json
import sys
import io
from pathlib import Path

try:
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    from PIL import Image
except ImportError as e:
    print(json.dumps({
        "error": str(e),
        "message": "mlx-vlm not installed. Please install it first."
    }), file=sys.stderr)
    sys.exit(1)


def compress_image(image_path: str) -> Image.Image:
    """
    Compress and optimize image for VLM inference.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Compressed PIL Image object
        
    Raises:
        FileNotFoundError: If image file doesn't exist
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    original_image = Image.open(image_path)
    
    # Smart compression to keep image within reasonable size
    # Target: max 1920x1080 or 2MB file size
    max_dimension = 1920
    max_file_size_mb = 2
    
    # Get original size
    width, height = original_image.size
    print(f"Original image size: {width}x{height}", file=sys.stderr)
    
    # Resize if too large
    if width > max_dimension or height > max_dimension:
        # Keep aspect ratio
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        
        print(f"Resizing image to: {new_width}x{new_height}", file=sys.stderr)
        compressed_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    else:
        compressed_image = original_image
    
    # Convert to RGB if needed (remove alpha channel)
    if compressed_image.mode != 'RGB':
        print(f"Converting image from {compressed_image.mode} to RGB", file=sys.stderr)
        compressed_image = compressed_image.convert('RGB')
    
    # Check file size by saving to memory
    buffer = io.BytesIO()
    compressed_image.save(buffer, format='JPEG', quality=85, optimize=True)
    size_mb = len(buffer.getvalue()) / (1024 * 1024)
    
    # If still too large, reduce quality
    quality = 85
    while size_mb > max_file_size_mb and quality > 50:
        buffer = io.BytesIO()
        quality -= 10
        compressed_image.save(buffer, format='JPEG', quality=quality, optimize=True)
        size_mb = len(buffer.getvalue()) / (1024 * 1024)
    
    print(f"Compressed image size: {size_mb:.2f}MB (quality: {quality})", file=sys.stderr)
    
    return compressed_image


def main():
    # Force offline mode for HuggingFace Hub
    # This prevents network access when loading models from local cache
    import os

    # Set HF_HOME to project data directory if not already set
    if "HF_HOME" not in os.environ:
        script_dir = Path(__file__).parent
        hf_cache = script_dir / "data"
        os.environ["HF_HOME"] = str(hf_cache)

    offline_mode = os.getenv("OFFLINE_MODE", "1").lower() in ("1", "true", "yes")
    if offline_mode:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print("Running in OFFLINE mode - models will be loaded from local cache only", file=sys.stderr)
    
    parser = argparse.ArgumentParser(description="VLM Direct Inference")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--model", default="mlx-community/Qwen3-VL-4B-Instruct-4bit", help="Model ID")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    
    args = parser.parse_args()
    
    try:
        # Load model
        print(f"Loading model: {args.model}", file=sys.stderr)
        model, processor = load(args.model)
        config = load_config(args.model)
        
        # Load and compress image
        compressed_image = compress_image(args.image)
        image = [compressed_image]
        
        # Format prompt
        formatted_prompt = apply_chat_template(
            processor,
            config,
            args.prompt,
            num_images=len(image),
        )
        
        # Generate
        print("Generating...", file=sys.stderr)
        output = generate(
            model,
            processor,
            formatted_prompt,
            image,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            verbose=False
        )
        
        # Format as OpenAI-compatible JSON
        result = {
            "choices": [{
                "message": {
                    "content": output.text,
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": output.prompt_tokens,
                "completion_tokens": output.generation_tokens,
                "total_tokens": output.total_tokens
            },
            "model": args.model
        }
        
        # Output JSON to stdout
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
    except Exception as e:
        error_result = {
            "error": str(e),
            "type": type(e).__name__
        }
        print(json.dumps(error_result, ensure_ascii=False, indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
