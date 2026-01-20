#!/usr/bin/env python3
"""
Local VLM Server Extension
Adds a local file path endpoint to mlx-vlm server
"""

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from pathlib import Path
from PIL import Image
from typing import Optional, List, Union, Literal
import os
import gc
import json
import time
import io
import sys

# Force offline mode for HuggingFace Hub
# This prevents network access when loading models from local cache
OFFLINE_MODE = os.getenv("OFFLINE_MODE", "0").lower() in ("1", "true", "yes")
if OFFLINE_MODE:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print("Running in OFFLINE mode - models will be loaded from local cache only", file=sys.stderr)

# --- Imports for Streaming ---
from fastapi.responses import StreamingResponse
import asyncio

# --- Library Imports ---
# --- Library Imports ---
import mlx_vlm
from mlx_vlm import load as load_vlm, generate as generate_vlm
from mlx_vlm.prompt_utils import apply_chat_template as apply_chat_template_vlm
from mlx_vlm.utils import load_config as load_config_vlm

from mlx_vlm.utils import load_config as load_config_vlm

# Need stream_generate from libs
from mlx_vlm import stream_generate as stream_generate_vlm

# Try importing mlx_lm for text-only models
try:
    import mlx_lm
    from mlx_lm import load as load_lm, generate as generate_lm
    from mlx_lm import stream_generate as stream_generate_lm
    HAS_MLX_LM = True
except ImportError:
    HAS_MLX_LM = False
    print("Warning: mlx_lm not found. Text-only models will not be supported.")

# Global model cache (Single Active Model)
# Value: (model, processor_or_tokenizer, config, backend_type: "vlm" | "lm")
_model_cache = {}

# Default model (can be overridden by VLM_MODEL env var)
DEFAULT_MODEL = "mlx-community/Qwen3-VL-4B-Instruct-4bit"

# Allowed models (from env var)
ALLOWED_MODELS_ENV = os.getenv("ALLOWED_MODELS", "")
ALLOWED_MODELS = set(filter(None, ALLOWED_MODELS_ENV.split(";")))

# --- OpenAI Request Models ---

class ChatMessageContentText(BaseModel):
    type: Literal["text"]
    text: str

class ChatMessageContentImageUrl(BaseModel):
    url: str

class ChatMessageContentImage(BaseModel):
    type: Literal["image_url"]
    image_url: ChatMessageContentImageUrl

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Union[ChatMessageContentText, ChatMessageContentImage]]]

class ChatCompletionRequest(BaseModel):
    model: str = Field(
        default_factory=lambda: os.getenv("VLM_MODEL", DEFAULT_MODEL)
    )
    messages: List[ChatMessage]
    max_tokens: int = Field(default=4096, ge=1, le=8192)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    stream: bool = False  # Add stream parameter

class ChatCompletionChoiceDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChoiceDelta
    finish_reason: Optional[str] = None

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]




class ChatCompletionChoiceMessage(BaseModel):
    role: str
    content: str

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionChoiceMessage
    finish_reason: str

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage

# --- Helper Functions ---

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


def unload_model_if_needed(new_model_id: str):
    """
    Ensure only one model stays in memory.
    If a different model is requested, unload the current one.
    """
    global _model_cache
    current_models = list(_model_cache.keys())
    
    for model_id in current_models:
        if model_id != new_model_id:
            print(f"Unloading model: {model_id} to free memory for {new_model_id}")
            del _model_cache[model_id]
            
            # Heavy cleanup
            gc.collect()
            try:
                import mlx.core as mx
                if hasattr(mx, "metal"):
                    mx.metal.clear_cache()
            except ImportError:
                pass

def extract_content_and_images(messages: List[ChatMessage]):
    # ... (Same logic as before) ...
    """
    Parses OpenAI messages to extract text prompt and local images.
    Returns: (text_prompt_for_mlx, list_of_PIL_images)
    """
    full_prompt_structure = []
    pil_images = []

    # Iterate messages to build up context
    for msg in messages:
        role = msg.role
        content = msg.content
        
        message_text = ""
        
        if isinstance(content, str):
            message_text = content
        elif isinstance(content, list):
            # Mixed content (text + images)
            for part in content:
                if isinstance(part, ChatMessageContentText) or (isinstance(part, dict) and part.get("type") == "text"):
                    # Pydantic or Dict (API tolerant)
                    text_val = part.text if isinstance(part, ChatMessageContentText) else part.get("text", "")
                    message_text += text_val
                elif isinstance(part, ChatMessageContentImage) or (isinstance(part, dict) and part.get("type") == "image_url"):
                    # Extract Image
                    img_url_obj = part.image_url if isinstance(part, ChatMessageContentImage) else part.get("image_url", {})
                    url_str = img_url_obj.url if hasattr(img_url_obj, 'url') else img_url_obj.get("url", "")
                    
                    # Handle Local File Check
                    if url_str.startswith("http"):
                        raise HTTPException(status_code=400, detail="Only local file paths are supported in image_url")
                    
                    if url_str.startswith("file://"):
                        local_path = url_str[7:]
                    else:
                        local_path = url_str
                        
                    p = Path(local_path)
                    if not p.exists():
                        raise HTTPException(status_code=404, detail=f"Image not found at local path: {local_path}")
                    
                    try:
                        pil_img = compress_image(str(p))
                        pil_images.append(pil_img)
                    except Exception as e:
                        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

        # Add to structure for template
        full_prompt_structure.append({"role": role, "content": message_text})
        
    return full_prompt_structure, pil_images


def create_local_infer_app():
    """Create FastAPI app with local inference endpoint"""
    app = FastAPI(title="MLX-VLM Unified Server")
    
    
    @app.post("/v1/chat/completions", response_model=Union[ChatCompletionResponse, str])
    async def chat_completions(
        request: ChatCompletionRequest,
        authorization: Optional[str] = Header(None, alias="Authorization")
    ):
        # ... (Auth and Access Control same as before - lines 130-144) ...
        # 0. API Key Validation
        server_api_key = os.getenv("VLM_API_KEY")
        if server_api_key:
            if not authorization:
                raise HTTPException(status_code=401, detail="Missing Authorization header")
            scheme, _, token = authorization.partition(" ")
            if scheme.lower() != "bearer" or token != server_api_key:
                 raise HTTPException(status_code=401, detail="Invalid API Key")

        # 1. Access Control
        if ALLOWED_MODELS and request.model not in ALLOWED_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' is not allowed. Allowed models: {sorted(list(ALLOWED_MODELS))}"
            )

        # 2. Parse Messages & Images
        try:
            mlx_messages, pil_images = extract_content_and_images(request.messages)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Message parsing failed: {str(e)}")

        try:
            # 3. Memory Management
            unload_model_if_needed(request.model)

            # 4. Load Model (Hybrid Logic)
            model_id = request.model
            
            if model_id not in _model_cache:
                print(f"Loading model: {model_id}")
                
                # Try VLM first
                backend_type = "vlm"
                model, processor, config = None, None, None
                
                try:
                    # Attempt load with mlx_vlm
                    model, processor = load_vlm(model_id)
                    config = load_config_vlm(model_id)
                    backend_type = "vlm"
                except Exception as e_vlm:
                    # If VLM load fails, try LLM if available
                    if HAS_MLX_LM:
                        print(f"VLM load failed ({e_vlm}), trying LLM load...")
                        try:
                            model, tokenizer = load_lm(model_id)
                            # For LLM, 'processor' is tokenizer because mlx_lm uses tokenizer
                            processor = tokenizer 
                            # mlx_lm models might not expose config directly or in same way. We don't strictly need it for generate().
                            config = getattr(model, "config", None) 
                            backend_type = "lm"
                        except Exception as e_lm:
                            raise HTTPException(status_code=500, detail=f"Failed to load model as VLM or LLM. VLM Error: {e_vlm}, LLM Error: {e_lm}")
                    else:
                        raise HTTPException(status_code=500, detail=f"Failed to load VLM model: {e_vlm}")

                _model_cache[model_id] = (model, processor, config, backend_type)
            
            # Retrieve from cache
            model, processor, config, backend_type = _model_cache[model_id]

            # 5. Validation
            if backend_type == "lm" and pil_images:
                raise HTTPException(status_code=400, detail=f"Model '{model_id}' is a text-only model (loaded via mlx_lm). It does not support image inputs.")

            # 6. Prepare Prompt
            if backend_type == "vlm":
                formatted_prompt = apply_chat_template_vlm(
                    processor,
                    config,
                    json.dumps(mlx_messages), 
                    num_images=len(pil_images),
                )
            elif backend_type == "lm":
                if hasattr(processor, "apply_chat_template"):
                     formatted_prompt = processor.apply_chat_template(
                         mlx_messages, 
                         tokenize=False, 
                         add_generation_prompt=True
                     )
                else:
                     formatted_prompt = "\n".join([m['content'] for m in mlx_messages])

            # 7. Streaming Logic
            if request.stream:
                async def stream_generator():
                    chat_id = f"chatcmpl-{int(time.time())}"
                    created_ts = int(time.time())
                    
                    # Generator Selection
                    if backend_type == "vlm":
                        # mlx_vlm.stream_generate yields objects with .text, .token, .logprobs?
                        # Usually yields: (token: int, text: str) tuple or similar?
                        # Check MLX VLM source or examples. 
                        # usually `stream_generate(model, processor, prompt, images, ...)` yields Detokenizer object or text chunks.
                        # It yields Output objects with text.
                        gen = stream_generate_vlm(
                            model, 
                            processor, 
                            formatted_prompt, 
                            pil_images if pil_images else None, 
                            max_tokens=request.max_tokens, 
                            temperature=request.temperature
                        )
                    else:
                        # mlx_lm.stream_generate(model, tokenizer, prompt, max_tokens, ...)
                        gen = stream_generate_lm(
                            model, 
                            processor, 
                            formatted_prompt, 
                            max_tokens=request.max_tokens, 
                            temp=request.temperature # mlx_lm uses 'temp' usually? wait, we checked generate uses kwargs. stream_generate?
                            # Let's assume temp=... works or verify signature.
                            # Standard mlx_lm.stream_generate(model, tokenizer, prompt, max_tokens, verbose=True/False, **kwargs)
                        )
                    
                    # Yield initial role
                    chunk_data = ChatCompletionChunk(
                        id=chat_id, created=created_ts, model=model_id,
                        choices=[ChatCompletionChunkChoice(index=0, delta=ChatCompletionChoiceDelta(role="assistant"))]
                    )
                    yield f"data: {chunk_data.json()}\n\n"

                    try:
                        for response in gen:
                            # Response structure depends on library
                            text_chunk = ""
                            if backend_type == "vlm":
                                # mlx_vlm stream yields an object with `text` attribute usually
                                if hasattr(response, "text"):
                                    text_chunk = response.text
                                else:
                                    # Fallback if it yields string? 
                                    text_chunk = str(response) 
                            else:
                                # mlx_lm stream yields objects with `text` attribute?
                                # help(mlx_lm.stream_generate) -> yield (token, text) tuple? Or just text string?
                                # Usually yields "GenerationOutput" or just string segments.
                                # IMPORTANT: mlx_lm 0.28+ yields Simple Object or named tuple?
                                # Let's assume it yields an object with .text or is a string.
                                if isinstance(response, str):
                                    text_chunk = response
                                elif hasattr(response, "text"):
                                    text_chunk = response.text
                                else:
                                    text_chunk = "" # Unsure

                            if text_chunk:
                                chunk_data = ChatCompletionChunk(
                                    id=chat_id, created=created_ts, model=model_id,
                                    choices=[ChatCompletionChunkChoice(index=0, delta=ChatCompletionChoiceDelta(content=text_chunk))]
                                )
                                yield f"data: {chunk_data.json()}\n\n"
                                # Yield to loop to allow async behavior? 
                                # Since this is blocking sync generator running in async function, it might block event loop.
                                # But FastAPI handles sync generators in StreamingResponse by running in threadpool?
                                # Actually StreamingResponse takes an iterator.
                                await asyncio.sleep(0) # Non-blocking yield attempt

                    except Exception as e:
                        print(f"Streaming error: {e}")
                        # Maybe yield error?
                    
                    # Yield finish
                    chunk_data = ChatCompletionChunk(
                        id=chat_id, created=created_ts, model=model_id,
                        choices=[ChatCompletionChunkChoice(index=0, delta=ChatCompletionChoiceDelta(), finish_reason="stop")]
                    )
                    yield f"data: {chunk_data.json()}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(stream_generator(), media_type="text/event-stream")

            # 8. Non-Streaming Generation (Existing Logic)
            output_text = ""
            usage_stats = ChatCompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

            if backend_type == "vlm":
                # ... same as before
                output = generate_vlm(
                    model,
                    processor,
                    formatted_prompt,
                    pil_images if pil_images else None,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    verbose=False
                )
                output_text = output.text
                usage_stats.prompt_tokens = output.prompt_tokens
                usage_stats.completion_tokens = output.generation_tokens
                usage_stats.total_tokens = output.total_tokens

            elif backend_type == "lm":
                # ... same as before
                output_text = generate_lm(
                    model,
                    processor,
                    formatted_prompt,
                    max_tokens=request.max_tokens,
                    verbose=True
                )
                usage_stats.completion_tokens = len(output_text.split())

            return ChatCompletionResponse(
                id=f"chatcmpl-{int(time.time())}",
                created=int(time.time()),
                model=model_id,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatCompletionChoiceMessage(
                            role="assistant",
                            content=output_text
                        ),
                        finish_reason="stop"
                    )
                ],
                usage=usage_stats
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/health")
    async def health():
        return {"status": "healthy"}
    
    return app

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("VLM_PORT", "58080"))
    app = create_local_infer_app()
    uvicorn.run(app, host="127.0.0.1", port=port)
