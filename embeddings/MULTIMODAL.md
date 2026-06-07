# Multimodal Image Embeddings Guide (CLIP)

This guide explains how to extract embeddings directly from images using **CLIP (Contrastive Language-Image Pre-training)**. 

Unlike text-only embedding models, CLIP embeds both **images** and **text** into the same high-dimensional vector space, allowing you to perform semantic operations like:
- **Image-to-Image Search**: Finding images similar to a query image.
- **Text-to-Image Search**: Finding images matching a natural language description (e.g., "a red sports car").
- **Zero-Shot Classification**: Categorizing images without explicit training.

---

## Setup & Prerequisites

CLIP runs locally via the Hugging Face `transformers` library (already supported by your platform backend environment). Ensure the required dependencies are installed:

```bash
pip install torch transformers pillow numpy
```

---

## Integration Code

Below is a complete, copy-pasteable script (`clip_embeddings.py`) showing how to generate embeddings for both images and text queries, and how to compute the semantic similarity between them.

```python
#!/usr/bin/env python3
import os
import argparse
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Configuration
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# Lazy-loaded globals
_model = None
_processor = None
_device = None

def get_clip_resources():
    """Initialize and cache the CLIP model and processor on the active device."""
    global _model, _processor, _device
    if _model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Initializing CLIP on: {_device.upper()}")
        
        # Load the model and processor from Hugging Face cache
        _model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(_device)
        _processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    return _model, _processor, _device

def get_image_embedding(image_path: str) -> list:
    """Generate a normalized feature vector directly from an image file."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
        
    model, processor, device = get_clip_resources()
    
    # 1. Load and convert image to RGB
    image = Image.open(image_path).convert("RGB")
    
    # 2. Preprocess image
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # 3. Extract features
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        
    # 4. L2 Normalize the embedding (critical for cosine similarity comparison)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    
    return image_features[0].cpu().numpy().tolist()

def get_text_embedding(text: str) -> list:
    """Generate a normalized feature vector for text in the same CLIP vector space."""
    model, processor, device = get_clip_resources()
    
    # Preprocess text
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    
    # Extract features
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        
    # L2 Normalize
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    
    return text_features[0].cpu().numpy().tolist()

def calculate_cosine_similarity(vec1: list, vec2: list) -> float:
    """Calculate the similarity index (dot product since vectors are normalized)."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    # Cosine similarity is simply the dot product when vectors are L2 normalized
    return float(np.dot(v1, v2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and match CLIP multimodal embeddings.")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--queries", nargs="+", default=["a photo of a cat", "a photo of a dog", "scenery"],
                        help="List of text descriptions to test similarity against")
    args = parser.parse_args()
    
    try:
        # 1. Extract image vector
        img_vector = get_image_embedding(args.image)
        print(f"\n[Success] Generated image embedding vector (Dimensions: {len(img_vector)})")
        
        # 2. Compare against text prompts
        print(f"\nComparing image '{args.image}' against queries:")
        print("-" * 50)
        
        for query in args.queries:
            txt_vector = get_text_embedding(query)
            similarity = calculate_cosine_similarity(img_vector, txt_vector)
            print(f"Similarity with '{query}': {similarity:.4f}")
            
    except Exception as e:
        print(f"[Error] Execution failed: {e}")
```

---

## Use Cases

### 1. Vector Database Ingestion (e.g., ChromaDB, Qdrant, Milvus)
When saving records to your database, generate the image embedding using `get_image_embedding()` and save it in your vector collection.
* Since CLIP embeddings are `512` dimensional (for `clip-vit-base-patch32`), configure your vector database collections to use **512 dimensions** with **Cosine Distance**.

### 2. Search / Retrieval Query Flow
For **Text-to-Image Search** (search for photos matching a user prompt):
1. User enters: `"sunsets over ocean"`
2. Convert that string into a vector: `query_vector = get_text_embedding("sunsets over ocean")`
3. Execute a vector search inside your database using `query_vector`.
4. The database returns the closest image records.

---

## CLIP Model Alternatives

Depending on accuracy requirements and hardware restrictions, you can swap the `CLIP_MODEL_ID` at the top of the file:

| Model ID | VRAM Usage | Accuracy / Performance |
| :--- | :--- | :--- |
| `openai/clip-vit-base-patch32` *(Default)* | ~600 MB | Fast, lightweight, standard accuracy. |
| `openai/clip-vit-large-patch14` | ~1.7 GB | High visual resolution detail, higher accuracy. |
| `google/siglip-base-patch16-224` | ~800 MB | State-of-the-art multilingual-friendly embedding quality. |
