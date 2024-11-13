import torch
import torch.nn.functional as F
from transformers import LLaMA2, LLava2

# Define the full pipeline function
def multimodal_attention_pipeline(text_input, image_input):
    # Pre-trained models
    text_encoder = LLaMA2.from_pretrained('LLaMA2-base')
    image_encoder = LLava2.from_pretrained('LLava2-base')
    
    # Encode text and image
    text_embedding = text_encoder.encode(text_input)
    image_embedding = image_encoder.encode(image_input)
    
    # Define projections
    query_projection = torch.nn.Linear(text_embedding.size(-1), 128)
    key_projection = torch.nn.Linear(image_embedding.size(-1), 128)
    value_projection = torch.nn.Linear(image_embedding.size(-1), 128)
    
    # Generate query, key, and value vectors
    queries = query_projection(text_embedding)
    keys = key_projection(image_embedding)
    values = value_projection(image_embedding)
    
    # Compute attention scores
    attention_scores = compute_attention_scores(queries, keys)
    attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
    
    # Aggregate using attention weights
    combined_features = aggregate_attention(values, attention_weights)
    
    return combined_features

def compute_attention_scores(queries, keys):
    # Scale dot-product
    d_k = keys.size(-1)
    scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(d_k)
    return scores

def aggregate_attention(values, attention_weights):
    aggregated_embeddings = torch.matmul(attention_weights, values)
    return aggregated_embeddings

# Example inputs
text_input = "Example medical text"
image_input = torch.rand((1, 3, 224, 224))  # Example image tensor

# Execute the pipeline
output = multimodal_attention_pipeline(text_input, image_input)
print("Combined Features: ", output)