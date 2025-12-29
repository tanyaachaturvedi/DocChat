import boto3
import json
from typing import List

# Create Bedrock Runtime client
# AWS credentials are automatically loaded from environment, IAM roles, or AWS CLI config
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

def get_embedding(text: str) -> List[float]:
    """
    Generate embeddings for input text using Amazon Titan Embeddings model.
    
    Args:
        text (str): Input text to generate embeddings for
        
    Returns:
        List[float]: Embedding vector as a list of floats
    """
    try:
        response = bedrock_client.invoke_model(
            modelId="amazon.titan-embed-text-v1",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": text})
        )

        result = json.loads(response["body"].read())
        return result["embedding"]
    
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise


def generate_text(prompt: str) -> str:
    """
    Generate text response using Amazon Titan Text Express model.
    
    Args:
        prompt (str): Input prompt for text generation
        
    Returns:
        str: Generated text response
    """
    try:
        response = bedrock_client.invoke_model(
            modelId="amazon.titan-text-express-v1",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": prompt})
        )

        result = json.loads(response["body"].read())
        return result["results"][0]["outputText"]
    
    except Exception as e:
        print(f"Error generating text: {e}")
        raise
