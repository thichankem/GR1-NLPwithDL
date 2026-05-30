"""
Example usage of the LLM query client for students
"""
import os

from dotenv import load_dotenv

from client.models import Query
from client.query import query_model

load_dotenv()

# Example usage for students
def main():
    project_name = os.getenv("GCP_PROJECT_NAME")
    student_email = os.getenv("STUDENT_EMAIL")

    if not project_name or not student_email:
        print("Error: required environment variables not set!")
        print("Please:")
        print("1. Copy .env.template to .env")
        print("2. Edit .env to include GCP_PROJECT_NAME and STUDENT_EMAIL")
        print("3. Re-run this script")
        return
    
    # Create a query with conversation turns
    query = Query(turns=[
        {"user": "Hello! Can you help me understand transformers?"},
    ])
    
    # Query model A
    response = query_model(
        model_id="A",
        query=query
    )
    
    print("=" * 100)
    print(f"Model A Response")
    print("=" * 100)
    print(f"\n{response.text}\n")
    print("=" * 100)
    print(f"Cost: ${response.cost:.8f}")
    print(f"Tokens used: {response.input_tokens} input, {response.output_tokens} output")
    print("=" * 100)
    
    # Multi-turn conversation example with model B
    conversation = Query(turns=[
        {"user": "What is attention in transformers?"},
        {"assistant": "Attention is a mechanism that allows the model to focus on relevant parts of the input when processing each token."},
        {"user": "Can you give me a simple example?"}
    ])
    
    response2 = query_model(
        model_id="B",
        query=conversation
    )
    
    print("\n" + "=" * 100)
    print(f"Model B Response")
    print("=" * 100)
    print(f"\n{response2.text}\n")
    print("=" * 100)
    print(f"Cost: ${response2.cost:.8f}")
    print(f"Tokens used: {response2.input_tokens} input, {response2.output_tokens} output")
    print("=" * 100)

if __name__ == "__main__":
    main()