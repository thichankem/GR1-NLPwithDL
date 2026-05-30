from client.models import Query, QueryResponse
from client.providers import get_provider

def query_model(
    model_id: str,
    query: Query,
) -> QueryResponse:

    # Get the appropriate provider
    provider_name, provider = get_provider(model_id=model_id)
    
    # Make the request to the LLM provider
    response: QueryResponse = provider.query(
        model_id=model_id, 
        query=query,
    )
    
    return response 