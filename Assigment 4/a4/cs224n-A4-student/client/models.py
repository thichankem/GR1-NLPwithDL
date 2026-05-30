"""
Shared models between client and server.
Students will receive this file along with query.py
"""
from typing import List, Dict, Literal
from pydantic import BaseModel, Field

class Query(BaseModel):
    # Use default_factory to avoid sharing the same list across instances.
    turns: List[Dict[Literal["user", "assistant"], str]] = Field(default_factory=list)

class QueryResponse(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int
    cost: float

class ClientQuery(BaseModel):
    model_config = {"protected_namespaces": ()}
    # ^ For surpressing namespace warning

    query: Query
    model_id: str
    # No per-student API key is used in this assignment setup.
