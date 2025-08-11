from typing import List, Optional
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    repo_name: str = Field(..., description="Short repo folder name, e.g., scikit-learn")
    text: str = Field(..., description="User query text")
    k: int = Field(3, ge=1, description="Top-k results to retrieve")
    model: str = Field(default="gemini-1.5-flash", description="Model name for generation")
    temperature: float = Field(0.2, ge=0.0, le=1.0, description="Temperature for generation")
    max_output_tokens: int = Field(500, ge=64, le=4096, description="Maximum output tokens for the answer")

class ContextItem(BaseModel):
    file_path: str = Field(..., description="Path to the file containing the context")
    chunk_id: str = Field(..., description="Unique identifier for the context chunk")
    distance: float = Field(..., description="Distance score for the context")
    text: Optional[str] = Field(None, description="Text content of the context")

class AnswerResponse(BaseModel):
    answer: str = Field(..., description="Generated answer based on the query and contexts")
    contexts: List[ContextItem] = Field(..., description="List of context items used to generate the answer")
    model: str = Field(default="gemini-1.5-flash", description="Model name for generation")
    k: int = Field(3, ge=1, description="Top-k results used for generating the answer")
    latency_ms: Optional[int] = Field(None, description="Latency in milliseconds for the generation request")