# run_chatbot.py

"""
run_chatbot.py
FastAPI app for processing semantic queries against documents stored in Qdrant vector DB.
"""

from typing import Optional

import uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.custom_logging import logger
from src.query import query_endpoint

# Initialize FastAPI app
app = FastAPI()

# Pydantic Models
class Query(BaseModel):
    """Pydantic model to structure incoming query requests"""
    text: str = Field(..., description="Query text")
    top_k: Optional[int] = 4

@app.post("/query")
async def query(
        request: Query
):
    """API endpoint to process a query using semantic search and LLM synthesis"""
    try:
        logger.info(f"Processing query: {request.text}...")
        return query_endpoint(request)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Example usage for testing
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
