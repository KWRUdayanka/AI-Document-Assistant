# src/query.py

from dotenv import load_dotenv
from fastapi import FastAPI
from llama_index.core.schema import TextNode, NodeWithScore
from pydantic import BaseModel
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import get_response_synthesizer, Settings
from qdrant_client import QdrantClient
import os
from typing import List, Dict, Optional
from custom_logging import logger

# Load environment variables
load_dotenv()

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")

# Initialize FastAPI app
app = FastAPI()

# Initialize Qdrant client
try:
    qdrant_client = QdrantClient("localhost", prefer_grpc=True)
    logger.info("Successfully initialized Qdrant client")
except Exception as e:
    logger.error(f"Failed to initialize Qdrant client: {e}")
    raise

# Initialize embedding model
try:
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
    Settings.embed_model = embed_model
    Settings.llm = llm
    logger.info("Successfully initialized embedding model and LLM")
except Exception as e:
    logger.error(f"Failed to initialize embedding model: {e}")
    raise

# Initialize response synthesizer
response_synthesizer = get_response_synthesizer(
    llm=llm,
    response_mode="refine"
)

# Pydantic model for query request
class QueryRequest(BaseModel):
    text: str
    top_k: Optional[int] = 4  # Make top_k optional with default value

# Function to retrieve documents
def retrieve_documents(query: str, collection_name: str, top_k: int = 5) -> List[Dict]:
    """
        Retrieve relevant documents from Qdrant based on a query.

        Args:
            query (str): The query string to search for.
            collection_name (str): The name of the Qdrant collection to search in.
            top_k (int, optional): Number of top results to retrieve. Defaults to 5.

        Returns:
            List[Dict]: A list of dictionaries containing retrieved document information
                       including text, document_id, chunk_index, and score.

        Raises:
            Exception: If there is an error during document retrieval.
    """
    try:
        # Generate query embedding
        query_vector = embed_model.get_text_embedding(query)
        logger.info(f"Generated query embedding for: {query}")

        # Perform semantic search
        results = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True
        )

        # Handle results.points, which may be a tuple
        points = results.points
        if isinstance(points, tuple):
            logger.debug(f"Results.points is a tuple: {points}")
            points = points[0]  # Extract the list of points
        else:
            logger.debug(f"Results.points type: {type(points)}, content: {points}")

        # Extract relevant document information
        retrieved_docs = []
        for hit in points:
            try:
                payload = getattr(hit, 'payload', None)
                score = getattr(hit, 'score', None)
                if payload is None or score is None:
                    logger.warning(f"Invalid point structure: {hit}")
                    continue
                retrieved_docs.append({
                    "text": payload.get("text"),
                    "document_id": payload.get("document_id"),
                    "chunk_index": payload.get("chunk_index"),
                    "score": score
                })
            except AttributeError as e:
                logger.error(f"Error processing point: {e}")
                continue
        logger.info(f"Retrieved {len(retrieved_docs)} documents from {collection_name}")
        return retrieved_docs
    except Exception as e:
        logger.error(f"Error retrieving documents from {collection_name}: {e}")
        return []

# Function to generate response using RAG
def generate_response(query: str, retrieved_docs: List[Dict]) -> str:
    """
        Generate a response using RAG (Retrieval-Augmented Generation) based on retrieved documents.

        Args:
            query (str): The input query to generate a response for.
            retrieved_docs (List[Dict]): List of retrieved documents with their metadata.

        Returns:
            str: The generated response based on the query and retrieved documents.

        Raises:
            Exception: If there is an error during response generation.
    """
    try:
        # Convert retrieved documents to NodeWithScore objects
        nodes = []
        for doc in retrieved_docs:
            if not doc["text"]:
                logger.warning(f"Skipping document with empty text: {doc}")
                continue
            node = TextNode(
                text=doc["text"],
                metadata={
                    "document_id": doc["document_id"],
                    "chunk_index": doc["chunk_index"]
                }
            )
            node_with_score = NodeWithScore(node=node, score=doc["score"])
            nodes.append(node_with_score)
        if not nodes:
            logger.warning("No valid nodes available for response generation")
            return "No relevant information found."
        # Generate response using response synthesizer
        response = response_synthesizer.synthesize(
            query=query,
            nodes=nodes,
            additional_prompt=f"""
                    Provide a concise, full-sentence answer to the query based on the provided context.
                    Query: {query}
                    """
        )
        logger.info(f"Generated response for query: {query}")
        return str(response)
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Error generating response."



# Query endpoint
def query_endpoint(
        request: QueryRequest
):
    """
        FastAPI endpoint to handle query requests and return responses with retrieved documents.

        Args:
            request (QueryRequest): The query request containing the query text and optional top_k parameter.

        Returns:
            dict: A dictionary containing the query, generated answer, retrieved documents,
                  and a message indicating the number of documents retrieved.

        Raises:
            Exception: If the specified collection does not exist or there is an error during processing.
    """
    collection_name = COLLECTION_NAME
    try:
        # Verify collection exists
        qdrant_client.get_collection(collection_name)
    except Exception as e:
        logger.error(f"Collection {collection_name} does not exist: {e}")

    # Retrieve documents
    retrieved_docs = retrieve_documents(
        query=request.text,
        collection_name=collection_name,
        top_k=request.top_k
    )

    # Generate response
    answer = generate_response(request.text, retrieved_docs)

    if not retrieved_docs:
        logger.warning(f"No documents retrieved for query: {request.text}")
        return {
            "query": request.text,
            "answer": answer,
            "documents": [],
            "message": "No relevant documents found"
        }

    return {
        "query": request.text,
        "answer": answer,
        "documents": retrieved_docs,
        "message": f"Retrieved {len(retrieved_docs)} documents"
    }
