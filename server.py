import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from document_loader import VectorStoreManager
from chat_logic import ChatRequest, ContextProcessor, ResponseGenerator

import ollama

app = FastAPI()
vector_store_manager = VectorStoreManager()

# Enable CORS for all origins (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    await vector_store_manager.initialize()
    yield
    # Add any necessary cleanup code here

app.router.lifespan_context = lifespan

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "operational",
        "vector_store": "ready" if vector_store_manager.vector_store else "not_ready"
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    """Retrieves relevant context and generates an answer using Ollama."""
    if vector_store_manager.vector_store is None:
        return {"response": "Vector store not loaded yet. Please try again in a moment."}
    
    try:
        # Stage 1: Analyze the question
        analysis = await ContextProcessor.analyze_question(request.question)
        
        # Stage 2: Retrieve context using similarity search (with fallback filtering)
        retrieved_docs = []
        for topic in analysis.topics:
            if analysis.required_context_types:
                docs = vector_store_manager.vector_store.similarity_search(
                    topic,
                    k=3,
                    filter={"confidence": {"$gte": analysis.confidence_threshold}}
                    if analysis.confidence_threshold else None
                )
                retrieved_docs.extend(docs)
        
        if not retrieved_docs:
            for topic in analysis.topics:
                docs = vector_store_manager.vector_store.similarity_search(topic, k=3, filter=None)
                retrieved_docs.extend(docs)
        
        # Deduplicate documents based on content
        seen_content = set()
        unique_docs = []
        for doc in retrieved_docs:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_docs.append(doc)
        
        if not unique_docs:
            return {
                "response": "I apologize, but I don't have enough context to properly answer your question. Could you please rephrase or provide more details?",
                "debug_info": {
                    "analysis": analysis.__dict__,
                    "context_count": 0
                } if request.include_debug_info else None
            }
        
        # Stage 3: Format the retrieved context and generate the prompt
        formatted_context = ResponseGenerator.format_context(
            [{"content": doc.page_content, "metadata": doc.metadata} for doc in unique_docs]
        )
        prompt = ResponseGenerator.generate_prompt(request.question, formatted_context, analysis)
        
        # Stage 4: Get the response from Ollama
        response = ollama.chat(
            model="mistral:latest",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        
        return {
            "response": response["message"]["content"],
            "debug_info": {
                "analysis": analysis.__dict__,
                "context_count": len(unique_docs),
                "topics": analysis.topics,
                "required_context_types": analysis.required_context_types
            } if request.include_debug_info else None
        }
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)