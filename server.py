import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from document_loader import ResumeDocumentProcessor  # Updated to correct class
from chat_logic import ChatRequest, ContextProcessor, ResponseGenerator

import ollama
import os

app = FastAPI()

# Initialize resume processor
resume_processor = ResumeDocumentProcessor()

# Enable CORS for frontend requests (adjust as needed)
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
    print("Initializing vector store...")
    processed_docs_count = await resume_processor.initialize_vector_store(docs_folder="docs")
    print(f"Loaded {processed_docs_count} documents into vector store.")
    yield
    print("Shutting down server...")

app.router.lifespan_context = lifespan

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "operational",
        "vector_store": "ready" if resume_processor.vector_store else "not_ready"
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    """Retrieves relevant context and generates an answer using Ollama."""
    if resume_processor.vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not loaded yet. Please try again later.")
    
    try:
        # 🔹 Step 1: Analyze the question
        analysis = await ContextProcessor.analyze_question(request.question)
        print("\n Question Analysis Summary:")
        print(analysis.__dict__)  # Debug print analysis

        # 🔹 Step 2: Retrieve relevant experience and context
        relevant_contexts = await ContextProcessor.get_relevant_context(analysis, resume_processor.vector_store)
        
        if not relevant_contexts:
            print("\n No relevant contexts found.")
            return {
                "response": "I don't have enough context to properly answer your question. Please provide more details or rephrase.",
                "debug_info": {
                    "analysis": analysis.__dict__,
                    "context_count": 0
                }
            }
        
        # Print retrieved context chunks
        print("\n Retrieved Context Chunks:")
        for idx, context in enumerate(relevant_contexts):
            print(f"\n🔹 Chunk {idx+1}:")
            print(f"Source: {context['source']}")
            print(f"Content: {context['content'][:500]}...")  # Print first 500 chars for readability

        # 🔹 Step 3: Format retrieved context and generate a response
        formatted_context = ResponseGenerator.format_context(relevant_contexts)
        prompt = ResponseGenerator.generate_prompt(request.question, formatted_context, analysis)

        # Print final prompt before sending to Ollama
        print("\n Final Generated Prompt:")
        print(prompt)

        # 🔹 Step 4: Generate response using Ollama
        response = ollama.chat(
            model="mistral:latest",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )

        return {
            "response": response["message"]["content"],
            "debug_info": {
                "analysis": analysis.__dict__,
                "context_count": len(relevant_contexts),
                "focus_area": analysis.focus_area,
                "skills_mentioned": analysis.skills_mentioned
            }
        }
        
    except Exception as e:
        print(f"\n Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
