"""
Enhanced Context-Aware Chatbot
-----------------------------
This implementation provides a sophisticated chatbot that processes documents,
analyzes questions, and generates context-aware responses using a multi-stage approach.
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import ollama
import json
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Configuration
CONTEXT_FOLDER = "docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MODEL_NAME = "mistral:latest"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Data Models
@dataclass
class DocumentMetadata:
    """Metadata for context documents"""
    category: str  # e.g., 'experience', 'technical', 'metadata'
    subcategory: str  # e.g., 'projects', 'skills', 'architecture'
    date_range: Optional[str]  # e.g., '2020-2022' or None
    confidence: float  # Reliability score of the information (0-1)

class ChatRequest(BaseModel):
    question: str
    include_debug_info: bool = False  # Optional flag for debugging information

@dataclass
class QuestionAnalysis:
    """Structured analysis of user questions"""
    original_question: str
    topics: List[str] = None
    required_context_types: List[str] = None
    temporal_context: Optional[str] = None
    confidence_threshold: float = 0.7

class VectorStoreManager:
    """Manages document processing and vector store operations"""
    
    def __init__(self):
        self.vector_store = None
        self.embeddings = None
        
    async def initialize(self):
        """Initialize embeddings and vector store"""
        print("Initializing Vector Store...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        await self.load_documents()
        print("Vector Store Initialization Complete!")
    
    async def load_documents(self):
        """Load and process all documents from flat file structure"""
        all_documents = []
        
        # Get all files in the docs directory
        for filename in os.listdir(CONTEXT_FOLDER):
            file_path = os.path.join(CONTEXT_FOLDER, filename)
            
            try:
                # Determine document type and load accordingly
                if filename.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                elif filename.endswith('.md'):
                    loader = TextLoader(file_path)
                else:
                    print(f"Skipping unsupported file type: {filename}")
                    continue
                
                # Load the document
                documents = loader.load()
                
                # Add metadata to each document
                for doc in documents:
                    doc.metadata.update({
                        'filename': filename,
                        'type': 'pdf' if filename.endswith('.pdf') else 'markdown',
                        # Assign confidence based on file type
                        'confidence': 0.9 if filename.endswith('.pdf') else 0.85,
                        # Extract category from filename (remove extension)
                        'category': os.path.splitext(filename)[0]
                    })
                
                all_documents.extend(documents)
                print(f"Loaded: {filename}")
                
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
                continue
        
        if not all_documents:
            raise Exception("No documents were successfully loaded")
            
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(all_documents)
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            chunks,
            embedding=self.embeddings
        )
        
        print(f"Loaded {len(all_documents)} documents, created {len(chunks)} chunks")

class ContextProcessor:
    """Processes and manages context retrieval and organization"""
    
    @staticmethod
    async def analyze_question(question: str) -> QuestionAnalysis:
        """First stage: Analyze the question to better understand what information we need."""
        analysis_prompt = f"""
        Analyze this question and break it down into key components:
        Question: {question}
        
        1. Identify main topics (at least 2-3 related terms)
        2. Specify what types of context would be most relevant
        3. Note any temporal aspects (time periods, dates, etc.)
        
        Format the response as JSON with keys: topics, context_types, temporal_context
        Ensure topics includes both specific and general terms related to the question.
        """
        
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": analysis_prompt}],
                stream=False
            )
            
            # Create base analysis object
            analysis = QuestionAnalysis(original_question=question)
            
            try:
                # Parse JSON response
                json_response = json.loads(response["message"]["content"])
                
                # Ensure we have topics, defaulting to question terms if none provided
                analysis.topics = json_response.get("topics", [])
                if not analysis.topics:
                    # Split question into words and use as topics
                    analysis.topics = [q.strip() for q in question.lower().split() if len(q.strip()) > 3]
                
                # Add the original question as a topic to ensure broad matching
                if question not in analysis.topics:
                    analysis.topics.append(question)
                
                # Don't enforce required context types to allow broader matching
                analysis.required_context_types = json_response.get("context_types", [])
                analysis.temporal_context = json_response.get("temporal_context")
                
                # Set a reasonable default confidence threshold
                analysis.confidence_threshold = 0.5  # Lowered from 0.7 to allow more matches
                
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                analysis.topics = [question]
                analysis.required_context_types = []
                analysis.temporal_context = None
                analysis.confidence_threshold = 0.5
                
            return analysis
            
        except Exception as e:
            print(f"Error in analyze_question: {str(e)}")
            analysis = QuestionAnalysis(original_question=question)
            analysis.topics = [question]
            analysis.confidence_threshold = 0.5
            return analysis
    
    @staticmethod
    async def get_relevant_context(
        analysis: QuestionAnalysis,
        vector_store: Chroma
    ) -> List[Dict]:
        """Retrieve and organize relevant context"""
        contexts = []
        
        for topic in analysis.topics:
            # Search with metadata filtering
            retrieved_docs = vector_store.similarity_search(
                topic,
                k=2,
                filter={
                    "category": {"$in": analysis.required_context_types},
                    "confidence": {"$gte": analysis.confidence_threshold}
                }
            )
            
            for doc in retrieved_docs:
                contexts.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
        
        return contexts

class ResponseGenerator:
    """Generates structured responses using context and analysis"""
    
    @staticmethod
    def format_context(contexts: List[Dict]) -> str:
        """Format contexts with metadata into a structured string"""
        formatted_contexts = []
        
        for ctx in contexts:
            formatted_contexts.append(
                f"""
                Source: {ctx['metadata']['category']}/{ctx['metadata'].get('subcategory', 'N/A')}
                Confidence: {ctx['metadata']['confidence']}
                ---
                {ctx['content']}
                """
            )
        
        return "\n\n=== Context Separator ===\n\n".join(formatted_contexts)
    
    @staticmethod
    def generate_prompt(
        question: str,
        context: str,
        analysis: QuestionAnalysis
    ) -> str:
        """Generate an enhanced prompt for the LLM"""
        return f"""
        You are an assistant specialized in Evan's work and project experience.
        
        Guidelines:
        1. Use ONLY the provided context for your response
        2. Maintain high accuracy and acknowledge information gaps
        3. Consider temporal context: {analysis.temporal_context or 'No specific time period'}
        4. Cite specific experiences and projects when relevant
        
        Context Information:
        {context}
        
        Question: {question}
        
        Response Requirements:
        1. Start with a direct answer to the question
        2. Reference specific examples from the context
        3. Maintain professional tone
        4. Clearly state if any part of the question cannot be answered with the given context
        """

# FastAPI Application
app = FastAPI()
vector_store_manager = VectorStoreManager()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    await vector_store_manager.initialize()
    yield
    # Cleanup code if needed

app.router.lifespan_context = lifespan

@app.get("/")
async def root():
    """Health check endpoint"""
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
        # Stage 1: Analyze question to extract topics and context requirements
        analysis = await ContextProcessor.analyze_question(request.question)
        
        # Stage 2: Retrieve relevant context with improved filtering
        retrieved_docs = []
        
        # First try with strict filtering
        for topic in analysis.topics:
            if analysis.required_context_types:
                docs = vector_store_manager.vector_store.similarity_search(
                    topic,
                    k=3,  # Increased from 2 to get more context
                    filter={"confidence": {"$gte": analysis.confidence_threshold}}
                    if analysis.confidence_threshold else None
                )
                retrieved_docs.extend(docs)
        
        # If no documents found, try without category filtering
        if not retrieved_docs:
            for topic in analysis.topics:
                docs = vector_store_manager.vector_store.similarity_search(
                    topic,
                    k=3,
                    filter=None  # Remove all filters to get any relevant context
                )
                retrieved_docs.extend(docs)
        
        # Deduplicate documents based on content
        seen_content = set()
        unique_docs = []
        for doc in retrieved_docs:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_docs.append(doc)
        
        # If still no context found, return appropriate message
        if not unique_docs:
            return {
                "response": "I apologize, but I don't have enough context to properly answer your question. Could you please rephrase or provide more details?",
                "debug_info": {
                    "analysis": vars(analysis),
                    "context_count": 0
                } if request.include_debug_info else None
            }

        # Stage 3: Format the retrieved context with metadata
        formatted_context = ResponseGenerator.format_context(
            [{"content": doc.page_content, "metadata": doc.metadata} for doc in unique_docs]
        )
        
        # Stage 4: Generate enhanced prompt
        prompt = ResponseGenerator.generate_prompt(request.question, formatted_context, analysis)
        
        # Stage 5: Generate answer
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        
        return {
            "response": response["message"]["content"],
            "debug_info": {
                "analysis": vars(analysis),
                "context_count": len(unique_docs),
                "topics": analysis.topics,
                "required_context_types": analysis.required_context_types
            } if request.include_debug_info else None
        }
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
