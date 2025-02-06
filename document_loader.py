import os
from dataclasses import dataclass
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration Constants
CONTEXT_FOLDER = "docs"
CHUNK_SIZE = 256
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@dataclass
class DocumentMetadata:
    """Metadata for context documents"""
    category: str        # e.g., 'experience', 'technical', etc.
    subcategory: str     # e.g., 'projects', 'skills', etc.

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
        """Load and process all documents from the docs folder"""
        all_documents = []
        
        # Loop over files in the context folder
        for filename in os.listdir(CONTEXT_FOLDER):
            file_path = os.path.join(CONTEXT_FOLDER, filename)
            try:
                if filename.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                elif filename.endswith('.md'):
                    loader = TextLoader(file_path)
                else:
                    print(f"Skipping unsupported file type: {filename}")
                    continue
                
                documents = loader.load()
                for doc in documents:
                    doc.metadata.update({
                        'filename': filename,
                        'type': 'pdf' if filename.endswith('.pdf') else 'markdown',
                        'category': os.path.splitext(filename)[0]
                    })

                all_documents.extend(documents)
                print(f"Loaded: {filename}")

            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
                continue
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(all_documents)
        
        # Create vector store from chunks
        self.vector_store = Chroma.from_documents(chunks, embedding=self.embeddings)
        print(f"Loaded {len(all_documents)} documents, created {len(chunks)} chunks")
