import os
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

@dataclass
class ResumeMetadata:
    """Enhanced metadata for resume content"""
    content_type: str
    title: str
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    skills: List[str] = None
    technologies: List[str] = None
    keywords: List[str] = None

class ResumeDocumentProcessor:
    """Processes resume-related documents with intelligent chunking"""
    
    def __init__(self):
        self.vector_store = None
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Headers to use for markdown splitting
        self.headers_to_split_on = [
            ("##", "section"),  # Major sections
            ("###", "subsection"),  # Subsections
        ]
        
    def extract_metadata_from_frontmatter(self, content: str) -> tuple[str, ResumeMetadata]:
        """Extract metadata from document frontmatter"""
        lines = content.split('\n')
        metadata_lines = []
        content_lines = []
        in_metadata = False
        
        for line in lines:
            if line.strip() == '---':
                in_metadata = not in_metadata
                continue
            if in_metadata:
                metadata_lines.append(line)
            else:
                content_lines.append(line)
        
        metadata = {}
        for line in metadata_lines:
            if ':' in line:
                key, value = line.split(':', 1)
                if '[' in value:  # Handle lists
                    value = [v.strip(' []"\'') for v in value.split(',')]
                if isinstance(value, list):  # Ensure lists are stored as is
                    metadata[key.strip()] = value
                else:
                    metadata[key.strip()] = value.strip()
        
        return '\n'.join(content_lines), ResumeMetadata(
            content_type=metadata.get('type', 'skill'),
            title=metadata.get('title', ''),
            date_start=metadata.get('date_start'),
            date_end=metadata.get('date_end'),
            skills=metadata.get('skills', []),
            technologies=metadata.get('technologies', []),
            keywords=metadata.get('keywords', [])
        )

    def process_document(self, file_path: str) -> List[Document]:
        """Process a single document while ensuring section titles remain strongly tied to content."""
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract metadata from frontmatter
        content, metadata = self.extract_metadata_from_frontmatter(content)
        
        # Identify section titles manually (lines that start with "## " or "### ")
        lines = content.split("\n")
        current_title = ""
        processed_lines = []

        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith("##"):  # Detect major section headers
                current_title = stripped_line  # Store the section title
            elif stripped_line.startswith("###"):  # Detect subsection headers
                current_title = stripped_line  # Update to the most recent subsection

            # Append the title to every line to preserve context
            processed_lines.append(f"{current_title}\n{line}" if current_title else line)

        processed_content = "\n".join(processed_lines)  # Reassemble the content

        # Chunk using RecursiveCharacterTextSplitter (no markdown splitting)
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=256,  # Increased overlap to ensure smooth transitions
            separators=["\n\n", "\n", ". ", " ", ""],  
            keep_separator=True
        )

        final_chunks = []
        chunks = recursive_splitter.split_text(processed_content)  # Chunk the modified text

        for chunk in chunks:
            final_chunks.append(Document(
                page_content=chunk,
                metadata={
                    **{
                        key: ", ".join(value) if isinstance(value, list) else (value if value is not None else "")
                        for key, value in metadata.__dict__.items()
                    },
                    'source': file_path
                }
            ))

        return final_chunks


    async def initialize_vector_store(self, docs_folder: str):
        """Initialize the vector store with processed documents"""
        all_chunks = []
        
        for root, _, files in os.walk(docs_folder):
            for filename in files:
                if filename.endswith('.md'):
                    file_path = os.path.join(root, filename)
                    chunks = self.process_document(file_path)
                    all_chunks.extend(chunks)

        
        self.vector_store = Chroma.from_documents(
            documents=all_chunks,
            embedding=self.embeddings
        )
        
        return len(all_chunks)