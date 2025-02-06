from typing import List, Dict, Optional
from dataclasses import dataclass
from pydantic import BaseModel
import ollama

MODEL_NAME = "mistral:latest"

@dataclass
class QuestionAnalysis:
    """Structured analysis of user questions"""
    original_question: str
    topics: List[str] = None
    required_context_types: List[str] = None

class ChatRequest(BaseModel):
    question: str

class ContextProcessor:
    """Processes the question to extract main topics and relevant context types."""
    
    @staticmethod
    async def analyze_question(question: str) -> QuestionAnalysis:
        analysis_prompt = f"""
        Analyze this question and break it down into key components:
        Question: {question}
        
        1. Identify the main topics (at least 2-3 related terms).
        2. Specify what types of context would be most relevant.
        
        Please output your answer in plain text using the following format:
        
        Topics: topic1, topic2, topic3.
        Context Types: type1, type2.
        """
        
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{"content": analysis_prompt}],
                stream=False
            )
            response_text = response["message"]["content"]
            
            # Parse the plain text output to extract topics and context types.
            topics = []
            context_types = []
            for line in response_text.splitlines():
                line = line.strip()
                if line.lower().startswith("topics:"):
                    topics = [item.strip() for item in line[len("Topics:"):].split(",") if item.strip()]
                elif line.lower().startswith("context types:"):
                    context_types = [item.strip() for item in line[len("Context Types:"):].split(",") if item.strip()]
            
            # Fallback: if no topics were parsed, split the original question into words.
            if not topics:
                topics = [word.strip() for word in question.lower().split() if len(word.strip()) > 3]
            # Ensure the original question is included as a topic.
            if question not in topics:
                topics.append(question)
            
            analysis = QuestionAnalysis(
                original_question=question,
                topics=topics,
                required_context_types=context_types
            )
            return analysis
            
        except Exception as e:
            print(f"Error in analyze_question: {str(e)}")
            return QuestionAnalysis(
                original_question=question,
                topics=[question]
            )
    
    @staticmethod
    async def get_relevant_context(
        analysis: QuestionAnalysis,
        vector_store
    ) -> List[Dict]:
        """Retrieve and organize relevant context from the vector store."""
        contexts = []
        for topic in analysis.topics:
            # Only filter by category if context types are provided.
            filter_criteria = {"category": {"$in": analysis.required_context_types}} if analysis.required_context_types else None
            retrieved_docs = vector_store.similarity_search(
                topic,
                k=2,
                filter=filter_criteria
            )
            for doc in retrieved_docs:
                contexts.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
        return contexts


class ResponseGenerator:
    """Generates and formats responses based on the analysis and retrieved context."""
    
    @staticmethod
    def format_context(contexts: List[Dict]) -> str:
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
