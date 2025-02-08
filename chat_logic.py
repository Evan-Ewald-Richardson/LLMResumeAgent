from typing import List, Dict
from dataclasses import dataclass
from pydantic import BaseModel
import ollama

MODEL_NAME = "mistral:latest"

@dataclass
class QuestionAnalysis:
    """Structured analysis of resume-related questions"""
    original_question: str
    focus_area: str  # 'projects', 'work_experience', 'education', 'skills'
    skills_mentioned: List[str] = None

class ChatRequest(BaseModel):
    question: str

class ContextProcessor:
    """Processes resume-related questions to determine focus area and extract relevant skills."""
    
    @staticmethod
    async def analyze_question(question: str) -> QuestionAnalysis:
        analysis_prompt = f"""
        Analyze this resume-related question and determine its focus:
        Question: {question}
        
        Output your answer in this format:
        Focus Area: [One of: projects, work_experience, education, skills]
        Skills Mentioned: [Any technical skills, languages, or tools mentioned in the question - if none are mentioned leave blank]
        """
        
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": analysis_prompt}],
                stream=False
            )

            response_text = response["message"]["content"]
            
            # Parse the response
            focus_area = "general"
            skills = []
            
            for line in response_text.splitlines():
                line = line.strip()
                if line.startswith("Focus Area:"):
                    focus_area = line[len("Focus Area:"):].strip().lower()
                elif line.startswith("Skills Mentioned:"):
                    skills = [skill.strip() for skill in line[len("Skills Mentioned:"):].split(",") if skill.strip()]
            
            return QuestionAnalysis(
                original_question=question,
                focus_area=focus_area,
                skills_mentioned=skills
            )
            
        except Exception as e:
            print(f"Error in analyze_question: {str(e)}")
            return QuestionAnalysis(
                original_question=question,
                focus_area="general"
            )
    
    @staticmethod
    async def get_relevant_context(
        analysis: QuestionAnalysis,
        vector_store
    ) -> List[Dict]:
        """Retrieve relevant experience and skills from the vector store."""
        contexts = []
        
        # First, search based on focus area
        focus_query = analysis.original_question
        if analysis.focus_area != "general":
            focus_query = f"{analysis.focus_area} {analysis.original_question}"
        
        retrieved_docs = vector_store.similarity_search(
            focus_query,
            k=2
        )
        contexts.extend([{
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown")
        } for doc in retrieved_docs])
        
        # If skills were mentioned, do an additional search
        if analysis.skills_mentioned:
            for skill in analysis.skills_mentioned:
                skill_docs = vector_store.similarity_search(
                    f"{skill} experience examples",
                    k=1
                )
                contexts.extend([{
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown")
                } for doc in skill_docs])
        
        return contexts

class ResponseGenerator:
    """Generates focused responses about resume experience."""
    
    @staticmethod
    def format_context(contexts: List[Dict]) -> str:
        return "\n\n---\n\n".join([
            ctx['content']
            for ctx in contexts
        ])
    
    @staticmethod
    def generate_prompt(
        question: str,
        context: str,
        analysis: QuestionAnalysis
    ) -> str:
        return f"""
        You are an assistant helping to answer questions about Evan's professional experience.
        
        Context Information:
        {context}
        
        Question: {question}
        Focus Area: {analysis.focus_area}
        {f'Relevant Skills: {", ".join(analysis.skills_mentioned)}' if analysis.skills_mentioned else ''}
        
        Please provide a clear, concise, and focused response that:
        1. Directly answers the question using only the provided context
        2. Highlights specific examples and experiences
        3. If technical skills are mentioned, includes concrete examples of their use
        4. Does not reference gaps in information or context
        5. Speaks in a 3rd person voice: Evan worked...
        6. Speaks in absolutes, no likely or might, etc.

        If the question is unrelated to Evan's projects, work experience, education, or skills,
        answer it directly without referencing the context. Avoid providing unnecessary or unrelated information.
        """