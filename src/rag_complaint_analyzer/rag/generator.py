"""Generator implementation for RAG pipeline using LLMs."""

import logging
from typing import Optional, List, Dict, Any
import re

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..utils.exceptions import RAGError

logger = logging.getLogger(__name__)


class Generator:
    """Generates answers using LLMs with retrieved context."""
    
    DEFAULT_PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust Financial. Your task is to answer questions about customer complaints based on the provided context from real complaint data.

Use the following retrieved complaint excerpts to formulate your answer. Only use information from the provided context. If the context doesn't contain enough information to answer the question, state that you don't have enough information rather than making assumptions.

Context:
{context}

Question: {question}

Answer:"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        prompt_template: Optional[str] = None,
        max_length: int = 512,
        temperature: float = 0.7
    ):
        """
        Initialize generator.
        
        Args:
            model_name: Hugging Face model name (defaults to small text generation model)
            prompt_template: Custom prompt template
            max_length: Maximum length of generated text
            temperature: Sampling temperature
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RAGError(
                "transformers library not installed. "
                "Install with: pip install transformers torch"
            )
        
        # Use a small, fast model by default (good for demo purposes)
        if model_name is None:
            model_name = "gpt2"  # Lightweight model, can be replaced with larger models
        
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        
        logger.info(f"Loading generator model: {model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load the language model."""
        try:
            # Use text-generation pipeline for simplicity
            self.generator_pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
                max_length=self.max_length,
                temperature=self.temperature,
                device=-1  # CPU, change to 0 for GPU if available
            )
            logger.info(f"Generator model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load {self.model_name}, using fallback: {e}")
            # Fallback to a template-based response
            self.generator_pipeline = None
    
    def generate(
        self,
        question: str,
        context: str,
        max_length: Optional[int] = None
    ) -> str:
        """
        Generate answer from question and context.
        
        Args:
            question: User question
            context: Retrieved context chunks
            max_length: Override default max length
            
        Returns:
            Generated answer
        """
        if max_length is None:
            max_length = self.max_length
        
        # Format prompt
        prompt = self.prompt_template.format(context=context, question=question)
        
        try:
            if self.generator_pipeline is not None:
                # Use LLM if available
                result = self.generator_pipeline(
                    prompt,
                    max_length=max_length,
                    num_return_sequences=1,
                    truncation=True,
                    do_sample=True
                )
                
                generated_text = result[0]['generated_text']
                # Extract just the answer part (after "Answer:")
                if "Answer:" in generated_text:
                    answer = generated_text.split("Answer:")[-1].strip()
                else:
                    answer = generated_text.replace(prompt, "").strip()
                
                return answer
            else:
                # Fallback: template-based response
                return self._template_response(question, context)
                
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            # Fallback to template-based response
            return self._template_response(question, context)
    
    def _template_response(self, question: str, context: str) -> str:
        """
        Fallback template-based response when LLM is not available.
        
        Args:
            question: User question
            context: Retrieved context chunks
            
        Returns:
            Template-based answer
        """
        # Extract key information from context
        if not context.strip():
            return "I don't have enough information to answer this question based on the available complaint data."
        
        # Simple extraction: look for common patterns
        answer_parts = []
        
        # Count products mentioned
        products = []
        if "Credit card" in context or "credit card" in context.lower():
            products.append("Credit Cards")
        if "Personal loan" in context or "personal loan" in context.lower():
            products.append("Personal Loans")
        if "Savings account" in context or "savings account" in context.lower():
            products.append("Savings Accounts")
        if "Money transfer" in context or "money transfer" in context.lower():
            products.append("Money Transfers")
        
        if products:
            answer_parts.append(f"Based on the complaint data, issues have been reported for: {', '.join(products)}.")
        
        # Extract key issues
        issue_keywords = {
            "billing": "billing disputes",
            "fraud": "fraud or unauthorized transactions",
            "communication": "communication issues",
            "service": "service problems"
        }
        
        found_issues = []
        for keyword, issue_label in issue_keywords.items():
            if keyword in context.lower():
                found_issues.append(issue_label)
        
        if found_issues:
            answer_parts.append(f"Common issues include: {', '.join(found_issues)}.")
        
        # Provide a summary
        if answer_parts:
            return " ".join(answer_parts) + " " + self._summarize_context(context)
        else:
            return self._summarize_context(context)
    
    def _summarize_context(self, context: str, max_chars: int = 300) -> str:
        """Extract a summary from context."""
        # Take first chunk and truncate
        chunks = context.split("[Chunk")
        if chunks:
            first_chunk = chunks[1] if len(chunks) > 1 else chunks[0]
            summary = first_chunk.split("\n\n")[0] if "\n\n" in first_chunk else first_chunk
            if len(summary) > max_chars:
                summary = summary[:max_chars] + "..."
            return f"Key details: {summary}"
        return "I don't have enough information to provide a detailed answer."

