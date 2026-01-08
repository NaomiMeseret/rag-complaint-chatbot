"""Interactive Gradio interface for RAG-powered complaint analysis chatbot."""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import gradio as gr
except ImportError:
    print("Error: Gradio not installed. Install with: pip install gradio")
    sys.exit(1)

from src.rag_complaint_analyzer.rag import RAGPipeline
from src.rag_complaint_analyzer.utils.logger import setup_logger

logger = setup_logger()


class ComplaintChatbot:
    """Interactive chatbot for complaint analysis."""
    
    def __init__(self, embeddings_path: str = "data/raw/complaint_embeddings.parquet"):
        """
        Initialize chatbot.
        
        Args:
            embeddings_path: Path to pre-built embeddings parquet file
        """
        logger.info("Initializing complaint analysis chatbot...")
        
        try:
            self.pipeline = RAGPipeline(
                embeddings_path=embeddings_path,
                retriever_top_k=5
            )
            logger.info("Chatbot initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize chatbot: {e}")
            raise
    
    def chat(self, question: str, history: list) -> list:
        """
        Process a user question and return response compatible with Gradio Chatbot.
        Returns a list of messages, each a dict with 'role' and 'content'.
        """
        if not question or not question.strip():
            if not history:
                history = []
            history = list(history)
            history.append({"role": "user", "content": ""})
            history.append({"role": "assistant", "content": "Please enter a question."})
            return history
        try:
            logger.info(f"Processing question: {question}")
            response = self.pipeline.query(question, return_sources=True)
            answer = response['answer']
            sources = response.get('sources', [])
            # Format answer plus sources info inline
            html_response = f"<b>Answer:</b><br>{answer}<br><br><b>Sources:</b>"
            for i, source in enumerate(sources, 1):
                metadata = source['metadata']
                similarity = source['similarity']
                document = source['document']
                doc_preview = document[:300] + "..." if len(document) > 300 else document
                html_response += f'<br><span style="font-size:13px;"><b>Source {i} (Similarity: {similarity:.3f})</b> | Product: {metadata.get('product', 'N/A')} | Issue: {metadata.get('issue', 'N/A')} | Company: {metadata.get('company', 'N/A')} | Date: {metadata.get('date_received', 'N/A')}<br><i>Excerpt:</i> {doc_preview}</span>'
            if not history:
                history = []
            history = list(history)
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": html_response})
            logger.info("Question processed successfully")
            return history
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if not history:
                history = []
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": f"<p style='color: red;'>{error_msg}</p>"})
            return history
    
    def clear_history(self):
        """Clear chat history for Gradio Chatbot markdown mode."""
        return []
    
    def create_interface(self):
        """Create Gradio interface."""
        # Create Gradio interface
        with gr.Blocks(
            title="CrediTrust Financial - Complaint Analysis Chatbot",
            theme=gr.themes.Soft()
        ) as demo:
            gr.Markdown(
                """
                # 🏦 CrediTrust Financial - Complaint Analysis Chatbot
                
                Welcome to the intelligent complaint analysis system. Ask questions about customer complaints 
                across our financial products and services.
                
                **How to use:**
                - Type your question in the text box below
                - Click "Ask" or press Enter to get an AI-powered answer
                - View the sources below each answer to see which complaint excerpts were used
                - Click "Clear" to start a new conversation
                
                **Example questions:**
                - "Why are people unhappy with Credit Cards?"
                - "What are the most common issues with Personal Loans?"
                - "What billing disputes have been reported recently?"
                """
            )
            
            chatbot = gr.Chatbot(
                label="Conversation",
                height=600,
                show_label=True
            )
            
            question_input = gr.Textbox(
                label="Ask a question about customer complaints",
                placeholder="e.g., What are customers complaining about regarding Credit Cards?",
                lines=2
            )
            
            with gr.Row():
                ask_button = gr.Button("Ask", variant="primary", size="lg")
                clear_button = gr.Button("Clear", variant="secondary", size="lg")
            
            # Event handlers
            question_input.submit(
                fn=self.chat,
                inputs=[question_input, chatbot],
                outputs=chatbot
            )
            ask_button.click(
                fn=self.chat,
                inputs=[question_input, chatbot],
                outputs=chatbot
            )
            clear_button.click(
                fn=self.clear_history,
                outputs=chatbot
            )
            
            gr.Markdown(
                """
                ---
                **Note:** This chatbot uses Retrieval-Augmented Generation (RAG) to provide answers 
                based on real customer complaint data. Sources are displayed for transparency and verification.
                """
            )
        
        return demo
    
    def launch(
        self,
        share: bool = False,
        server_name: str = "0.0.0.0",
        server_port: int = 7860
    ):
        """
        Launch Gradio interface.
        
        Args:
            share: Whether to create public link
            server_name: Server hostname
            server_port: Server port
        """
        logger.info("Launching Gradio interface...")
        
        demo = self.create_interface()
        demo.launch(share=share, server_name=server_name, server_port=server_port)


def main():
    """Main entry point for the application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch complaint analysis chatbot")
    parser.add_argument(
        "--embeddings",
        type=str,
        default="data/raw/complaint_embeddings.parquet",
        help="Path to pre-built embeddings parquet file"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public Gradio link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server hostname"
    )
    
    args = parser.parse_args()
    
    try:
        chatbot = ComplaintChatbot(embeddings_path=args.embeddings)
        chatbot.launch(share=args.share, server_name=args.host, server_port=args.port)
    except Exception as e:
        logger.error(f"Failed to launch chatbot: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

