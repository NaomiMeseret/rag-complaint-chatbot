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
    
    def chat(self, question: str, history: list) -> tuple:
        """
        Process a user question and return response.
        
        Args:
            question: User question
            history: Chat history (not used, but required by Gradio)
            
        Returns:
            Tuple of (answer_html, history)
        """
        if not question or not question.strip():
            return "Please enter a question.", history
        
        try:
            logger.info(f"Processing question: {question}")
            
            # Query pipeline
            response = self.pipeline.query(question, return_sources=True)
            
            # Format answer with sources
            answer = response['answer']
            sources = response.get('sources', [])
            
            # Create HTML response with sources
            html_response = f"""
            <div style="padding: 10px;">
                <h3 style="color: #2563eb; margin-bottom: 10px;">Answer:</h3>
                <p style="font-size: 16px; line-height: 1.6; margin-bottom: 20px;">{answer}</p>
                
                <h3 style="color: #2563eb; margin-bottom: 10px; margin-top: 30px;">Sources:</h3>
                <div style="background-color: #f3f4f6; padding: 15px; border-radius: 8px; margin-top: 10px;">
            """
            
            for i, source in enumerate(sources, 1):
                metadata = source['metadata']
                similarity = source['similarity']
                document = source['document']
                
                # Truncate document if too long
                doc_preview = document[:300] + "..." if len(document) > 300 else document
                
                html_response += f"""
                    <div style="margin-bottom: 20px; padding: 10px; background-color: white; border-left: 4px solid #2563eb; border-radius: 4px;">
                        <h4 style="color: #1f2937; margin-bottom: 8px;">Source {i} (Similarity: {similarity:.3f})</h4>
                        <p style="margin: 5px 0;"><strong>Product:</strong> {metadata.get('product', 'N/A')}</p>
                        <p style="margin: 5px 0;"><strong>Issue:</strong> {metadata.get('issue', 'N/A')}</p>
                        <p style="margin: 5px 0;"><strong>Company:</strong> {metadata.get('company', 'N/A')}</p>
                        <p style="margin: 5px 0;"><strong>Date:</strong> {metadata.get('date_received', 'N/A')}</p>
                        <p style="margin-top: 10px; padding: 10px; background-color: #f9fafb; border-radius: 4px; font-size: 14px;">
                            <strong>Excerpt:</strong> {doc_preview}
                        </p>
                    </div>
                """
            
            html_response += """
                </div>
            </div>
            """
            
            # Update history
            history.append((question, html_response))
            
            logger.info("Question processed successfully")
            return "", history
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            logger.error(error_msg, exc_info=True)
            history.append((question, f"<p style='color: red;'>{error_msg}</p>"))
            return "", history
    
    def clear_history(self) -> tuple:
        """Clear chat history."""
        return [], ""
    
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
                show_label=True,
                avatar_images=(None, "🤖"),
                bubble_full_width=False
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
                outputs=[question_input, chatbot]
            )
            
            ask_button.click(
                fn=self.chat,
                inputs=[question_input, chatbot],
                outputs=[question_input, chatbot]
            )
            
            clear_button.click(
                fn=self.clear_history,
                outputs=[chatbot, question_input]
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

