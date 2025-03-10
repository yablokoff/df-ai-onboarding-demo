import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
import os
from typing import Dict, Optional
import json
import time
from pathlib import Path
from pydantic import BaseModel
from typing import List
import tempfile

# Configure API keys
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Initialize Gemini
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

class MaterialSpec(BaseModel):
    name: str
    specifications: Optional[List[str]]
    post_production: Optional[List[str]]

class DocumentAnalysis(BaseModel):
    text_summary: str
    materials: List[MaterialSpec]
    post_production: List[str]  # General post-production options not tied to specific materials
    machines: List[str]

class DigiFabsterAgent:
    def __init__(self):
        self.generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
            "response_schema": DocumentAnalysis
        }
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-pro-exp-02-05",
            generation_config=self.generation_config
        )
        
        # Initialize FAISS vector store
        self.embeddings = OpenAIEmbeddings()
        if os.path.exists("faiss_index"):
            self.vector_store = FAISS.load_local("faiss_index", self.embeddings, allow_dangerous_deserialization=True)
        else:
            self.vector_store = FAISS.from_texts(
                ["Initial empty index"], 
                self.embeddings
            )
        
    def process_pdf(self, file) -> Dict:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name

        try:
            # Upload file to Gemini
            gemini_file = genai.upload_file(tmp_path, mime_type="application/pdf")
            
            # Wait for file processing
            while True:
                file_status = genai.get_file(gemini_file.name)
                if file_status.state.name == "ACTIVE":
                    break
                elif file_status.state.name != "PROCESSING":
                    raise Exception(f"File processing failed: {file_status.state.name}")
                time.sleep(10)

            # Start chat session with the model
            chat = self.model.start_chat(history=[])
            print(gemini_file)
            
            prompt = """
            Analyze this document and give a summary of what's there.

            1. Text summary
                - Pricing information (including quantities and prices)
                - Material specifications
                - Manufacturing specifications (tolerances, finishes, etc.)
                - Any mentioned machines or manufacturing processes

            2. List of materials
                - Name
                - Specifications
                - Post-production processes that can be applied to this material
            3. List of post-production processes
                - General post-production processes that can be applied to any material
            4. List of machines
                - Machines that can be used to manufacture with the materials
            
            Provide a clear, structured summary of each category.
            """

            response = chat.send_message([gemini_file, prompt])
            
            # Parse the response into DocumentAnalysis
            try:
                # First try to parse as JSON directly
                analysis_data = response.candidates[0].content.parts[0].function_response.outputs
                analysis = DocumentAnalysis(**analysis_data)
            except (AttributeError, KeyError, TypeError):
                # If direct JSON parsing fails, try to extract from text
                try:
                    # Try to parse the text as JSON
                    analysis_data = json.loads(response.text)
                    analysis = DocumentAnalysis(**analysis_data)
                except json.JSONDecodeError:
                    # Fallback: Create a basic structure from the text
                    analysis = DocumentAnalysis(
                        text_summary=response.text,
                        materials=[],
                        post_production=[],
                        machines=[]
                    )
            
            # Store in vector database
            self.vector_store.add_texts(
                texts=[response.text],
                metadatas=[{"source": file.name, "type": "pricing_specs"}]
            )
            self.vector_store.save_local("faiss_index")  # Save after adding new data
            
            return {
                "raw_text": response.text,
                "structured_data": analysis.model_dump()
            }

        finally:
            # Clean up temporary file
            Path(tmp_path).unlink()
    
    def query_knowledge(self, query: str) -> str:
        results = self.vector_store.similarity_search(query, k=3)
        return self._format_results(results)
    
    def _format_results(self, results) -> str:
        # Format vector store results into readable text
        response = "Here's what I found:\n\n"
        for doc in results:
            response += f"From document: {doc.metadata['source']}\n"
            response += doc.page_content + "\n\n"
        return response

def main():
    st.set_page_config(page_title="DigiFabster Data Collection", layout="wide")
    
    if "agent" not in st.session_state:
        st.session_state.agent = DigiFabsterAgent()

    st.title("DigiFabster AI Onboarding")

    # Introduction section
    with st.expander("🎯 Our Goal", expanded=True):
        st.markdown("""
        ### Welcome to DigiFabster AI Onboarding [Phase 1 / 3]!
        
        Upload your PDF documents containing:
        - Pricing information
        - Material specifications
        - Manufacturing requirements
        - Machine capabilities
        
        I'll analyze them and build a knowledge base that you can query.
        """)

    # File upload section
    st.subheader("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Drop your PDF files here",
        accept_multiple_files=True,
        type=['pdf']
    )

    if uploaded_files:
        for file in uploaded_files:
            with st.spinner(f'Processing {file.name}...'):
                try:
                    result = st.session_state.agent.process_pdf(file)
                    st.success(f"Successfully processed {file.name}")
                    with st.expander("View extracted data"):
                        # Display structured data in a more readable format
                        st.subheader("Summary")
                        st.write(result["structured_data"]["text_summary"])

                        st.subheader("Raw JSON")
                        st.write(result)
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")

    # Query section
    st.subheader("Query Your Data")
    query = st.text_input("Ask about your processed documents:")
    if query:
        with st.spinner('Searching...'):
            results = st.session_state.agent.query_knowledge(query)
            st.write(results)

if __name__ == "__main__":
    main() 