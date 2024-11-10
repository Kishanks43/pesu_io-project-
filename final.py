#pip installs :
#!pip install -q langchain-core langchain-community langchain-text-splitters langchain-groq gradio soundfile assemblyai requests pyPDF2 pypdf qdrant-client


#Interrogation Model

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
import assemblyai as aai
import os
from typing import List
import numpy as np
import gradio as gr
import tempfile
import soundfile as sf
import requests
import time

class VoiceRAG:
    def __init__(self, api_key: str, assembly_api_key: str, max_tokens: int = 500):
        """Initialize the VoiceRAG system with necessary components."""
        self.assembly_api_key = assembly_api_key
        self.max_tokens = max_tokens  # Add max_tokens parameter

        # Initialize Groq LLM with max_tokens
        self.llm = ChatGroq(api_key=api_key, model_name="mixtral-8x7b-32768", max_tokens=self.max_tokens)

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Specify your model name

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        # Initialize vector store
        self.vector_store = None

        # Initialize memory
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Initialize prompt templates with max_tokens if necessary
        self.reflection_prompt = PromptTemplate(
            input_variables=["context", "response"],
            template="Given the context: {context}\nAnd the response: {response}\nGenerate a thoughtful follow-up question.",
            max_tokens=self.max_tokens
        )

        self.analysis_prompt = PromptTemplate(
            input_variables=["context", "response"],
            template="Given the context: {context}\nAnalyze the following response: {response}",
            max_tokens=self.max_tokens
        )

        # Create runnable sequences correctly
        self.reflection_chain = RunnableSequence(first=self.reflection_prompt, last=self.llm)
        self.analysis_chain = RunnableSequence(first=self.analysis_prompt, last=self.llm)

    def transcribe_audio(self, audio_input: tuple) -> str:
        """Transcribe audio input using AssemblyAI API."""
        try:
            # Extract audio data and sample rate from the tuple
            sample_rate, audio_data = audio_input

            # Ensure audio data is a NumPy array
            if not isinstance(audio_data, np.ndarray):
                raise ValueError("Expected audio data to be a NumPy array.")

            # Convert audio data from int16 to float32 (normalize to -1 to 1 range)
            audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max

            # Save audio to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                sf.write(temp_audio.name, audio_data, sample_rate)
                temp_audio_path = temp_audio.name

            # Step 1: Upload audio to AssemblyAI
            headers = {'authorization': self.assembly_api_key}
            with open(temp_audio_path, 'rb') as f:
                upload_response = requests.post(
                    'https://api.assemblyai.com/v2/upload',
                    headers=headers,
                    files={'file': f}
                )

            if upload_response.status_code != 200:
                raise Exception("Failed to upload audio to AssemblyAI")

            upload_url = upload_response.json()['upload_url']

            # Step 2: Request transcription
            transcription_response = requests.post(
                'https://api.assemblyai.com/v2/transcript',
                headers=headers,
                json={'audio_url': upload_url}
            )

            if transcription_response.status_code != 200:
                raise Exception("Failed to initiate transcription")

            transcript_id = transcription_response.json()['id']

            # Step 3: Poll for transcription completion
            while True:
                polling_response = requests.get(
                    f'https://api.assemblyai.com/v2/transcript/{transcript_id}',
                    headers=headers
                )

                if polling_response.status_code != 200:
                    raise Exception("Failed during transcription polling")

                polling_result = polling_response.json()

                if polling_result['status'] == 'completed':
                    # Cleanup: Delete the temp file after processing
                    os.remove(temp_audio_path)
                    return polling_result['text']

                elif polling_result['status'] == 'failed':
                    raise Exception("Transcription failed")

                time.sleep(2)  # Wait before polling again

        except Exception as e:
            raise Exception(f"Error transcribing audio: {str(e)}")

    # The rest of your class remains the same as before


    def process_pdf(self, pdf_bytes) -> str:
        """Process a PDF file and add its contents to the vector store."""
        try:
            # Create a temporary file and write the bytes content
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                temp_pdf.write(pdf_bytes)
                temp_pdf_path = temp_pdf.name

            # Load the PDF using PyPDFLoader
            loader = PyPDFLoader(temp_pdf_path)
            documents = loader.load()
            texts = self.text_splitter.split_documents(documents)

            # Initialize vector store if not already done
            if self.vector_store is None:
                self.vector_store = Qdrant.from_documents(documents=texts, embedding=self.embeddings, location=":memory:")
            else:
                self.vector_store.add_documents(texts)

            # Clean up temporary file
            os.unlink(temp_pdf_path)

            return f"Successfully processed PDF."

        except Exception as e:
            return f"Error processing PDF: {str(e)}"

    def get_relevant_context(self, query: str) -> List[str]:
        """Retrieve relevant context from the vector store"""
        if self.vector_store is None:
            return []

        return self.vector_store.similarity_search(query)

    def generate_reflection_question(self, response: str, context: str) -> str:
        """Generate a reflection question based on the response and context"""
        return self.reflection_chain.invoke({"response": response, "context": context})

    def analyze_response(self, response: str, context: str) -> str:
        """Analyze the response in context"""
        return self.analysis_chain.invoke({"response": response, "context": context})

    def update_context(self, new_information: str) -> None:
        """Update the conversation context with new information"""
        self.memory.save_context({"input": "User"}, {"output": new_information})

def create_interface(voicerag: VoiceRAG):
    def process_interaction(audio, text_input):
        response = None

        if audio is not None:
            try:
                response = voicerag.transcribe_audio(audio)
            except Exception as e:
                print(f"Error processing audio: {e}")
                return f"Error: {str(e)}", "Error processing audio", "Please try again"
        elif text_input:
            response = text_input
        else:
            return None, "No input provided", "Please provide either audio or text input"

        # Get relevant context
        context = voicerag.get_relevant_context(response)

        # Generate reflection question
        question = voicerag.generate_reflection_question(response, str(context))

        # Analyze response
        analysis = voicerag.analyze_response(response, str(context))

        # Update context with new information
        voicerag.update_context(response)

        return response, question, analysis

    with gr.Blocks() as interface:
      gr.Markdown("# VoiceRAG Interrogation System for Accused Individuals")

      with gr.Tab("Document Upload"):
          pdf_input = gr.File(
              label="Upload PDF Document",
              file_types=[".pdf"],
              type="binary"
          )
          pdf_upload_button = gr.Button("Process PDF")
          pdf_output = gr.Textbox(label="PDF Processing Result")

          pdf_upload_button.click(
              fn=lambda x: voicerag.process_pdf(x) if x else "No file uploaded",
              inputs=[pdf_input],
              outputs=[pdf_output]
          )

      with gr.Tab("Interaction"):
          with gr.Row():
              audio_input = gr.Audio(
                  sources=["microphone"],
                  type="numpy",
                  label="Audio Input"
              )
              text_input = gr.Textbox(
                  label="Text Input (if no audio)",
                  placeholder="Type your message here..."
              )

          with gr.Row():
              submit_btn = gr.Button("Process")

          with gr.Row():
              transcription_output = gr.Textbox(label="Transcribed Response")
              question_output = gr.Textbox(label="Generated Question")
              analysis_output = gr.Textbox(label="Response Analysis")

          submit_btn.click(
              process_interaction,
              inputs=[audio_input, text_input],
              outputs=[transcription_output, question_output, analysis_output]
          )

      return interface



if __name__ == "__main__":
    GROQ_API_KEY = "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"
    ASSEMBLY_API_KEY = "0ab2b85808b149d68af469e85bcc17bf"
    voicerag = VoiceRAG(api_key=GROQ_API_KEY, assembly_api_key=ASSEMBLY_API_KEY)
    interface = create_interface(voicerag)
    interface.launch()
