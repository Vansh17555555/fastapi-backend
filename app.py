from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import aiohttp
import os
import json
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# Allow CORS for trusted domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yoom-htm.vercel.app"],  # Add your frontend URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Validate API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found in environment variables.")

# Initialize components
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7, openai_api_key=OPENAI_API_KEY)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
vector_store = None

# Utility Functions
async def process_jsonl(url: str) -> str:
    """
    Function to process JSONL from a given URL and extract transcription text.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                transcription_text = ""
                async for line in response.content:
                    line = line.decode('utf-8')
                    if line.strip():
                        data = json.loads(line)
                        if data.get('type') == 'speech':
                            transcription_text += data.get('text', '') + " "
                return transcription_text
    except aiohttp.ClientError as e:
        logger.error(f"Error fetching transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching transcription: {str(e)}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON data received: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Invalid JSON data received: {str(e)}")

def update_vector_store(text: str):
    """
    Update the FAISS vector store with new text chunks.
    """
    global vector_store
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        if vector_store is None:
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            vector_store.save_local("faiss_index")  # Save the initial vector store
        else:
            vector_store.add_texts(chunks)
            vector_store.save_local("faiss_index")  # Update the saved vector store
        logger.info("Vector store updated successfully")
    except Exception as e:
        logger.error(f"Error updating vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating vector store: {str(e)}")

def load_vector_store():
    """
    Load the FAISS vector store from disk.
    """
    global vector_store
    try:
        if os.path.exists("faiss_index"):
            vector_store = FAISS.load_local("faiss_index", embeddings)
            logger.info("Vector store loaded successfully")
        else:
            vector_store = None
            logger.warning("No existing vector store found")
    except Exception as e:
        logger.error(f"Error loading vector store: {str(e)}")
        vector_store = None

# FastAPI Endpoints
@app.post("/process_transcription")
async def process_transcription(request: Request):
    """
    Endpoint to process transcription from a given URL.
    """
    try:
        body = await request.json()
        url = body.get('url')
        if not url:
            raise HTTPException(status_code=400, detail="No URL provided")
        
        transcription_text = await process_jsonl(url)
        update_vector_store(transcription_text)
        return {"message": "Transcription processed successfully", "transcription_text": transcription_text}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in process_transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/chat")
async def chat(request: Request):
    """
    Endpoint to interact with the conversational retrieval chain.
    """
    global vector_store
    try:
        body = await request.json()
        user_question = body.get('question', '')

        if vector_store is None:
            raise HTTPException(status_code=400, detail="No transcription data available. Please process a transcription first.")
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory
        )
        response = chain({"question": user_question})
        return {"reply": response["answer"]}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Load the vector store on startup
@app.on_event("startup")
async def startup_event():
    load_vector_store()

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
