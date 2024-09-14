from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7, openai_api_key=OPENAI_API_KEY)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
vector_store = None


def process_jsonl(url: str) -> str:
    """
    Function to process JSONL from a given URL and extract transcription text.
    """
    response = requests.get(url)
    transcription_text = ""
    for line in response.text.split('\n'):
        if line.strip():
            data = json.loads(line)
            if data.get('type') == 'speech':
                transcription_text += data.get('text', '') + " "
    return transcription_text


def update_vector_store(text: str):
    """
    Update the FAISS vector store with new text chunks.
    """
    global vector_store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    if vector_store is None:
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    else:
        vector_store.add_texts(chunks)


@app.post("/process_transcription")
async def process_transcription(request: Request):
    """
    Endpoint to process transcription from a given URL.
    """
    body = await request.json()
    url = body.get('url')
    if not url:
        raise HTTPException(status_code=400, detail="No URL provided")
    
    transcription_text = process_jsonl(url)
    update_vector_store(transcription_text)
    return {"message": "Transcription processed successfully", "transcription_text": transcription_text}


@app.post("/chat")
async def chat(request: Request):
    """
    Endpoint to interact with the conversational retrieval chain.
    """
    global vector_store
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
