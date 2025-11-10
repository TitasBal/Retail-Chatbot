import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger
from langchain_google_genai import GoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_pinecone import Pinecone as LangChainPinecone
from langchain_huggingface import HuggingFaceEmbeddings
from slowapi import Limiter
from slowapi.util import get_remote_address



# This program implements a FastAPI web server hosting an AI-powered retail chatbot for Senukai.
# It integrates a Pinecone vector store for product embedding retrieval,
# and uses Google's Gemini LLM to generate concise responses based on semantic search results.



limiter = Limiter(key_func=get_remote_address)
qa_chain = None

def get_secret(key: str):
    return os.getenv(key)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global qa_chain
    logger.info("Server starting up...")

    load_dotenv()
    PINECONE_API_KEY = get_secret("PINECONE_API_KEY")
    GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
    INDEX_NAME = "senukai-chatbot-mvp"

    if not PINECONE_API_KEY or not GOOGLE_API_KEY:
        logger.error("API keys for Pinecone and Google must be set in the .env file.")
        raise ValueError("API keys for Pinecone and Google must be set in the .env file.")

    logger.info("Loading embedding model...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    logger.info("Connecting to Pinecone vector store...")
    vectorstore = LangChainPinecone.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings_model,
        text_key="combined_features"
    )

    logger.info("Initializing Google Gemini LLM...")
    llm = GoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )

    formatting_prompt = (
        "You are a helpful and concise retail assistant for Senukai.\n"
        "Your answers are short\n"
        "Give only the name and the price of the product with a very short description of the product (1 sentence)\n"
        "Customer Query: {query}\nYour reply:"
    )

    class CustomRetrievalQA(RetrievalQA):
        def invoke(self, input, **kwargs):
            input['query'] = formatting_prompt.format(query=input['query'])
            return super().invoke(input, **kwargs)

    logger.info("Creating RetrievalQA chain...")
    qa_chain = CustomRetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    logger.info("Application setup complete. Server is ready.")
    yield
    logger.info("Server shutting down...")

app = FastAPI(
    title="Senukai AI Retail Chatbot",
    description="An API for the chatbot",
    version="1.0.0",
    lifespan=lifespan
)
app.state.limiter = limiter

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    text: str




@app.post("/ask")
@limiter.limit("10/minute")
async def ask_chatbot(query: Query, request: Request):
    global qa_chain
    if not qa_chain:
        raise HTTPException(status_code=503, detail="AI components are not ready. Please try again in a moment.")

    if not query.text:
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    try:
        result = qa_chain.invoke({"query": query.text})
        answer = result.get("result", "Sorry, I could not find an answer.")
        source_docs = result.get("source_documents", [])
        sources = [
            {"id": doc.metadata.get('id'), "name": doc.metadata.get('name')}
            for doc in source_docs
        ]

        logger.info(f"Question answered for request: {request.client.host}")
        return {"answer": answer.strip(), "sources": sources}
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the query.")


def normalize(txt):
    return ''.join(e.lower() for e in txt if e.isalnum())

@app.post("/complementary")
async def get_complementary_items(query: Query):
    global qa_chain
    product_query = query.text.strip()
    try:
        results = qa_chain.retriever.vectorstore.similarity_search(product_query, k=10)
        complementary = []
        for doc in results:
            name = doc.metadata.get('name') or doc.page_content
            if normalize(name) != normalize(product_query) and name not in complementary:
                complementary.append(name)
            if len(complementary) == 3:
                break
        return {"complementary": complementary}
    except Exception as e:
        logger.error(f"VectorDB complementary query error: {e}")
        return {"complementary": []}


@app.get("/health")
async def health_check():
    global qa_chain
    if qa_chain:
        return {"status": "ok"}
    raise HTTPException(status_code=503, detail="QA component not ready.")


@app.get("/", summary="Root endpoint")
def root():
    return {"message": "Senukai AI Retail Chatbot is running!"}
