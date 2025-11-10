OVERVIEW
-----------------------------------------------------------------------------------------------------------------------

This REST API powers a retail chatbot. It uses Google Gemini LLM, Pinecone vector search, FastAPI,
and LangChain to provide intelligent Q&A, product suggestions, and complementary recommendations to shoppers using
plain conversational language.

<img width="885" height="1063" alt="image" src="https://github.com/user-attachments/assets/c087b75c-5585-4350-9e51-5a5480ef93c7" />

-----------------------------------------------------------------------------------------------------------------------
CAPABILITIES
-----------------------------------------------------------------------------------------------------------------------
Product Recommendation:
Can recommend products from the store’s live assortment, matching queries about categories, needs, or specific requests.
Uses vector RAG search with Pinecone + LLM (Google Gemini) backend.

Product QA:
Answers detailed questions about specific products or use cases.

Complementary/Bundle Suggestions:
For any product, suggests 3 related or complementary items (similar, frequently bought together, or natural add-ons).

-----------------------------------------------------------------------------------------------------------------------
TECH STACK
-----------------------------------------------------------------------------------------------------------------------
Python 3 / FastAPI:
API built for speed and ease of cloud deployment.

LangChain:
RAG pipeline orchestration connecting Pinecone (vector search) and Gemini LLM.

Pinecone:
Scalable vector database for product/service embeddings.

Google Gemini (LLM):
High-quality natural language answers.

SlowAPI:
Simple per-IP rate limiting for production safety.

Frontend:
HTML/CSS/JS demo UI provided for real-time chat.

-----------------------------------------------------------------------------------------------------------------------
REQUIREMENTS
-----------------------------------------------------------------------------------------------------------------------
- Python 3.9+
- Dependencies from requirements.txt
- Pinecone.io account and index
- Google Gemini API credentials
- .env with PINECONE_API_KEY and GOOGLE_API_KEY

-----------------------------------------------------------------------------------------------------------------------
SETUP
-----------------------------------------------------------------------------------------------------------------------
1. Install dependencies:
   pip install -r requirements.txt
   
2. Set up your configuration:
   Enter your API keys in the .env file.

3. Run data_indexer.py:
   This program will generate text embeddings of a product catalog and index them in a Pinecone vector database.
   
4. Start the backend:
   uvicorn main:app --reload

5. Use index.html for the frontend.

-----------------------------------------------------------------------------------------------------------------------
TO BE ADDED
-----------------------------------------------------------------------------------------------------------------------
1. Multi-language support;

2. Integration with dynamic pricing/inventory APIs;

3. Deeper analytics:
    Track suggestion click-through, conversion, and product search trends;

4. Admin API for catalog re-indexing, prompt adjustment, etc;

5. More detailed prompt for edge cases and consistent answer standardization;

6. Better monitoring;

7. Scale via Gunicorn.

-----------------------------------------------------------------------------------------------------------------------
