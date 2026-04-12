from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
import os

app = FastAPI(title="Wiki RAG + CoT Backend")

DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# Only build embeddings if directory exists to avoid crashing tests if ingestion hasn't run
# But actually, downloading the small model takes 1s
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=DB_DIR)

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

# 1. Augmentation / Context Prompt
AUGMENTATION_TEMPLATE = """
Information from internal knowledge base:
{context}

Based only on the information retrieved above, please answer the user's query:
Query: {query}
"""
augmentation_prompt = PromptTemplate.from_template(AUGMENTATION_TEMPLATE)

# 2. Reasoning Prompt (Chain-of-Thought)
REASONING_TEMPLATE = """
Please think step-by-step about the following augmented prompt to deduce the correct answer.

{augmented_query}

Your thinking process:
Let's think step by step.
"""
reasoning_prompt = PromptTemplate.from_template(REASONING_TEMPLATE)

# 3. Synthesis Prompt
SYNTHESIS_TEMPLATE = """
You are a parser. Given the following unstructured reasoning trace, extract the final concise answer.

Reasoning Trace:
{reasoning_trace}

Final Answer:
"""
synthesis_prompt = PromptTemplate.from_template(SYNTHESIS_TEMPLATE)

@app.post("/query")
def process_query(req: QueryRequest):
    # Phase B: Real Retrieval Engine execution
    try:
        collection = client.get_collection(name="wiki_rag_collection")
        query_embedding = embeddings_model.embed_query(req.query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        retrieved_contexts = results['documents'][0] if results['documents'] else []
    except Exception as e:
        # Fallback if DB doesn't exist yet
        retrieved_contexts = ["Wikipedia DB not yet populated. Please execute backend/ingest.py to populate."]
        
    formatted_context = "\\n".join([f"- {c}" for c in retrieved_contexts])
    
    # 1. Augmentation execution
    augmented_text = augmentation_prompt.format(context=formatted_context, query=req.query)
    
    # 2. Reasoning (CoT) execution
    reasoning_input = reasoning_prompt.format(augmented_query=augmented_text)
    
    # Mock LLM generation simulating CoT inference based on actual search text
    trace_intro = "First, looking at the contexts: "
    trace_body = " ... ".join([c[:50] + "..." for c in retrieved_contexts])
    mock_llm_reason_trace = trace_intro + trace_body + " Therefore, we analyze this sequentially and deduce the answer."
    
    # 3. Synthesis execution
    synthesis_input = synthesis_prompt.format(reasoning_trace=mock_llm_reason_trace)
    
    # Mock LLM synthesis generation
    mock_llm_final_answer = "Synthesized Answer derived from real RAG context."

    return {
        "query": req.query,
        "retrieved_contexts": retrieved_contexts,
        "reasoning": mock_llm_reason_trace,
        "answer": mock_llm_final_answer
    }
