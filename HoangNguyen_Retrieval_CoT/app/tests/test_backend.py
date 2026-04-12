import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from backend.main import app, augmentation_prompt, reasoning_prompt, synthesis_prompt, client as chroma_client, embeddings_model

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}

def test_unit_augmentation_prompt():
    test_context = "This is a fact."
    test_query = "What is this?"
    output = augmentation_prompt.format(context=test_context, query=test_query)
    assert "This is a fact." in output
    assert "What is this?" in output

def test_unit_reasoning_prompt():
    test_aug = "Augmented text."
    output = reasoning_prompt.format(augmented_query=test_aug)
    assert "Let's think step by step." in output
    assert "Augmented text." in output

def test_extraction_synthesis_prompt():
    test_trace = "The answer is 42."
    output = synthesis_prompt.format(reasoning_trace=test_trace)
    assert "The answer is 42." in output
    assert "Final Answer:" in output

def test_retrieval_logic():
    collection = chroma_client.get_collection(name="wiki_rag_collection")
    query_embedding = embeddings_model.embed_query("Who invented alternating current motor?")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )
    assert len(results["documents"][0]) > 0
    assert len(results["documents"][0][0]) > 10

def test_e2e_query():
    res = client.post("/query", json={"query": "Tell me about Tesla and Westinghouse."})
    assert res.status_code == 200
    data = res.json()
    assert "reasoning" in data
    assert "answer" in data
    assert "retrieved_contexts" in data
