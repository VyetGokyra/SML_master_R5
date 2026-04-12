import streamlit as st
import requests
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Wiki RAG + CoT Demo", layout="wide")
st.title("Wiki RAG + CoT Demo")

query = st.text_input("Enter your complex factual query:")

if st.button("Generate Answer"):
    with st.spinner("Retrieving and Thinking..."):
        try:
            res = requests.post(f"{BACKEND_URL}/query", json={"query": query})
            res.raise_for_status()
            data = res.json()
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("RAG Retrieval Nodes")
                st.write(data.get("retrieved_contexts", []))
            
            with col2:
                st.subheader("Chain-of-Thought (Reasoning)")
                st.info(data.get("reasoning", ""))
            
            st.success(f"Final Answer: {data.get('answer', '')}")
            
        except Exception as e:
            st.error(f"Error communicating with backend: {e}")
