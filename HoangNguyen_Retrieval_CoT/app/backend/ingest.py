import wikipedia
import chromadb
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os

DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

ARTICLES_TO_SCRAPE = [
    "Nikola Tesla",
    "Thomas Edison",
    "War of the currents",
    "Alternating current",
    "Direct current",
    "George Westinghouse",
    "Electric power transmission",
    "Transformer",
    "Niagara Falls",
    "World's Columbian Exposition"
]

def ingest_data():
    print("Initializing embedding model...")
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=DB_DIR)
    
    # Try getting or creating collection
    collection = client.get_or_create_collection(name="wiki_rag_collection")
    
    # Check if empty, to avoid re-scraping repeatedly 
    if collection.count() > 0:
        print("Collection already contains data. Scrubbing and starting fresh for ingestion script...")
        client.delete_collection(name="wiki_rag_collection")
        collection = client.create_collection(name="wiki_rag_collection")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for article_title in ARTICLES_TO_SCRAPE:
        print(f"Fetching {article_title} from Wikipedia...")
        try:
            page = wikipedia.page(article_title, auto_suggest=False)
            content = page.content
            
            chunks = text_splitter.split_text(content)
            print(f"  -> Generated {len(chunks)} chunks.")
            
            # Embed and insert chunks directly into chroma
            embeddings = embeddings_model.embed_documents(chunks)
            ids = [f"{article_title.replace(' ', '_')}_{i}" for i in range(len(chunks))]
            metadatas = [{"source": article_title} for _ in chunks]
            
            collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            print(f"  -> Data inserted for {article_title}.")
        except Exception as e:
            print(f"Error fetching {article_title}: {e}")

    print(f"Ingestion complete. Total items in DB: {collection.count()}")

if __name__ == "__main__":
    ingest_data()
