# Implementation Checklist: Wiki RAG + CoT Demo

## Phase A: Environment & Dockerization
- [x] Create `Dockerfile.backend` for Python/FastAPI environment.
- [x] Create `Dockerfile.frontend` for lightweight UI (Streamlit/Gradio).
- [x] Configure `docker-compose.yml` orchestrating Frontend, Backend, and Vector DB.
- [x] Write `launch.sh` wrapper script for fast, one-click start.
- [x] Scaffold FastAPI backend routes (e.g., `/query`, `/health`).
- [x] Scaffold Frontend layout with distinct views: Input, Retrieval Nodes, and Streaming Thoughts.

## Phase B: Retrieval Engine & DB
- [x] Initialize vector database container (e.g., ChromaDB or Qdrant).
- [x] Write Wikipedia ingestion script (scrape 10-15 dense, interconnected articles).
- [x] Implement text chunking using `RecursiveCharacterTextSplitter`.
- [x] Embed and index chunks using a local HuggingFace embedding engine.
- [x] Validate Backend search mechanism correctly returns Top-K documents based on similarity.

## Phase C: Multi-Prompt Pipeline Integration
- [x] Code **Augmentation Prompt**: Inject retrieved arrays into the `[Context]` block.
- [x] Code **Reasoning Prompt (CoT)**: Enforce step-by-step logic generation.
- [x] Code **Synthesis Prompt**: Extract final answer out of unstructured reasoning traces.
- [x] Integrate pipeline into FastAPI so the LLM sequence executes smoothly.
- [ ] Set up frontend bridging (e.g., SSE or WebSockets) to stream CoT traces live to UI.

## Phase D: Comprehensive Testing
- [x] **Unit Tests:** Verify prompt templates reliably absorb variables without failing.
- [x] **Retrieval Tests:** Assert that known factual keywords return the exact targeted chunks.
- [x] **Extraction Tests:** Mock LLM outputs to verify Synthesis step formats correctly into JSON.
- [x] **E2E Tests:** Execute full PyTest flows mapping User Input $
ightarrow$ Output.

## Phase E: Benchmarking & Official Validation
- [ ] Build internal benchmark array (25 multi-hop factual queries based on scraped articles).
- [ ] Download & integrate a target sample of a public benchmark (e.g., HotpotQA).
- [ ] Write `scripts/run_benchmarks.sh` logic.
- [ ] Automate pipeline to loop datasets and compare Naive vs RAG vs RAG+CoT.
- [ ] Export Accuracy and inference Latency metrics to a compiled CSV/Markdown report.
