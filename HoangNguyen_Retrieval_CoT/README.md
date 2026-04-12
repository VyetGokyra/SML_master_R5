# Wiki RAG + CoT Demo

This project provides a robust, containerized demonstration of Multi-Prompt Learning, combining Retrieval-Augmented Generation (RAG) and Chain-of-Thought (CoT) reasoning.

## Overview
The architecture features:
1. **Lightweight Frontend:** Displays detailed RAG retrieval processes, and visualizes the model's step-by-step thinking (CoT).
2. **Python Backend:** A fully dockerized backend API serving the LLM pipeline and vector search logic.
3. **Automated Testing & Benchmarking:** Scripts to aggressively test the pipeline against internal questions and recognized public benchmarks.

## Quick Start
To launch the demo with complete backend and frontend services:

./launch.sh

This script spins up the vector database, Python backend API, and frontend interface using Docker Compose.

## Benchmarking
You can evaluate the RAG+CoT implementation across various datasets via simple scripts:

./scripts/run_benchmarks.sh

This tests both internal multi-hop queries and official public QA benchmarks (e.g. HotpotQA), outputting detailed accuracy and latency metrics.

For extensive architectural details, refer to the files in the /internal_docs directory.
