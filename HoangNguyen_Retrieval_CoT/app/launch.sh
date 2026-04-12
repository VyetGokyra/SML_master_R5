#!/bin/bash
echo "Starting Wiki RAG + CoT Backend & Frontend..."
docker-compose up --build -d
echo "Done! Frontend is available at http://localhost:8501"
