# Building a Local RAG System with Go and Ollama

A simple Retrieval-Augmented Generation (RAG) implementation using Go and Ollama. This project explores how semantic retrieval can provide relevant context to a local LLM, how retrieval quality can be evaluated, and where context-grounded generation has limitations.

## Overview

Large Language Models (LLMs) generate answers based on knowledge learned during training. However, they may not have access to information stored in specific documents or custom knowledge bases.

Retrieval-Augmented Generation (RAG) addresses this limitation by retrieving relevant information from an external knowledge source and providing it to the LLM as context before generating an answer.

This project implements the RAG pipeline locally using Go and Ollama, from document retrieval to context-grounded generation.

## Architecture

The system follows this workflow:

1. **Knowledge Base**: Store knowledge in a text file and split it into topic-based chunks.
2. **Embedding**: Convert each chunk into a numerical vector using `nomic-embed-text`.
3. **Retrieval**: Retrieve relevant chunks using keyword matching or semantic similarity.
4. **Context Construction**: Combine the retrieved chunks into a context for the LLM.
5. **Generation**: Use `qwen2.5:3b` to generate an answer based on the retrieved context.
6. **Evaluation**: Measure retrieval quality using Precision@K, Recall@K, and out-of-scope rejection.

## Requirements

- Go
- [Ollama](https://ollama.com/)
- `qwen2.5:3b` for text generation
- `nomic-embed-text` for embeddings

## Getting Started

### 1. Pull the Models

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

Make sure the Ollama service is running before executing the application.

### 2. Clone the Repository

```bash
git clone <repository-url>
cd rag
```

Replace `<repository-url>` with your repository URL.

### 3. Install Dependencies

```bash
go mod download
```

### 4. Run the Application

```bash
go run .
```

Select a mode from the available options to explore different parts of the implementation.

## Project Structure

```text
rag/
├── data/
│   └── knowledge.txt
├── go.mod
├── go.sum
├── main.go
├── retriever.go
├── semantic.go
└── evaluation.go
```

- `main.go`: Application entry point and execution modes.
- `retriever.go`: Keyword-based retrieval.
- `semantic.go`: Embedding generation and semantic retrieval.
- `evaluation.go`: Retrieval evaluation using a query dataset and retrieval metrics.
- `data/knowledge.txt`: Knowledge base containing topic-based text chunks.

## Features

The application provides several modes for exploring LLM and RAG concepts:

| Mode | Description |
|---|---|
| BASIC | Basic LLM interaction |
| INTERACTION | Interactive question-and-answer |
| TEMPLATE | Prompt template usage |
| CHAIN | Chaining LLM operations |
| JSON | Structured JSON output |
| RETRIEVAL | Keyword-based retrieval |
| SEMANTIC | Embedding-based semantic retrieval |
| RAG | End-to-end retrieval-augmented generation |
| EVALUATE | Retrieval quality evaluation |

## Retrieval Configuration

The RAG implementation uses the following baseline configuration:

| Parameter | Value |
|---|---:|
| Embedding model | `nomic-embed-text` |
| Embedding dimensions | 768 |
| LLM | `qwen2.5:3b` |
| Top-K | 2 |
| Minimum similarity | 0.45 |

Top-K limits the number of retrieved chunks, while the similarity threshold filters out chunks with scores below the configured minimum.

These values are experimental baselines for the knowledge base and evaluation dataset in this project. They are not universal optimal settings.

## Evaluation

The retrieval evaluation uses a dataset of 28 queries, including in-scope questions and questions outside the knowledge base.

Baseline results with Top-K = 2 and a semantic similarity threshold of 0.45:

| Metric | Keyword Retrieval | Semantic Retrieval |
|---|---:|---:|
| Precision@K | 0.50 | 0.48 |
| Recall@K | 0.93 | 0.96 |
| Out-of-scope rejection | 1.00 | 1.00 |

The results show that semantic retrieval achieved higher recall on this dataset, while keyword retrieval achieved slightly higher precision. Both approaches rejected the tested out-of-scope queries.

These results apply only to the current knowledge base and evaluation dataset.

## Limitations

This project is an educational implementation intended to explore the RAG pipeline and its behavior.

- Retrieval quality depends on the knowledge base, chunking strategy, and retrieval configuration.
- Relevant retrieved context does not guarantee a correct answer.
- The LLM may overlook important facts or refuse to answer even when supporting information is available.
- Prompt instructions alone cannot guarantee that every generated claim is fully grounded in the retrieved context.

RAG can help ground LLM responses in external information, but it does not automatically eliminate hallucinations.

## Scope

This project focuses on understanding and evaluating the core RAG workflow using a local LLM. It does not aim to provide a production-ready retrieval system.

## License

Add a license file if you intend to distribute this project as open source.