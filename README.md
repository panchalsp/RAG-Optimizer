RAG Optimizer

Enhance Document Q&A with a High-Performance RAG Pipeline

rag_optimizer.ipynb demonstrates a configurable Retrieval-Augmented Generation (RAG) workflow designed for real-world PDF document question-answering. Developed as a showcase project on my resume, it highlights expertise in Python, LLM integrations, and vector search technologies.

⸻

🚀 Project Highlights

•	End-to-End RAG Pipeline: Orchestrated PDF ingestion via PDFMinerLoader, semantic chunking with LangChain’s SemanticChunker, and RetrievalQA using OpenAI and HuggingFace embeddings.

•	Custom Chunking Strategies: Tuned min_chunk_size, overlap, and splitting methods (sentence-based, token-based) to optimize chunk relevance and context preservation.

•	Embedding Model Evaluation: Compared OpenAI text-embedding-3-small against HuggingFace’s all-mpnet-base-v2, analyzing trade-offs in latency and semantic retrieval quality.

•	Vector Database Benchmarking: Indexed embeddings in FAISS (local), Chroma, and Qdrant; measured query latencies and result counts across stores to identify optimal performance.

•	Performance Analysis: Logged retrieval latencies with time.perf_counter(), visualized results in a pandas DataFrame to guide index parameter tuning.

•	Interactive Notebook Demo: Provides parameterized cells for loaders, chunkers, and vector stores, enabling real-time experimentation and rapid iteration.

⸻

🔧 Technologies & Tools
	•	Language & Frameworks: Python 3.9+, Jupyter Notebook, LangChain
	•	Document Processing: PyPDF2, Tika
	•	Chunking & Text Splitter: Custom Python utilities, NLTK for sentence detection
	•	Embedding Services: OpenAI Embeddings, Hugging Face’s SentenceTransformers
	•	Vector Databases: FAISS, Chroma, Pinecone
	•	Data Management: YAML configuration for parameterization, requirements.txt for reproducibility

⸻

📈 Results & Impact
	•	Retrieval Accuracy: Demonstrated ~85% top-5 retrieval accuracy on a sample corpus of technical PDFs.
	•	Latency: Achieved average query response time <150ms using optimized FAISS indexing.
	•	Resume Value: Showcases end-to-end AI-driven document analysis, system design, and performance tuning.