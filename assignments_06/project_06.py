from pathlib import Path
from dotenv import load_dotenv
import string
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms.openai import OpenAI
import sys
import asyncio
import nest_asyncio
nest_asyncio.apply()

sys.stdout.reconfigure(encoding="utf-8")

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")
    
# Step 1: Setup
docs_dir = Path("D:/CTD_CodeTheDream/python-200-v1/lessons/06_AI_augmentation/resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

# Step 2: Load the Documents
docs = SimpleDirectoryReader(docs_dir).load_data()
print(f"Number of documents Loaded: {len(docs)}")

print("\nFile Names:")
for i, doc in enumerate(docs, start=1):
    print(f"{i}. {doc.metadata['file_name']}")
    
# Step 3: Build the Index and Query Engine

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(
    model="gpt-4o-mini"
)

index = VectorStoreIndex.from_documents(docs)

query_engine = index.as_query_engine(similarity_top_k=3)

print("\nIndex built successfully. Ready to answer questions.")

# Step 4: Query the Assistant

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response.response)
    
    if response.source_nodes:
        top_node = response.source_nodes[0]

        document_name = top_node.node.metadata.get(
            "file_name",
            top_node.node.metadata.get("document_name", "Unknown")
        )

        similarity_score = top_node.score
        chunk_preview = top_node.node.text[:200]

        print("\nTOP RETRIEVED SOURCE NODE")
        print(f"Document: {document_name}")
        print(f"Similarity Score: {similarity_score}")
        print(f"Chunk Text (first 200 chars): {chunk_preview}")
    else:
        print("\nNo source nodes returned.")

"""
Reflection: The assistant generally sounded confident and accurate. Most answers matched the retrieved documents, 
especially the questions about Groundwork's history and catering services. However, it was surprising that 
some answers were supported by source chunks that did not clearly relate to the question, suggesting the 
retrieval step could be improved for better relevance.
"""
        
# Step 5: Find a Failure
question = "Which dairy-free milk option is the most popular among loyalty program members?"

response = query_engine.query(question)
print("\nStep 5: Answer")
print(f"\nQ: {question}")
print(f"A: {response.response}")

for i, node in enumerate(response.source_nodes[:3], start=1):
    document_name = node.node.metadata.get(
        "file_name",
        node.node.metadata.get("document_name", "Unknown")
    )

    similarity_score = node.score
    chunk_preview = node.node.text[:200]

    print(f"\nRETRIEVED SOURCE NODE {i}")
    print(f"Document: {document_name}")
    print(f"Similarity Score: {similarity_score}")
    print(f"Chunk Text (first 200 chars): {chunk_preview}")
    
"""
1. I asked which dairy-free milk option is most popular among loyalty program members because it requires 
connecting two topics and relies on data that is unlikely to exist in the documents.

2. The retrieval system returned documents about menu items, FAQs, and the loyalty program, but none contained 
information about customer preferences or popularity. The information was simply missing.

3. If the model still provided a specific answer (for example, claiming oat milk is the most popular), it would 
be guessing. This shows that AI can sound confident even when supporting evidence is unavailable, so responses 
should always be checked against retrieved sources.

4. To improve the system, I would add stronger safeguards that instruct the model to say "I don't know" when 
retrieved documents do not contain enough evidence. I would also improve retrieval ranking and include source 
validation to ensure answers are grounded in the retrieved content.
"""

# Step 6: Reflection
# 1. My LlamaIndex implementation took only a few dozen lines of code to load
# documents, create embeddings, build an index, and query the data. If I had
# implemented chunking, embedding generation, storage, similarity search, and
# retrieval myself, it would have required significantly more code and effort.
# This shows the value of a framework because it automates complex tasks and
# lets developers focus on building the application rather than the underlying
# infrastructure.
#
# 2. A useful business application would be a company knowledge base for
# employees. Workers could ask questions about HR policies, benefits,
# procedures, or onboarding documents and receive answers directly from the
# organization's documents instead of searching through multiple files.
#
# 3. One failure mode that RAG cannot fully prevent is hallucination. Even when
# the correct documents are retrieved, the language model may still misinterpret
# information, make unsupported assumptions, or generate details that are not
# actually present in the source documents.