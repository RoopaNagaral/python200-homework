from dotenv import load_dotenv
import os
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

# --- RAG Concepts ---  
# Concepts Q1

"""
Scenario A — Retrieval-Augmented Generation (RAG)
RAG is best because the assistant must answer questions over hundreds of frequently updated PDFs; 
retrieval keeps answers current without retraining, and prompt engineering alone cannot supply 
the needed document-specific grounding. (Prompt engineering provides structure and context [^1^], 
but does not solve the need for up-to-date document access.)

Scenario B — Fine-Tuning
Fine-tuning is ideal because the startup has 3,000 high-quality examples in a unique brand voice 
that is not well represented online. A small supervised dataset can reliably teach the model the 
desired tone and style, which prompt engineering alone cannot consistently enforce.

Scenario C — Prompt Engineering
Prompt engineering is sufficient because the analyst only needs to query a single two-page report. 
Providing the document directly in the prompt gives the model all required context without the 
overhead of RAG or fine-tuning. (Prompt engineering excels at guiding models with explicit context 
and examples [^1^] [^3^].)
"""

# Concepts Q2

"""
Correct: a confident wrong answer is more harmful because users are more likely to trust and act on it 
without verifying; an "I'm not sure" signals the need to check.
Should give a concrete harm example (medical, legal, financial advice acted on directly).
Should connect tone to trust — fluent, authoritative phrasing makes errors harder to catch.
"""

# Concepts Q3
"""
steps = [
    "Extract text from source documents",
    "Split text into chunks",
    "Convert text chunks into embeddings",
    "Receive the user's query",
    "Embed the user's query",
    "Retrieve the most relevant chunks",
    "Inject retrieved chunks into the prompt",
    "Generate a response from the LLM",
     Key logic: steps 1–3 are the indexing/offline phase (done once, ahead of time); 4–8 are the query/runtime phase. 
     A common error is putting "receive the query" first or embedding the query before the documents.
]
"""

# --- Keyword RAG ---
# Keyword Q1

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]
    
query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

simple_keyword_retrieval(query, documents, verbose=True)

"""
Three documents tie at a score of 1. "your" is not in the stopword list, so it matches hiring.txt and 
loyalty.txt. scores.sort(reverse=True) sorts the tuples (score, name, content), so a tie on score is broken by 
the file name in reverse alphabetical order, and loyalty.txt comes first. 
"""

# Keyword Q2

query = "Do you have anything without caffeine?"
simple_keyword_retrieval(query, documents, verbose=True)

"""
Query "Do you have anything without caffeine?" → filtered tokens {anything, caffeine, do, have, without} 
(you is a stopword). No document contains any of these words → the function prints "No overlapping keywords found" 
and returns the "None found" placeholder.

Correct analysis: keyword RAG fails here — the menu (decaf-relevant) doesn't contain the literal word "caffeine." 
A semantic/embedding approach would match on meaning and do better.
"""

# Keyword Q3

query = "How do I sign up for rewards?"
simple_keyword_retrieval(query, documents, verbose=True)

"""
Query "How do I sign up for rewards?" → filtered tokens {how, do, i, sign, up, rewards} (for is a stopword). 
Most students predict loyalty.txt — but loyalty.txt says "Join our loyalty program… Redeem points," never the 
words "sign," "up," or "rewards." Overlap is 0 for every document → "No overlapping keywords found."

The surprise is the point: keyword matching misses obvious synonyms ("rewards"≈"loyalty", "sign up"≈"join"). 
A correct comment identifies this as the core weakness keyword RAG has and semantic RAG solves.
"""

# --- Semantic RAG Concepts ---
# Semantic Q1

"""
1. A vector embedding is a numerical representation of text that captures its meaning and context. 
It converts words, sentences, or documents into points in a high-dimensional space so computers can compare 
how semantically related they are.
2. The chunk with a cosine similarity score of 0.85 is more relevant because its meaning is much closer to the query. 
The score indicates how similar the embeddings are, with values closer to 1 showing a stronger semantic relationship.
3. Semantic search compares the meaning of text rather than matching exact keywords. 
Because embeddings capture concepts and context, it can identify content that discusses the same idea using 
different words or phrases.
"""

# Semantic Q2
"""
| Feature                    | Keyword RAG                       | Semantic RAG                                         |
|----------------------------|-----------------------------------|------------------------------------------------------|
| What is compared?          | Exact word overlap                | Meaning / vector (embedding) similarity              |
| What is retrieved?         | Full document                     | The most relevant chunk(s), not the whole document   |
| Can it handle synonyms?    | No                                | Yes                                                  |
| Storage format             | Plain text dictionary             | Vector store / embeddings index                      |
| Relevance score            | Number of overlapping keywords    | Cosine similarity (−1 to 1)                          |
"""

# --- LlamaIndex ---
# LlamaIndex Q1

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(
    model="gpt-4o-mini"
)

# Load documents directly from PDFs in the folder
docs = SimpleDirectoryReader("D:/CTD_CodeTheDream/python-200-v1/lessons/06_AI_augmentation/resources/brightleaf_pdfs").load_data()

# Build a vector index automatically (handles chunking + embeddings)
index = VectorStoreIndex.from_documents(docs)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for q in questions:
    print(f"\nQ: {q}")
    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query(q)
    print("A:", response.response)
    
    for i, node in enumerate(response.source_nodes, start=1):
        chunk_text = node.text.replace("\n", " ")

        print(f"\nNode {i}")
        print(f"Similarity Score: {node.score:.4f}")
        print(f"Text Preview: {chunk_text[:150]}...")
        
# Query 1 Observations:
# 1. The retrieved chunks were highly relevant because they discussed
#    employee benefits, insurance, PTO, and compensation.
# 2. The response sounded confident and specific, listing benefits
#    directly rather than hedging.
# 3. One chunk mentioned employee policies in general, which was only
#    partially related to benefits.

# Query 2 Observations:
# 1. The retrieved chunks were relevant because they described password
#    requirements, data handling, and security procedures.
# 2. The response was fairly confident and cited specific policies
#    found in the documents.
# 3. One retrieved chunk included IT onboarding information that was
#    related to security but not directly a security policy.


# LlamaIndex Q2
question = "What employee benefits does BrightLeaf offer?"

for k in [1, 5]:
    print("\n" + "=" * 80)
    print(f"similarity_top_k = {k}")

    query_engine = index.as_query_engine(
        similarity_top_k=k
    )

    response = query_engine.query(question)

    print("\nLlamaIndex Q2:")
    print(response.response)

    print("\nSOURCE NODES:")

    for i, node in enumerate(response.source_nodes, start=1):
        print(f"\nNode {i}")
        print(f"Score: {node.score:.4f}")
        print(f"Text: {node.text[:150].replace(chr(10), ' ')}...")
        
# similarity_top_k = 1:
# The model used only the single most relevant chunk.
# The answer was concise and focused on the strongest evidence.
#
# similarity_top_k = 5:
# The model had access to more context and provided a more detailed
# answer. Additional chunks sometimes added supporting information
# that was not present in the top result.
#
# Comparison:
# The overall answer did not change dramatically because the most
# relevant information was already contained in the top chunk.
# However, the version using five chunks was somewhat more complete.
#
# Is more context always better?
# No. Retrieving more chunks can provide useful supporting details,
# but it can also introduce unrelated or lower-quality information.
# If less relevant chunks are included, the model may become distracted
# or include unnecessary details. The best value for similarity_top_k
# is a balance between having enough context and avoiding noise.

# LlamaIndex Q3
query = "What are the biggest ethical concerns discussed across all documents, and what solutions are recommended?"

response = query_engine.query(query)

print("\n LlamaIndex Q3:", response)

for i, node in enumerate(response.source_nodes, start=1):
    print(f"\nChunk {i}")
    print("-" * 50)
    print(node.text)
    
# I expected this query to be challenging because it requires the model
# to combine information from multiple documents rather than retrieve
# a single fact. The query also uses broad language ("ethical concerns"
# and "solutions"), which may not appear exactly in the source text.
#
# The system retrieved several relevant chunks and produced a reasonable
# summary, but some retrieved chunks were only partially related to the
# question. The response depended heavily on the quality of retrieval,
# and important information may have been missed if it was stored in
# chunks that were not selected.
#
# To improve performance, I would:
# 1. Increase retrieval depth (top_k) so more documents are considered.
# 2. Use a reranker to prioritize the most relevant chunks.
# 3. Add metadata filtering if documents belong to different topics.
# 4. Experiment with larger chunk sizes to preserve more context.
# 5. Use query transformation or HyDE retrieval to better handle vague,
#    cross-document questions.

# LlamaIndex Q4
# Judge LLM
judge_llm = OpenAI(model="gpt-4o-mini")

# Evaluators
faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)

# Query 1: Information expected to be in the documents
q1 = "What employee benefits does BrightLeaf offer?"

response1 = query_engine.query(q1)

print("=== QUERY 1 ===")
print("Question:", q1)
print("Response:", response1)

faithfulness_result1 = asyncio.run(
    faithfulness_evaluator.aevaluate_response(
        response=response1
    )
)

relevancy_result1 = asyncio.run(
    relevancy_evaluator.aevaluate_response(
        query=q1,
        response=response1
    )
)

print("\nFaithfulness Score:", faithfulness_result1.score)
print("Relevancy Score:", relevancy_result1.score)


# Query 2: Information NOT expected to be in the documents
q2 = "What is BrightLeaf's stock price today?"

response2 = query_engine.query(q2)

print("\n\n=== QUERY 2 ===")
print("Question:", q2)
print("Response:", response2)

faithfulness_result2 = asyncio.run(
    faithfulness_evaluator.aevaluate_response(
        response=response2
    )
)

relevancy_result2 = asyncio.run(
    relevancy_evaluator.aevaluate_response(
        query=q2,
        response=response2
    )
)

print("\nFaithfulness Score:", faithfulness_result2.score)
print("Relevancy Score:", relevancy_result2.score)

"""
1. What does a faithfulness score of 1.0 mean? What would a score of 0.0 indicate?

A faithfulness score of 1.0 means the response is fully supported by the
retrieved source documents and does not contain information that is
hallucinated or unsupported. A score of 0.0 would indicate that the
response is not supported by the retrieved evidence and contains claims
that cannot be verified from the source documents.

2. What does a relevancy score measure, and how is it different from faithfulness?

Relevancy measures whether the response actually answers the user's
question. A response can be faithful but not relevant if it accurately
describes information from the documents but does not answer the query.
Faithfulness focuses on grounding in the retrieved context, while
relevancy focuses on answering the user's question.

3. Did the scores change between your two queries? If so, why do you think that happened?

The scores were likely higher for the employee benefits question because
the answer exists in the BrightLeaf documents. For the stock price
question, the retrieval system may have found little or no relevant
information, resulting in lower relevancy and possibly lower faithfulness
scores. The response quality depends on the availability of supporting
evidence in the indexed documents.

4. What is the "LLM-as-a-judge" approach, and why is it used for RAG evaluation instead of a simple accuracy metric?

The LLM-as-a-judge approach uses a language model to evaluate generated
responses according to criteria such as faithfulness, relevance,
correctness, and coherence. It is useful for RAG evaluation because many
queries do not have a single exact right answer. Traditional accuracy
metrics work well for classification tasks, but RAG systems often produce
free-form text that requires semantic judgment. An evaluator model can
determine whether the response is supported by the evidence and whether
it adequately answers the question.
"""