from dotenv import load_dotenv
from openai import OpenAI
import sys
import json

sys.stdout.reconfigure(encoding="utf-8")

# --- Completions API --- 
# API Q1

load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

print("Response text:", response.choices[0].message.content)
print("Model Name:", response.model)
print("Token count:", response.usage.total_tokens)

# API Q2
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    response1 = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        temperature=temp
    )
    
    print(f"\nTemperature: {temp}")
    print(response1.output_text)
    
# Temperature 0 tends to give the most predictable and repeatable answer.
# Temperature 0.7 usually produces more varied and creative names while
# remaining reasonably focused. Temperature 1.5 often generates the most
# diverse and unconventional ideas, but results may be less consistent.
# If you need a consistent, reproducible output, use temperature = 0.

# API Q3
response2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)

for i, choice in enumerate(response2.choices, start=1):
    print(f"\nCompletion {i}:")
    print(choice.message.content)


# API Q4
response3 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Explain how neural networks work."
        }
    ],
    max_tokens=15
)

print("")
print(response3.choices[0].message.content)

# What happened?
# The response was cut off after reaching the max_tokens limit, so it
# could not provide a complete explanation.
# Why use max_tokens?
# In a real application, max_tokens helps control cost, reduce latency,
# and prevent overly long responses when you only need a short answer.

# ---- System Messages and Personas ----
# System Q1

# 1. Start with the system message that defines the bot's personality
# Personality 1: Patient Python Tutor
messages = [
    {
        "role": "system",
        "content": ("You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement.")
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is."
    }
]

response4 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("=== Python Tutor ===")
print(response4.choices[0].message.content)


# Personality 2: Pirate Captain
messages = [
    {
        "role": "system", "content": ("You are a dramatic pirate captain. You explain everything using pirate language and sea-related metaphors.")
    },
    {
        "role": "user", "content": "I don't understand what a list comprehension is."
    }
]

response5 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("\n=== Pirate Captain ===")
print(response5.choices[0].message.content)

# What changed?
# The answer stayed about the same topic, but the tone, style,
# vocabulary, and personality changed based on the system message.
# The tutor gives a simple and encouraging explanation, while the
# pirate explains the concept with pirate-themed language and metaphors.

# Sysytem Q2
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response6 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("\nSystem Q2:")
print(response6.choices[0].message.content)

# Why does the model know Jordan's name, even though it's stateless?
# The model does not remember previous API calls. It knows the name
# because the entire conversation history, including the message
# "My name is Jordan and I'm learning Python.", was included in the
# messages list sent with this request. The context is provided by
# the application, not stored by the model between calls.

# ---- Prompt Engineering ----
# Prompt Q1 — Zero-Shot

def get_completion(prompt: str, model="gpt-4o-mini", temperature=0):
    """
    Send a prompt to the model and return the assistant's text reply.
    This helper keeps our examples clean and focused on the prompt itself.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}], 
        temperature=temperature,
    )
    return response.choices[0].message.content

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = (
    "Classify the sentiment of each review as positive, negative, or mixed.\n\n"
)

for i, review in enumerate(reviews, start=1):
    prompt += f"Review {i}: {review}\n"

response7 = get_completion(prompt, temperature=0)

print("\nPrompt Q1:")
print(response7)

# Prompt Q2 — One-Shot
prompt = """
Classify each review as positive, negative, or mixed.

Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Review 1: The onboarding process was smooth and the team was welcoming.
Review 2: The software crashes constantly and support never responds.
Review 3: Great price, but the documentation is nearly impossible to follow.

Return your answers in this format:
Review 1: <sentiment>
Review 2: <sentiment>
Review 3: <sentiment>
"""

response8 = get_completion(prompt, temperature=0)

print("\nPrompt Q2:")
print(response8)

# Did adding one example change the format or consistency?
# Yes. Providing a single example (few-shot prompting) usually makes
# the output format more consistent and predictable because the model
# can imitate the pattern shown in the example. Without an example,
# the model might still give correct labels, but it could vary in
# formatting or add extra explanations.

# Prompt Q3 — Few-Shot
# Three examples to classify
examples = [
    "I loved the movie. The acting was excellent and the story was inspiring.",  # positive
    "The service was terrible. I would not recommend this restaurant.",           # negative
    "The food was good, but the wait time was far too long."                      # mixed
]

# -----------------------
# Zero-shot prompting
# -----------------------
print("\nZERO-SHOT RESULTS")
for text in examples:
    prompt = f"""
    Classify the sentiment of the following text as Positive, Negative, or Mixed.

    Text: "{text}"
    """
    result = get_completion(prompt)
    print(result)

# -----------------------
# One-shot prompting
# -----------------------
print("\nONE-SHOT RESULTS")
for text in examples:
    prompt = f"""
    Classify the sentiment as Positive, Negative, or Mixed.

    Example:
    Text: "The product is great and works perfectly."
    Sentiment: Positive

    Text: "{text}"
    Sentiment:
    """
    result = get_completion(prompt)
    print(result)

# -----------------------
# Few-shot prompting
# -----------------------
print("\nFEW-SHOT RESULTS")

for text in examples:
    prompt = f"""
    Classify the sentiment as Positive, Negative, or Neutral.

    Example 1:
    Text: "I love this product."
    Sentiment: Positive

    Example 2:
    Text: "This was terrible."
    Sentiment: Negative
    
    Example 3:
    Text: "The price was great, but the documentation was confusing."
    Sentiment: Mixed

    Text: "{text}"
    Sentiment:
    """

    result = get_completion(prompt)
    print(f"Text: {text}")
    print(f"Sentiment: {result}")

# Comparison:
# Zero-shot is best when the task is simple and the model already understands it well.
# One-shot is useful when you want to show a single example of the desired format or behavior.
# Few-shot is preferred for more complex tasks because multiple examples help guide the model,
# usually producing the most accurate and consistent results.

# prompt Q4 — Chain of Thought
prompt = """
    Show your step-by-step reasoning, then give the final answer on its own line labelled: Final answer: <value>
    
    Problem: A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later takes a new job that 
    pays $7,500 more per year than her post-raise salary. What is her final annual salary?
"""
response9 = get_completion(prompt, temperature=0)

print("\nPrompt Q4:")
print(response9)

# Comment:
# Asking a model to work through the key steps of a multi-step math problem
# often improves accuracy because it encourages the model to calculate and
# track intermediate values instead of jumping directly to an answer.

# Prompt Q5 — Structured Output
prompt = """
Classify the sentiment of the review and respond ONLY with valid JSON.

Keys:
- sentiment (positive/negative/mixed)
- confidence (0-1 scale)
- reason (one short sentence)

Review:
I've been using this tool for three months. It handles large datasets well,
but the UI is clunky and the export options are limited.
"""

response10 = get_completion(prompt, temperature=0)

print("\nPrompt Q5:")
print("Raw response:", response10)

try:
    result = json.loads(response10)
    print("\nAfter parsing to json:")
    print("Sentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])
    print("Reason:", result["reason"])

except json.JSONDecodeError as e:
    print("\nJSON PARSE ERROR:")
    print(e)
    print("\nRaw response for debugging:")
    print(response10)
    
# Prompt Q6 — Delimiters
user_text = "First boil a pot of water. Once boiling, add a handful of salt and the pasta. Cook for 8-10 minutes until all dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

Review : 
```
{user_text}
```
"""

user_text1 = "The sun was setting behind the hills, and a cool breeze moved through the trees. Everyone enjoyed the quiet evening."

prompt1 = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

Review : 
```{user_text1}```
"""

response11 = get_completion(prompt, temperature=0)
response12 = get_completion(prompt1, temperature=0)

print("\nPrompt Q6:")
print(response11)
print("\n",response12)

#Delimiters clearly separate the user's content from the instructions given to the model. 
#This helps prevent confusion about which text is data and which text is a command, 
#reducing the risk that content inside the user input is mistakenly interpreted as instructions.

# ---- Ollama ----
# Ollama Q1

prompt = "Explain what a large language model is in two sentences."

response13 = get_completion(prompt, temperature=0)
print("\nOllama:")
print(response13)

"""
Ollama (qwen3:0.6b) output:
A large language model is a complex system trained on vast amounts of text to understand and generate human-like
conversations, enabling tasks like writing articles, answering questions, or creating stories. It processes
information by identifying patterns and learning from massive datasets, making it versatile for various
applications.
"""

# Differences noticed:
# The Ollama response focused more on what LLMs can do (writing articles,
# answering questions, creating stories), while the OpenAI response explained
# more about how LLMs work by mentioning deep learning and neural networks.
# The OpenAI response was slightly more technical and precise, whereas the
# Ollama response was broader and easier for a beginner to understand.

# One advantage of running a model locally:
# - Better privacy because prompts and data remain on your own computer and
#   do not need to be sent to a cloud service.

# One disadvantage of running a model locally:
# - Local models are often smaller and less capable than large cloud-hosted
#   models, so their responses may be less accurate or detailed.