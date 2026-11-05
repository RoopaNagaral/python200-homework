import os
import json
import sys
from dotenv import load_dotenv
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")

# Task 1: Setup and System Prompt
if load_dotenv():
    print("Loaded environment variables from .env file")
else:
    print("Warning: .env file not found. Make sure OPENAI_API_KEY is set.")

# Initialize OpenAI client
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

# 1. Start with the system message that defines the bot's personality
messages = [
    {"role": "system", 
    "content": "You are a friendly, competent job application coach who helps candidates find jobs that fit their needs. Ask questions to learn their job preferences, and use case before recommending options."}
]

system_prompt = """
You are an experienced job application coach assisting job seekers with resumes,
cover letters, interview preparation, LinkedIn profiles, professional summaries,
and other job application materials.

Your goal is to help users present their skills, experience, and achievements
clearly and professionally. Provide constructive feedback, suggest improvements,
and explain the reasoning behind your recommendations.

Stay focused on job application materials and career-related communication.
If a user asks about unrelated topics, politely redirect the conversation back
to job application support.

Always remind the user to carefully review, edit, and personalize any content
you generate before submitting it to an employer, recruiter, or professional
networking site.

Acknowledge that you may not know the specific standards, expectations, or
industry norms for every field, company, or location. Encourage users to apply
their own judgment and adapt your suggestions to their situation.

Do not invent qualifications, experience, certifications, or achievements.
Help users present their real experience effectively and honestly.

Maintain a professional, encouraging, and supportive tone.
"""

print(system_prompt)

# Deliberate design choice:
# I explicitly instructed the assistant not to invent qualifications or
# accomplishments because job applicants may be tempted to exaggerate their
# experience. This helps keep the chatbot's advice ethical and realistic while
# producing application materials that accurately represent the user.

# Task 2: Bullet Point Rewriter

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list. Each item should have two keys:
    - "original"   
    - "improved"
     
    Respond ONLY with valid JSON, no other text.

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    response = get_completion(messages)
    
    response = response.strip()

    if response.startswith("```json"):
        response = response.removeprefix("```json").strip()

    if response.startswith("```"):
        response = response.removeprefix("```").strip()

    if response.endswith("```"):
        response = response.removesuffix("```").strip()

    results = json.loads(response)
    # Print side-by-side comparison 
    for item in results: 
        print(f"Original : {item['original']}") 
        print(f"Improved : {item['improved']}") 
        print("-" * 80) 
    return results
    
bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

print("\nBullet points:")
rewritten_bullets = rewrite_bullets(bullets)

# Comment: 
# The original bullets are weak because they use vague language, provide little 
# context, and do not clearly communicate impact or responsibility. The model 
# improved them by using stronger action verbs such as "assisted," 
# "prepared," and "collaborated," while making the contributions sound more 
# professional and specific without inventing achievements or metrics.

# Task 3: Cover Letter Generator
def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    response_text = get_completion(messages)
    
    return response_text
    # Your code here: call get_completion() and return the result
    
job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed \
a Python course and built data pipelines using Pandas."

generate_cover_letters = generate_cover_letter(job_title, background)
print("\nCover letter:")
print(generate_cover_letters)

# Comment:
# I chose examples that demonstrate successful career transitions because they
# show how to connect previous experience to a new role rather than simply
# listing skills. The few-shot examples help control the tone, structure, and
# level of specificity of the output. By showing examples that are confident,
# concrete, and tailored to the applicant's background, the model is more
# likely to generate a personalized cover letter opening instead of generic
# statements such as "I am excited to apply for this position."

# Task 4: Moderation Check
def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    if flagged:
        print(
            "\nYour message may contain content that cannot be processed."
            "Please rephrase your request and try again."
        )
        return False

    return True
    
text = "I want to hit my friend"
text1 = "Iwant to help my friend"

print("Flagged text", is_safe(text))
print("Safe text", is_safe(text1))

# Task 5: The Chatbot Loop
def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            rewritten_bullets = rewrite_bullets(raw_bullets)
            # YOUR CODE: call rewrite_bullets() and print the results

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            generate_cover_letters = generate_cover_letter(job_title, background)
            print("\nCover letter opening:")
            print(generate_cover_letters)

        # 7. Otherwise, handle it as a regular chat turn
        else:
            
            messages.append(
                {"role": "user", "content": user_input}
            )

            reply = get_completion(messages)

            print(f"\nJob Application Helper: {reply}\n")

            messages.append(
                {"role": "assistant", "content": reply}
            )
            pass


if __name__ == "__main__":
    run_chatbot()
    

# Task 6: Ethics Reflection
# Chose Option A — Comment block:
"""
1. How might the bot produce biased advice?

A bot is trained on existing text, and that text may not represent all people equally. 
If most of the training data comes from certain regions, industries, or professional groups, 
the bot may favor those perspectives. For example, it might recommend a direct, confident writing style 
that is common in U.S. business culture while undervaluing communication styles that are more formal or 
indirect in other cultures. It may also be more familiar with careers in technology, business, or other 
heavily documented industries than with less-represented fields, leading to uneven or biased recommendations.

2. What could go wrong if a job seeker submitted the bot's output without reviewing it?

The generated content might contain mistakes, exaggerations, or details that do not accurately reflect 
the applicant's experience. A resume or cover letter could sound generic and fail to highlight the person's 
unique strengths. In some cases, the bot might misunderstand a job description and include irrelevant skills or 
achievements. Submitting AI-generated content without reviewing it could therefore reduce credibility and even 
create issues if the applicant is asked about information that is inaccurate.

3. One guardrail I would add

I would require a review reminder before the user copies or downloads the final document. 
The warning could say: "Please verify that all skills, experiences, dates, and achievements are accurate 
before submitting this material to an employer." This simple guardrail encourages users to check for errors and 
take responsibility for the final content, reducing the chance of misinformation being sent to employers.
"""