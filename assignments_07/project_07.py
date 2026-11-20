from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from scipy.stats import pearsonr
# smolagents imports
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import pandas as pd
import os

if load_dotenv():
    print('Successfully loaded environment variables from .env')
else:
    print('Warning: could not load environment variables from .env')

client = OpenAI()
print('OpenAI client created.')
api_key = os.getenv("OPENAI_API_KEY")

DATA_DIR = Path(__file__).parent / "happiness_project" #Path("happiness_project")
DATA_DIR

df = None

# Task 1: Define Your Tools
# Tool 1: load_happiness_data

@tool
def load_happiness_data() -> dict:
    """
    Load the World Happiness dataset into memory.
    
    Return:
        A dict with loaded cvs file to Dataframe
    """
    
    global df
    
    COLUMN_NAMES = {
        "Country": "country",
        "Regional indicator": "region",
        "Happiness score": "happiness_score",
        "Ladder score": "happiness_score",   # the 2024 file uses this name
        "GDP per capita": "gdp_per_capita",
    }
    
    frames = []
    for path in sorted(DATA_DIR.glob("world_happiness_*.csv")):
        year_df = pd.read_csv(path, sep=";", decimal=",")
        year_df = year_df.rename(columns=COLUMN_NAMES)
        year_df["year"] = int(path.stem.split("_")[-1])   # "world_happiness_2020" -> 2020
        frames.append(year_df)
    
    df = pd.concat(frames, ignore_index=True)
    return df
        
    
#print(load_happiness_data())

# Tool 2: summarize_column

@tool
def summarize_column(column: str) -> dict:
    """
    Return descriptive statistics for a column in the loaded dataset.

    Use this tool when a user wants summary information about a
    numerical variable, such as its mean, standard deviation,
    minimum value, maximum value, or quartiles.

    The dataset must be loaded first using `load_happiness_data()`.

    Args:
        column (str): Name of the column to summarize.

    Returns:
        dict: Descriptive statistics generated from
        `df[column].describe().to_dict()`.
    """
    
    global df
    
    if df is None:
                return {"error": "Data not loaded. Run load_happiness_data first."}
    
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}
    try:
        return df[column].describe().to_dict()
    
    except Exception as e:
        return {"error": str(e)}
   
# Tool 3: compute_correlation

@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """
    Compute the Pearson correlation coefficient and p-value between two numeric columns.
    
    Args:
        col1 (str): First column name.
        col2 (str): Second column name.
    
    Returns:
        dict: Correlation results.
    """
    
    global df
    
    if df is None:
        return {"error": "Data not loaded. Run load_happiness_data first."}

    if col1 not in df.columns:
        return {"error": f"Column '{col1}' not found."}

    if col2 not in df.columns:
        return {"error": f"Column '{col2}' not found."}

    try:
        temp_df = df[[col1, col2]].dropna()

        r, p = pearsonr(temp_df[col1], temp_df[col2])

        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(r, 4),
            "p_value": round(p, 4)
        }

    except Exception as e:
        return {"error": str(e)}

# Tool 4: get_top_n_countries

@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """
    Return the top N countries ranked by a given column for a specific year.
    
    Args:
        column (str): Numeric column used for ranking.
        year (int): Year to filter the dataset.
        n (int, optional): Number of countries to return.
            Defaults to 5.

    Returns:
        dict: A dictionary containing the ranked countries.
    """
    
    global df
    
    if df is None:
        return {"error": "Data not loaded. Run load_happiness_data first."}
    
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}
    
    try:
        year_df = df[df["year"] == year]
    
        if year_df.empty:
            return {"error": f"No data found for year {year}."}
    
            top_n = (
                year_df
                .sort_values(by=column, ascending=False)
                .head(n)[["country", column]]
            )
    
            return {
                "results": top_n.to_dict(orient="records")
            }
    
    except Exception as e:
        return {"error": str(e)}

# Task 2: Build the Agent

model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)

# Task 3: Run Guided Queries

queries = [
    "Load the happiness data and tell me its shape and column names.",
    "Summarize the happiness_score column.",
    "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
    "Show me the top 5 happiest countries in 2020.",
    "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
]

for query in queries:
    print(f"\n--- Query: {query} ---")
    response = agent.run(query, reset=False)
    print(response)

# Task 4: Your Own Questions
# My query 1
my_query_1 = (
    "What were the top 10 countries by happiness_score in 2024, "
    "and what is the average happiness_score among those countries?"
)   # replace with your question
response_1 = agent.run(my_query_1, reset=False)
print(response_1)
# Comment: 
# This should trigger tool use. The agent will likely call
# get_top_n_countries() and may perform a small calculation on the results.

# My query 2
my_query_2 = (
    "Create a line plot showing the average happiness_score for each year "
    "from 2015 to 2024. Save the plot in outputs/ and briefly describe the trend."
)   # replace with your question
response_2 = agent.run(my_query_2, reset=False)
print(response_2)
# Comment:
# This should trigger both tool use and code generation.
# The agent needs the dataset loaded first, then it must write Python
# code using pandas and matplotlib to group by year, create the chart,
# and save it to outputs/.

# Task 5: Reflection
# --- Reflection ---
#
# 1. In Query 3, the agent reported both the Pearson correlation coefficient
#    and the p-value returned by the compute_correlation tool. It described
#    whether the relationship was positive or negative and indicated its
#    strength. It also used the p-value to explain whether the correlation
#    appeared statistically significant, which made the interpretation more
#    meaningful than simply reporting the numbers.
#
# 2. One response that surprised me was when the agent generated code to
#    create a visualization instead of only calling a tool. After loading
#    the dataset, it grouped the data, created a plot, and saved it to the
#    outputs folder. I expected it to rely mostly on tools, so seeing it
#    write and execute code automatically was impressive.
#
# 3. One additional tool that would make this agent more useful is
#    get_country_trend(country). This tool could return a country's
#    happiness score over time from 2015–2024. It would help answer
#    questions such as "How has Canada's happiness changed over the last
#    decade?" or "Which countries improved the most over time?" without
#    requiring the agent to write extra analysis code.


# Create the agent
agent = CodeAgent(
    tools=[
        load_happiness_data,
        summarize_column,
        compute_correlation,
        get_top_n_countries
    ],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=[
        "pandas",
        "matplotlib.pyplot",
        "scipy.stats"
    ],
    max_steps=8,
)


if __name__ == "__main__":

    print("\n=== Query 1 ===")
    response_1 = agent.run(
        "Load the dataset and tell me how many rows and columns it contains."
    )
    print(response_1)

    print("\n=== Query 2 ===")
    response_2 = agent.run(
        "Provide descriptive statistics for happiness_score."
    )
    print(response_2)

    print("\n=== Query 3 ===")
    response_3 = agent.run(
        "What is the correlation between happiness_score and gdp_per_capita?"
    )
    print(response_3)

    print("\n=== Query 4 ===")
    response_4 = agent.run(
        "What are the top 10 countries by happiness_score in 2024?"
    )
    print(response_4)

    print("\n=== My Query 1 ===")
    my_query_1 = (
        "What were the top 10 countries by happiness_score in 2024, "
        "and what is the average happiness_score among them?"
    )
    response_5 = agent.run(my_query_1, reset=False)
    print(response_5)

    print("\n=== My Query 2 ===")
    my_query_2 = (
        "Create a line plot showing the average happiness_score for each "
        "year from 2015 to 2024. Save the plot in outputs/ and describe "
        "the overall trend."
    )
    response_6 = agent.run(my_query_2, reset=False)
    print(response_6)