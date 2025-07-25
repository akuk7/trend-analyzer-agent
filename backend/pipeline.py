from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import Agent
from datetime import datetime, timedelta
from google.genai import types

from exa_py import Exa
from tavily import TavilyClient
from firecrawl import FirecrawlApp
from dotenv import load_dotenv
import os
import asyncio

# Load environment variables from .env file
load_dotenv()
api_base = os.getenv("NEBIUS_API_BASE")
api_key = os.getenv("NEBIUS_API_KEY")

# Model configuration
nebius_model = LiteLlm(
    model="openai/meta-llama/Meta-Llama-3.1-8B-Instruct",
    api_base=api_base,
    api_key=api_key
)

# --- Tool 1: Exa Search ---
def exa_search_ai(_: str) -> dict:
    try:
        results = Exa(api_key=os.getenv("EXA_API_KEY")).search_and_contents(
            query="Latest AI news OR new LLM models OR AI/Agents advancements",
            include_domains=["twitter.com", "x.com"],
            num_results=10,
            text=True,
            type="auto",
            highlights={"highlights_per_url": 2, "num_sentences": 3},
            start_published_date=(datetime.now() - timedelta(days=30)).isoformat()
        )
        return {
            "type": "exa",
            "results": [r.__dict__ for r in results.results]
        }
    except Exception as e:
        return {
            "type": "exa",
            "error": f"Exa search failed: {str(e)}",
            "results": []
        }

# --- Tool 2: Tavily Search ---
def tavily_search_ai_analysis(_: str) -> dict:
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search(
            query="AI benchmarks OR AI/LLM statistics OR AI providers analysis",
            search_depth="advanced",  # search depth for more comprehensive results
            time_range="week",        # time range one week
            include_domains=["artificialanalysis.ai"]  # Replace with relevant websites
        )
        return {
            "type": "tavily",
            "results": response.get("results", [])
        }
    except Exception as e:
        return {
            "type": "tavily",
            "error": f"Tavily search failed: {str(e)}",
            "results": []
        }

# --- Tool 3: Firecrawl scrapper ---
def firecrawl_scrape_nebius(_: str) -> dict:
    firecrawl = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
    try:
        scrape_result = firecrawl.scrape_url(
            url="https://studio.nebius.com/",
            formats=["markdown"],
            only_main_content=True
        )
        
        if scrape_result.success:
            return {
                "type": "firecrawl",
                "markdown": scrape_result.markdown
            }
        else:
            return {
                "type": "firecrawl",
                "error": "Scraping failed."
            }

    except Exception as e:
        return {
            "type": "firecrawl",
            "error": str(e)
        }

print("[Pipeline] Starting AI analysis pipeline...")

# --- Agent 1: Exa AI News ---
print("[Pipeline] Running ExaAgent...")
exa_agent = LlmAgent(
    name="ExaAgent",
    model=nebius_model,
    description="Fetches latest AI news, LLMs, and advancements using Exa.",
    instruction="""
       Use the exa_search_ai tool to fetch the latest information about AI, new LLMs, and advancements in the field from Twitter and X.
       Prefix your response with "**🔥ExaAgent:**" to clearly identify your output.
       """,
    tools=[exa_search_ai],
    output_key="exa_results"
)
print("[Pipeline] ExaAgent setup complete.")

# --- Agent 2: Tavily AI Analysis ---
print("[Pipeline] Running TavilyAgent...")
tavily_agent = LlmAgent(
    name="TavilyAgent",
    model=nebius_model,
    description="Fetches AI benchmarks, statistics, and analysis using Tavily.",
    instruction="""
    Use the tavily_search_ai_analysis tool to retrieve benchmarks, statistics, and relevant analysis on AI.
    Prefix your response with "**🐳TavilyAgent:**" to clearly identify your output.
    """,
    tools=[tavily_search_ai_analysis],
    output_key="tavily_results"
)
print("[Pipeline] TavilyAgent setup complete.")

# --- Agent 3: Summary & Formatting ---
print("[Pipeline] Running SummaryAgent...")
summary_agent = LlmAgent(
    name="SummaryAgent",
    model=nebius_model,
    description="Summarizes and formats Exa and Tavily results.",
    instruction="""
You are a summarizer and formatter.
- Combine the information from 'exa_results' (latest AI updates) and 'tavily_results' (AI benchmarks and analysis).
- Present a structured summary, highlighting key trends, new LLMs, and relevant statistics.
- Use markdown formatting for clarity and readability.
- Use emojis like 🚀 for new launches, 📊 for statistics, and 📈 for trends to make the summary more engaging.
- Structure information using bullet points and headings for better organization.
- Prefix your response with "**🍥SummaryAgent:**" to clearly identify your output.
- **Only use the tools provided to you. Do not call any other functions or tools.**
""",
    tools=[],
    output_key="final_summary"
)
print("[Pipeline] SummaryAgent setup complete.")

# --- Agent 4: Firecrawl Scrape ---
print("[Pipeline] Running FirecrawlAgent...")
firecrawl_agent = LlmAgent(
    name="FirecrawlAgent",
    model=nebius_model,
    description="Scrapes Nebius Studio homepage using Firecrawl.",
    instruction="""
    Use the firecrawl_scrape_nebius tool to fetch markdown content from Nebius Studio website in proper format. 
    Prefix your response with "**🔥FirecrawlAgent:**"
    **Only use the tools provided to you. Do not call any other functions or tools.**
    """,
    tools=[firecrawl_scrape_nebius],
    output_key="firecrawl_content"
)
print("[Pipeline] FirecrawlAgent setup complete.")

# --- Agent 5: Analysis & Stats ---
print("[Pipeline] Running AnalysisAgent...")
analysis_agent = LlmAgent(  
    name="AnalysisAgent",
    model=LiteLlm(
        model="openai/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1",  # New Nebius model
        api_base=api_base,
        api_key=api_key
    ),
    instruction="""
You are an AI analyst specializing in the latest AI trends and Large Language Models (LLMs).
- ONLY output the final analysis. Do NOT include any 'think', 'chain-of-thought', scratchpad text, or any agent prefix (such as 'AnalysisAgent:') in your response.
- Do NOT include any commentary, meta statements, or process explanations. Only output the analysis itself.
- Do NOT invent or hallucinate model names, features, or recommendations. Only use information explicitly provided in 'final_summary', 'exa_results', 'tavily_results', and 'firecrawl_content'.
- If Nebius model details are not present in 'firecrawl_content', state that no specific Nebius model information is available.
- Present your analysis with clear and concise language, supported by quantifiable data and insights.
- Use markdown tables for statistics, but do not include any agent labels or process explanations.
""",
    description="Analyzes the summary and presents insights and statistics.",
    output_key="analysis_results"
)
print("[Pipeline] AnalysisAgent setup complete.")

# --- Agent 6: Sequential pipeline (Orchestrator Agent) ---
print("[Pipeline] Setting up ParallelAgent for web research...")
parallel_research_agent = ParallelAgent(
     name="ParallelWebResearchAgent",
     sub_agents=[exa_agent, tavily_agent, firecrawl_agent],
     description="Runs multiple research agents in parallel to gather information."
)
print("[Pipeline] ParallelAgent setup complete.")

print("[Pipeline] Setting up SequentialAgent pipeline...")
pipeline = SequentialAgent(
    name="AIPipelineAgent",
    sub_agents=[ parallel_research_agent, summary_agent, analysis_agent]
)
print("[Pipeline] Pipeline setup complete.")

APP_NAME = "ai_analysis_pipeline"
USER_ID = "colab_user"
SESSION_ID = "ai_analysis_session"

session_service = InMemorySessionService()

async def ensure_session():
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

runner = Runner(agent=pipeline, app_name=APP_NAME, session_service=session_service)

def log_event(event):
    print(f"[Pipeline] Event: {event}")

def log_final_response(response):
    print("[Pipeline] Final response from pipeline:")
    print(response)

def run_ai_analysis() -> str:
    print("[Pipeline] Running pipeline...")
    asyncio.run(ensure_session())
    content = types.Content(role="user", parts=[types.Part(text="Start the AI analysis")])
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)
    last_response = None
    for event in events:
        log_event(event)
        if event.is_final_response():
            last_response = event.content.parts[0].text
    if last_response:
        log_final_response(last_response)
        return last_response
    print("[Pipeline] No final response from pipeline.")
    return "No final response from pipeline."

print("[Pipeline] Pipeline module loaded.") 