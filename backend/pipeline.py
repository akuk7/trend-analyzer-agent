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

# --- Topic Configurations ---
TOPICS = [
    {
        "name": "AI & Machine Learning",
        "emoji": "🤖",
        "exa_query": "Latest AI news OR new LLM models OR AI/Agents advancements",
        "exa_instruction": "Use the exa_search_ai tool to fetch the latest information about AI, new LLMs, and advancements in the field from top sources. Prefix your response with \"**🔥ExaAgent:**\" to clearly identify your output. Provide concise, up-to-date news and trends.",
        "tavily_query": "AI benchmarks OR AI/LLM statistics OR AI providers analysis",
        "tavily_instruction": "Use the tavily_search_ai_analysis tool to retrieve benchmarks, statistics, and relevant analysis on AI. Prefix your response with \"**🐳TavilyAgent:**\" to clearly identify your output. Highlight key statistics and provider comparisons.",
        "tavily_domains": ["technologyreview.com"],
        "firecrawl_url": "https://www.technologyreview.com/topic/artificial-intelligence/",
        "firecrawl_instruction": "Use the firecrawl_scrape_topic tool to fetch markdown content from MIT Technology Review's AI section. Prefix your response with \"**🔥FirecrawlAgent:**\". Only use the tools provided to you."
    },
    {
        "name": "Business",
        "emoji": "💼",
        "exa_query": "Latest business news OR finance OR startups OR markets",
        "exa_instruction": "Use the exa_search_ai tool to fetch the latest business, finance, and market news from top sources. Prefix your response with \"**🔥ExaAgent:**\". Focus on major events, trends, and financial updates.",
        "tavily_query": "Business benchmarks OR finance statistics OR market analysis",
        "tavily_instruction": "Use the tavily_search_ai_analysis tool to retrieve business benchmarks, finance statistics, and market analysis. Prefix your response with \"**🐳TavilyAgent:**\". Highlight key financial indicators and market trends.",
        "tavily_domains": ["bloomberg.com"],
        "firecrawl_url": "https://www.bloomberg.com/",
        "firecrawl_instruction": "Use the firecrawl_scrape_topic tool to fetch markdown content from Bloomberg. Prefix your response with \"**🔥FirecrawlAgent:**\". Only use the tools provided to you."
    },
    {
        "name": "Cinema",
        "emoji": "🎬",
        "exa_query": "Latest cinema news OR movie releases OR box office analysis",
        "exa_instruction": "Use the exa_search_ai tool to fetch the latest cinema news, movie releases, and box office analysis. Prefix your response with \"**🔥ExaAgent:**\". Focus on new releases, reviews, and industry trends.",
        "tavily_query": "Cinema statistics OR movie industry analysis OR box office trends",
        "tavily_instruction": "Use the tavily_search_ai_analysis tool to retrieve cinema statistics, industry analysis, and box office trends. Prefix your response with \"**🐳TavilyAgent:**\". Highlight top-grossing films and industry shifts.",
        "tavily_domains": ["variety.com"],
        "firecrawl_url": "https://variety.com/",
        "firecrawl_instruction": "Use the firecrawl_scrape_topic tool to fetch markdown content from Variety. Prefix your response with \"**🔥FirecrawlAgent:**\". Only use the tools provided to you."
    },
    {
        "name": "Music",
        "emoji": "🎵",
        "exa_query": "Latest music news OR artist updates OR music charts",
        "exa_instruction": "Use the exa_search_ai tool to fetch the latest music news, artist updates, and chart movements. Prefix your response with \"**🔥ExaAgent:**\". Focus on trending artists, releases, and industry news.",
        "tavily_query": "Music industry statistics OR artist analysis OR chart trends",
        "tavily_instruction": "Use the tavily_search_ai_analysis tool to retrieve music industry statistics, artist analysis, and chart trends. Prefix your response with \"**🐳TavilyAgent:**\". Highlight top artists and market shifts.",
        "tavily_domains": ["billboard.com"],
        "firecrawl_url": "https://www.billboard.com/",
        "firecrawl_instruction": "Use the firecrawl_scrape_topic tool to fetch markdown content from Billboard. Prefix your response with \"**🔥FirecrawlAgent:**\". Only use the tools provided to you."
    },
    {
        "name": "Gaming",
        "emoji": "🎮",
        "exa_query": "Latest gaming news OR game releases OR reviews",
        "exa_instruction": "Use the exa_search_ai tool to fetch the latest gaming news, releases, and reviews. Prefix your response with \"**🔥ExaAgent:**\". Focus on new games, industry trends, and major updates.",
        "tavily_query": "Gaming industry statistics OR game analysis OR review trends",
        "tavily_instruction": "Use the tavily_search_ai_analysis tool to retrieve gaming industry statistics, game analysis, and review trends. Prefix your response with \"**🐳TavilyAgent:**\". Highlight top games and market trends.",
        "tavily_domains": ["ign.com"],
        "firecrawl_url": "https://www.ign.com/",
        "firecrawl_instruction": "Use the firecrawl_scrape_topic tool to fetch markdown content from IGN. Prefix your response with \"**🔥FirecrawlAgent:**\". Only use the tools provided to you."
    },
    {
        "name": "Sports",
        "emoji": "⚽",
        "exa_query": "Latest sports news OR match results OR player updates",
        "exa_instruction": "Use the exa_search_ai tool to fetch the latest sports news, match results, and player updates. Prefix your response with \"**🔥ExaAgent:**\". Focus on major events, scores, and athlete news.",
        "tavily_query": "Sports statistics OR match analysis OR league trends",
        "tavily_instruction": "Use the tavily_search_ai_analysis tool to retrieve sports statistics, match analysis, and league trends. Prefix your response with \"**🐳TavilyAgent:**\". Highlight top teams, players, and league standings.",
        "tavily_domains": ["espn.com"],
        "firecrawl_url": "https://www.espn.com/",
        "firecrawl_instruction": "Use the firecrawl_scrape_topic tool to fetch markdown content from ESPN. Prefix your response with \"**🔥FirecrawlAgent:**\". Only use the tools provided to you."
    },
    {
        "name": "Technology & Gadgets",
        "emoji": "💻",
        "exa_query": "Latest technology news OR gadget reviews OR product launches",
        "exa_instruction": "Use the exa_search_ai tool to fetch the latest technology news, gadget reviews, and product launches. Prefix your response with \"**🔥ExaAgent:**\". Focus on new products, reviews, and tech trends.",
        "tavily_query": "Tech industry statistics OR gadget analysis OR product trends",
        "tavily_instruction": "Use the tavily_search_ai_analysis tool to retrieve tech industry statistics, gadget analysis, and product trends. Prefix your response with \"**🐳TavilyAgent:**\". Highlight top gadgets and market shifts.",
        "tavily_domains": ["theverge.com"],
        "firecrawl_url": "https://www.theverge.com/",
        "firecrawl_instruction": "Use the firecrawl_scrape_topic tool to fetch markdown content from The Verge. Prefix your response with \"**🔥FirecrawlAgent:**\". Only use the tools provided to you."
    },
    {
        "name": "Politics & World Events",
        "emoji": "🌍",
        "exa_query": "Latest politics news OR world events OR global analysis",
        "exa_instruction": "Use the exa_search_ai tool to fetch the latest politics news and world events. Prefix your response with \"**🔥ExaAgent:**\". Focus on major global developments and policy changes.",
        "tavily_query": "Political statistics OR world event analysis OR global trends",
        "tavily_instruction": "Use the tavily_search_ai_analysis tool to retrieve political statistics, world event analysis, and global trends. Prefix your response with \"**🐳TavilyAgent:**\". Highlight key events and global impacts.",
        "tavily_domains": ["reuters.com"],
        "firecrawl_url": "https://www.reuters.com/",
        "firecrawl_instruction": "Use the firecrawl_scrape_topic tool to fetch markdown content from Reuters. Prefix your response with \"**🔥FirecrawlAgent:**\". Only use the tools provided to you."
    },
    {
        "name": "Health & Wellness",
        "emoji": "🌿",
        "exa_query": "Latest health news OR wellness tips OR medical research",
        "exa_instruction": "Use the exa_search_ai tool to fetch the latest health news, wellness tips, and medical research. Prefix your response with \"**🔥ExaAgent:**\". Focus on new studies, health trends, and wellness advice.",
        "tavily_query": "Health statistics OR wellness analysis OR medical trends",
        "tavily_instruction": "Use the tavily_search_ai_analysis tool to retrieve health statistics, wellness analysis, and medical trends. Prefix your response with \"**🐳TavilyAgent:**\". Highlight key findings and health recommendations.",
        "tavily_domains": ["healthline.com"],
        "firecrawl_url": "https://www.healthline.com/",
        "firecrawl_instruction": "Use the firecrawl_scrape_topic tool to fetch markdown content from Healthline. Prefix your response with \"**🔥FirecrawlAgent:**\". Only use the tools provided to you."
    },
    {
        "name": "Space Exploration",
        "emoji": "🚀",
        "exa_query": "Latest space news OR missions OR discoveries OR business of space",
        "exa_instruction": "Use the exa_search_ai tool to fetch the latest space news, missions, and discoveries. Prefix your response with \"**🔥ExaAgent:**\". Focus on new missions, discoveries, and industry news.",
        "tavily_query": "Space statistics OR mission analysis OR discovery trends",
        "tavily_instruction": "Use the tavily_search_ai_analysis tool to retrieve space statistics, mission analysis, and discovery trends. Prefix your response with \"**🐳TavilyAgent:**\". Highlight key missions and discoveries.",
        "tavily_domains": ["space.com"],
        "firecrawl_url": "https://www.space.com/",
        "firecrawl_instruction": "Use the firecrawl_scrape_topic tool to fetch markdown content from Space.com. Prefix your response with \"**🔥FirecrawlAgent:**\". Only use the tools provided to you."
    },
]

def get_topic_config(topic_name):
    for t in TOPICS:
        if t["name"] == topic_name:
            return t
    return TOPICS[0]  # Default to AI & Machine Learning


APP_NAME = "ai_analysis_pipeline"
USER_ID = "colab_user"
SESSION_ID = "ai_analysis_session"

session_service = InMemorySessionService()

async def ensure_session():
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)


def log_event(event):
    print(f"[Pipeline] Event: {event}")

def log_final_response(response):
    print("[Pipeline] Final response from pipeline:")
    print(response)

def exa_search_ai(topic_name: str) -> dict:
    topic = get_topic_config(topic_name)
    try:
        results = Exa(api_key=os.getenv("EXA_API_KEY")).search_and_contents(
            query=topic["exa_query"],
            include_domains=topic.get("exa_domains", []),
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

def tavily_search_ai_analysis(topic_name: str) -> dict:
    topic = get_topic_config(topic_name)
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search(
            query=topic["tavily_query"],
            search_depth="advanced",
            time_range="week",
            include_domains=topic["tavily_domains"]
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

def firecrawl_scrape_topic(topic_name: str) -> dict:
    topic = get_topic_config(topic_name)
    firecrawl = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
    try:
        scrape_result = firecrawl.scrape_url(
            url=topic["firecrawl_url"],
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

async def run_ai_analysis(topic_name="AI & Machine Learning") -> str:
    topic = get_topic_config(topic_name)
    await ensure_session()

    exa_agent = LlmAgent(
        name="ExaAgent",
        model=nebius_model,
        description="Fetches latest news...",
        instruction=topic["exa_instruction"],
        tools=[exa_search_ai],
        output_key="exa_results"
    )
    tavily_agent = LlmAgent(
        name="TavilyAgent",
        model=nebius_model,
        description="Fetches stats...",
        instruction=topic["tavily_instruction"],
        tools=[tavily_search_ai_analysis],
        output_key="tavily_results"
    )
    firecrawl_agent = LlmAgent(
        name="FirecrawlAgent",
        model=nebius_model,
        description="Scrapes...",
        instruction=topic["firecrawl_instruction"],
        tools=[firecrawl_scrape_topic],
        output_key="firecrawl_content"
    )
    summary_agent = LlmAgent(
        name="SummaryAgent",
        model=nebius_model,
        description="Summarizes and formats Exa and Tavily results.",
        instruction="""
You are a summarizer and formatter.
- Combine the information from 'exa_results' (latest updates) and 'tavily_results' (benchmarks and analysis).
- Present a structured summary, highlighting key trends, new developments, and relevant statistics.
- Use markdown formatting for clarity and readability.
- Use emojis like 🚀 for new launches, 📊 for statistics, and 📈 for trends to make the summary more engaging.
- Structure information using bullet points and headings for better organization.
- Prefix your response with \"**🍥SummaryAgent:**\" to clearly identify your output.
- **Only use the tools provided to you. Do not call any other functions or tools.**
""",
        tools=[],
        output_key="final_summary"
    )
    analysis_agent = LlmAgent(
        name="AnalysisAgent",
        model=LiteLlm(
            model="openai/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1",
            api_base=api_base,
            api_key=api_key
        ),
        instruction="""
You are an AI analyst specializing in the latest trends and statistics.
- ONLY output the final analysis. Do NOT include any 'think', 'chain-of-thought', scratchpad text, or any agent prefix (such as 'AnalysisAgent:') in your response.
- Do NOT include any commentary, meta statements, or process explanations. Only output the analysis itself.
- Do NOT invent or hallucinate names, features, or recommendations. Only use information explicitly provided in 'final_summary', 'exa_results', 'tavily_results', and 'firecrawl_content'.
- If details are not present in 'firecrawl_content', state that no specific information is available.
- Present your analysis with clear and concise language, supported by quantifiable data and insights.
- Use markdown tables for statistics, but do not include any agent labels or process explanations.
""",
        description="Analyzes the summary and presents insights and statistics.",
        output_key="analysis_results"
    )

    parallel_research_agent = ParallelAgent(
        name="ParallelWebResearchAgent",
        sub_agents=[exa_agent, tavily_agent, firecrawl_agent],
        description="Runs multiple research agents in parallel to gather information."
    )
    pipeline = SequentialAgent(
        name="AIPipelineAgent",
        sub_agents=[parallel_research_agent, summary_agent, analysis_agent]
    )
    runner = Runner(agent=pipeline, app_name=APP_NAME, session_service=session_service)
    content = types.Content(role="user", parts=[types.Part(text=topic_name)])
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