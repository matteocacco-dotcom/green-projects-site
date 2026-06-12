import os
import json
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

llm = ChatGroq(model="llama3-70b-8192", temperature=0.2)
tools = [TavilySearchResults(max_results=5)]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that searches the web for information."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

user_prompt = """
Search the web for an innovative project, eco startup, or recent initiative (from the last year) related to environmental protection or green tech.
Find the details and return the response STRICTLY in JSON format with this structure:
{
    "titolo": "Project name",
    "categoria": "e.g. Reforestation, Renewable Energy, Recycling, Wildlife Protection",
    "descrizione": "A clear summary of what the project does and why it matters (maximum 4 lines).",
    "link": "The URL of the official website or news article about the project"
}
Return exclusively valid JSON code. Do not add greetings or plain text explanations.
"""

print("AI is searching for a green project on the web...")
try:
    result = agent_executor.invoke({"input": user_prompt})
    risposta = result["output"]

    if "```json" in risposta:
        risposta = risposta.split("```json")[1].split("```")[0].strip()
    elif "```" in risposta:
        risposta = risposta.split("```")[1].split("```")[0].strip()

    dati_progetto = json.loads(risposta)
    supabase.table("notizie").insert(dati_progetto).execute()
    print("Database updated successfully!")

except Exception as e:
    print(f"Error during process: {e}")
