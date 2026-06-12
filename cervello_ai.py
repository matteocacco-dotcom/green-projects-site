import os
import json
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import initialize_agent, AgentType
from supabase import create_client, Client

# Legge le credenziali dalle variabili d'ambiente (impostate da GitHub Actions)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Configura L'LLM Gratis (Llama 3 su Groq) e lo strumento di ricerca online
llm = ChatGroq(model="llama3-70b-8192", temperature=0.2)
search_tool = TavilySearchResults(max_results=5)

# Inizializza l'agente
agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Comando per l'AI
prompt = """
Cerca sul web un progetto innovativo, una startup ecologica o un'iniziativa recente (dell'ultimo anno) legata alla salvaguardia dell'ambiente o alla green tech.
Trova i dettagli e restituisci la risposta TASSATIVAMENTE in formato JSON con questa struttura:
{
    "titolo": "Nome del progetto",
    "categoria": "es. Riforestazione, Energia Rinnovabile, Riciclo, Tutela Animali",
    "descrizione": "Un riassunto chiaro di cosa fa il progetto e perché è importante (massimo 4 righe).",
    "link": "L'URL del sito ufficiale o della notizia del progetto"
}
Restituisci esclusivamente il codice JSON valido. Non aggiungere saluti o spiegazioni in testo semplice.
"""

print("L'AI Gratis sta cercando un progetto green sul web...")
try:
    risposta_agente = agent.run(prompt)
    
    # Pulizia dell'output nel caso il modello aggiunga del testo inutile prima del JSON
    if "```json" in risposta_agente:
        risposta_agente = risposta_agente.split("```json")[1].split("```")[0].strip()
    elif "```" in risposta_agente:
        risposta_agente = risposta_agente.split("```")[1].split("```")[0].strip()

    # Carica i dati su Supabase
    dati_progetto = json.loads(risposta_agente)
    supabase.table("notizie").insert(dati_progetto).execute()
    print("Database aggiornato con successo a costo zero!")
    
except Exception as e:
    print(f"Errore durante il processo: {e}")
