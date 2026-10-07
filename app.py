import os
import re
import operator
from typing import TypedDict, Annotated, Literal
import streamlit as st
import pandas as pd
from pypdf import PdfReader
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START, END

# --- CONFIGURATION INTERFACE STREAMLIT ---
st.set_page_config(page_title="Mes 10 Agents IA", page_icon="⚡", layout="wide")
st.title("⚡ Équipe Multi-Agents à 10 IA")

# --- RÉCUPÉRATION DES CLÉS DEPUIS STREAMLIT SECRETS ---
google_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
groq_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
mistral_key = st.secrets.get("MISTRAL_API_KEY", os.getenv("MISTRAL_API_KEY"))

if not google_key or not groq_key or not mistral_key:
    st.error("⚠️ Clés d'API manquantes dans les Secrets de Streamlit.")
    st.stop()

# --- INITIALISATION DES MODÈLES D'IA ---
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=google_key, temperature=0)
llama_heavy = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_key, temperature=0)
llama_fast = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_key, temperature=0)
deepseek_model = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key=groq_key, temperature=0)
mistral_coder = ChatMistralAI(model="codestral-latest", api_key=mistral_key, temperature=0)
mistral_text = ChatMistralAI(model="mistral-small-latest", api_key=mistral_key, temperature=0)

search_tool = DuckDuckGoSearchRun()

# --- EXTRACTION DU CONTENU DES FICHIERS ---
def extract_file_content(file_obj) -> str:
    if file_obj is None:
        return ""
    ext = os.path.splitext(file_obj.name)[1].lower()
    try:
        if ext in [".txt", ".py", ".js", ".json", ".csv", ".md"]:
            return f"\n\n--- FICHIER JOINT ({file_obj.name}) ---\n" + file_obj.read().decode("utf-8", errors="ignore")[:6000]
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_obj)
            return f"\n\n--- FICHIER EXCEL ({file_obj.name}) ---\n" + df.to_string()[:6000]
        elif ext == ".pdf":
            reader = PdfReader(file_obj)
            text = "".join([page.extract_text() or "" for page in reader.pages[:10]])
            return f"\n\n--- FICHIER PDF ({file_obj.name}) ---\n" + text[:6000]
    except Exception as e:
        return f"\n[Erreur de lecture du fichier : {e}]"
    return f"\n[Fichier joint : {file_obj.name}]"

# --- ÉTAT DU WORKFLOW LANGGRAPH ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    next_step: str

class Router(BaseModel):
    next_agent: Literal[
        "Chercheur", "Analyste", "Codeur", "DataExcel", 
        "Redacteur", "Critic", "Traducteur", "Vision", "Formateur", "FINISH"
    ] = Field(description="Nom du prochain agent ou FINISH.")

# --- DÉFINITION DES 10 AGENTS ---
def supervisor_node(state: AgentState):
    prompt = SystemMessage(content=(
        "Tu es le Superviseur d'une équipe de 9 agents experts.\n"
        "Analyse la demande et attribue la main à l'agent le plus qualifié :\n"
        "- 'Chercheur' : recherche web / actualités fraîches.\n"
        "- 'Analyste' : logique pure, maths, raisonnement complexe.\n"
        "- 'Codeur' : écriture/débogage de code informatique.\n"
        "- 'DataExcel' : analyse de données, mise en forme de tableaux, exports Excel.\n"
        "- 'Redacteur' : rédaction de textes, e-mails, rapports impeccables.\n"
        "- 'Critic' : contrôle qualité des réponses fournies.\n"
        "- 'Traducteur' : traduction et adaptation linguistique.\n"
        "- 'Vision' : analyse de fichiers/visuels.\n"
        "- 'Formateur' : explications pédagogiques simples.\n"
        "- 'FINISH' : quand la réponse globale est complète et satisfaisante."
    ))
    planner = llama_heavy.with_structured_output(Router)
    decision = planner.invoke([prompt] + state["messages"])
    return {"next_step": decision.next_agent}

def researcher_node(state: AgentState):
    query = state["messages"][0].content
    try:
        web_res = search_tool.run(query)
    except Exception as e:
        web_res = f"Recherche indisponible : {e}"
    res = llama_heavy.invoke([SystemMessage(content=f"Tu es le Chercheur Web. Résultats web actuels :\n{web_res}\nSynthétise l'essentiel.")] + state["messages"])
    return {"messages": [res]}

def analyst_node(state: AgentState):
    res = deepseek_model.invoke([SystemMessage(content="Tu es l'Analyste Logique expert. Résous le problème étape par étape.")] + state["messages"])
    return {"messages": [res]}

def coder_node(state: AgentState):
    res = mistral_coder.invoke([SystemMessage(content="Tu es le Codeur Senior. Fournis du code fonctionnel et documenté.")] + state["messages"])
    return {"messages": [res]}

def data_excel_node(state: AgentState):
    prompt = SystemMessage(content=(
        "Tu es l'Expert Data et Excel.\n"
        "Si un fichier Excel est demandé, génère obligatoirement un bloc de code Python (avec Pandas) "
        "qui crée et enregistre le fichier sous le nom 'export_resultat.xlsx'."
    ))
    res = llama_heavy.invoke([prompt] + state["messages"])
    return {"messages": [res]}

def writer_node(state: AgentState):
    res = mistral_text.invoke([SystemMessage(content="Tu es le Rédacteur. Soigne parfaitement la langue et la structure.")] + state["messages"])
    return {"messages": [res]}

def critic_node(state: AgentState):
    res = llama_heavy.invoke([SystemMessage(content="Tu es le Relecteur Qualité. Analyse et valide la réponse finale.")] + state["messages"])
    return {"messages": [res]}

def translator_node(state: AgentState):
    res = llama_fast.invoke([SystemMessage(content="Tu es le Traducteur Expert.")] + state["messages"])
    return {"messages": [res]}

def vision_node(state: AgentState):
    res = gemini_model.invoke([SystemMessage(content="Tu es l'Expert Vision et Documentaire.")] + state["messages"])
    return {"messages": [res]}

def teacher_node(state: AgentState):
    res = llama_heavy.invoke([SystemMessage(content="Tu es le Formateur. Explique clairement avec des mots simples.")] + state["messages"])
    return {"messages": [res]}

# --- ASSEMBLAGE DU GRAPHE LANGGRAPH ---
workflow = StateGraph(AgentState)

workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Chercheur", researcher_node)
workflow.add_node("Analyste", analyst_node)
workflow.add_node("Codeur", coder_node)
workflow.add_node("DataExcel", data_excel_node)
workflow.add_node("Redacteur", writer_node)
workflow.add_node("Critic", critic_node)
workflow.add_node("Traducteur", translator_node)
workflow.add_node("Vision", vision_node)
workflow.add_node("Formateur", teacher_node)

workflow.add_edge(START, "Supervisor")
workflow.add_conditional_edges("Supervisor", lambda s: s["next_step"], {
    "Chercheur": "Chercheur", "Analyste": "Analyste", "Codeur": "Codeur",
    "DataExcel": "DataExcel", "Redacteur": "Redacteur", "Critic": "Critic",
    "Traducteur": "Traducteur", "Vision": "Vision", "Formateur": "Formateur",
    "FINISH": END
})

for node in ["Chercheur", "Analyste", "Codeur", "DataExcel", "Redacteur", "Critic", "Traducteur", "Vision", "Formateur"]:
    workflow.add_edge(node, "Supervisor")

app_agents = workflow.compile()

# --- INTERFACE UTILISATEUR ---
user_prompt = st.text_area("Pose ta question ou décris ta demande :", placeholder="Ex : Recherche les derniers équipements réseaux, fais une analyse et crée un tableau comparatif...")
uploaded_file = st.file_uploader("Joindre un fichier (PDF, Excel, TXT, CSV) - Optionnel", type=["pdf", "xlsx", "csv", "txt", "py", "md"])

if st.button("🚀 Lancer les agents", type="primary"):
    if not user_prompt and not uploaded_file:
        st.warning("Merci de saisir un texte ou de joindre un fichier.")
    else:
        full_prompt = user_prompt + extract_file_content(uploaded_file)
        inputs = {"messages": [HumanMessage(content=full_prompt)]}
        
        with st.spinner("Les agents s'activent et collaborent..."):
            for event in app_agents.stream(inputs):
                for node_name, data in event.items():
                    if "messages" in data and data["messages"]:
                        last_msg = data["messages"][-1].content
                        st.markdown(f"### 🤖 Agent [{node_name}]")
                        st.markdown(last_msg)
                        st.divider()
                        
                        if "import pandas" in last_msg and "export_resultat.xlsx" in last_msg:
                            try:
                                code_blocks = re.findall(r"```python(.*?)```", last_msg, re.DOTALL)
                                if code_blocks:
                                    exec(code_blocks[-1], globals())
                                    if os.path.exists("export_resultat.xlsx"):
                                        with open("export_resultat.xlsx", "rb") as f:
                                            st.download_button(
                                                "📥 Télécharger le fichier Excel généré", 
                                                f, 
                                                file_name="export_resultat.xlsx", 
                                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                            )
                            except Exception as e:
                                st.error(f"Erreur lors de la génération du fichier Excel : {e}")
