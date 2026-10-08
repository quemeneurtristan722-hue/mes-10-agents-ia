import os
import re
import time
import httpx
import streamlit as st
import pandas as pd
from pypdf import PdfReader

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI

# --- CONFIGURATION INTERFACE STREAMLIT ---
st.set_page_config(page_title="Mes Agents IA", page_icon="⚡", layout="wide")
st.title("⚡ Équipe Multi-Agents (Mistral AI)")

# --- RÉCUPÉRATION ET NETTOYAGE DE LA CLÉ ---
raw_key = st.secrets.get("MISTRAL_API_KEY", os.getenv("MISTRAL_API_KEY", ""))
mistral_key = raw_key.strip().strip('"').strip("'")

if not mistral_key:
    st.error("⚠️ La clé MISTRAL_API_KEY est manquante dans les Secrets Streamlit.")
    st.stop()

# --- INITIALISATION ---
try:
    llm_text = ChatMistralAI(model="mistral-small-latest", api_key=mistral_key, temperature=0)
    llm_code = ChatMistralAI(model="codestral-latest", api_key=mistral_key, temperature=0)
except Exception as e:
    st.error(f"Erreur d'initialisation : {e}")
    st.stop()

# --- FONCTION D'APPEL TEMPORISÉ ET SÉCURISÉ ---
def call_agent_with_tempo(llm, messages, agent_name: str, pause_seconds=4):
    """Fait une pause préventive avant de solliciter l'agent pour respecter les quotas."""
    st.write(f"⏳ **{agent_name}** prépare sa réponse (pause de sécurité de {pause_seconds}s)...")
    time.sleep(pause_seconds)
    
    try:
        return llm.invoke(messages)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            st.warning("⚠️ Quota temporairement dépassé. Seconde pause de sécurité (6s)...")
            time.sleep(6)
            return llm.invoke(messages)
        raise e

# --- EXTRACTION DES FICHIERS ---
def extract_file_content(file_obj) -> str:
    if file_obj is None:
        return ""
    ext = os.path.splitext(file_obj.name)[1].lower()
    try:
        if ext in [".txt", ".py", ".js", ".json", ".csv", ".md"]:
            return f"\n\n--- FICHIER JOINT ({file_obj.name}) ---\n" + file_obj.read().decode("utf-8", errors="ignore")[:6000]
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_obj)
            return f"\n\n--- DONNÉES EXCEL ({file_obj.name}) ---\n" + df.to_string()[:6000]
        elif ext == ".pdf":
            reader = PdfReader(file_obj)
            text = "".join([page.extract_text() or "" for page in reader.pages[:10]])
            return f"\n\n--- EXTRAIT PDF ({file_obj.name}) ---\n" + text[:6000]
    except Exception as e:
        return f"\n[Erreur de lecture : {e}]"
    return f"\n[Fichier joint : {file_obj.name}]"

# --- INTERFACE UTILISATEUR ---
user_prompt = st.text_area("Saisis ta demande :", placeholder="Ex : Analyse ces données, fais une synthèse et crée un tableau Excel...")
uploaded_file = st.file_uploader("Joindre un fichier (Optionnel)", type=["pdf", "xlsx", "csv", "txt", "py", "md"])

if st.button("🚀 Lancer l'équipe d'agents", type="primary"):
    if not user_prompt and not uploaded_file:
        st.warning("Merci de saisir un texte ou de joindre un fichier.")
    else:
        full_context = user_prompt + extract_file_content(uploaded_file)
        
        with st.status("Collaboration des agents en cours...", expanded=True) as status:
            # 1. Superviseur
            try:
                plan_res = call_agent_with_tempo(
                    llm_text,
                    [
                        SystemMessage(content="Tu es le Superviseur. Définis rapidement la stratégie pour répondre au mieux à la demande."),
                        HumanMessage(content=full_context)
                    ],
                    agent_name="Superviseur",
                    pause_seconds=1 # Première requête immédiate
                )
                plan = plan_res.content
                st.markdown(f"**Plan d'action :**\n{plan}")
                st.divider()
            except Exception as e:
                status.update(label="Erreur d'exécution", state="error")
                st.error(f"❌ Erreur du Superviseur : {e}")
                st.stop()

            # 2. Agent Réalisateur / Analyste
            try:
                response_res = call_agent_with_tempo(
                    llm_text,
                    [
                        SystemMessage(content="Tu es l'Expert Principal. Réponds à la demande de manière complète et bien structurée en français."),
                        HumanMessage(content=f"Demande initiale : {full_context}\n\nPlan à suivre : {plan}")
                    ],
                    agent_name="Expert Rédacteur",
                    pause_seconds=4 # Pause préventive de 4s
                )
                response = response_res.content
                st.markdown(response)
                st.divider()
            except Exception as e:
                status.update(label="Erreur d'exécution", state="error")
                st.error(f"❌ Erreur du Rédacteur : {e}")
                st.stop()

            # 3. Agent Codeur / Data (si besoin d'Excel)
            if "excel" in user_prompt.lower() or "tableau" in user_prompt.lower():
                try:
                    code_res = call_agent_with_tempo(
                        llm_code,
                        [
                            SystemMessage(content="Génère du code Python exécutable avec Pandas pour créer un DataFrame et le sauvegarder sous le nom 'export_resultat.xlsx'."),
                            HumanMessage(content=response)
                        ],
                        agent_name="Expert Data / Excel",
                        pause_seconds=4 # Pause préventive de 4s
                    )
                    
                    code_blocks = re.findall(r"```python(.*?)```", code_res.content, re.DOTALL)
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
                    st.error(f"Erreur de création Excel : {e}")

            status.update(label="Analyse terminée avec succès !", state="complete")
