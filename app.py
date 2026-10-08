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
st.title("⚡ Équipe Multi-Agents (Mistral AI - Mode Optimisé Quotas)")

# --- RÉCUPÉRATION DE LA CLÉ ---
raw_key = st.secrets.get("MISTRAL_API_KEY", os.getenv("MISTRAL_API_KEY", ""))
mistral_key = raw_key.strip().strip('"').strip("'")

if not mistral_key:
    st.error("⚠️ La clé MISTRAL_API_KEY est manquante dans les Secrets Streamlit.")
    st.stop()

# --- INITIALISATION ---
try:
    llm_text = ChatMistralAI(model="mistral-small-latest", api_key=mistral_key, temperature=0)
except Exception as e:
    st.error(f"Erreur d'initialisation : {e}")
    st.stop()

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
        
        with st.status("L'équipe d'agents collabore en tâche de fond...", expanded=True) as status:
            st.write("🤖 **Agents actifs : Superviseur, Analyste/Rédacteur, Expert Data**")
            
            system_prompt = (
                "Tu diriges une équipe de 3 agents IA experts :\n"
                "1. SUPERVISEUR : Définit brièvement le plan d'action.\n"
                "2. RÉDACTEUR & ANALYSTE : Fournit une réponse complète, rigoureuse et bien structurée.\n"
                "3. EXPERT DATA / EXCEL (uniquement si la demande implique un tableau/Excel) : Génère un bloc de code Python avec Pandas créant 'export_resultat.xlsx'.\n\n"
                "Présente clairement chaque étape avec des titres lisibles."
            )
            
            # Boucle de retry avec pause exponentielle en cas de 429
            response_content = None
            max_retries = 3
            wait_time = 12
            
            for attempt in range(max_retries):
                try:
                    res = llm_text.invoke([
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=full_context)
                    ])
                    response_content = res.content
                    break
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429 and attempt < max_retries - 1:
                        st.info(f"⏳ Quota Mistral atteint. Attente de {wait_time}s pour réinitialiser la fenêtre API ({attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        wait_time += 10
                    else:
                        st.error(f"❌ Erreur API Mistral ({e.response.status_code}) : {e.response.text}")
                        st.stop()
                except Exception as e:
                    st.error(f"❌ Erreur : {e}")
                    st.stop()

            if response_content:
                st.markdown(response_content)
                
                # Extraction et exécution du code Python Excel s'il est présent
                if "import pandas" in response_content and "export_resultat.xlsx" in response_content:
                    try:
                        code_blocks = re.findall(r"```python(.*?)```", response_content, re.DOTALL)
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
                        st.error(f"Erreur de création du fichier Excel : {e}")

            status.update(label="Analyse terminée avec succès !", state="complete")
