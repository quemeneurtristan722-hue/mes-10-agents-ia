import os
import re
import streamlit as st
import pandas as pd
from pypdf import PdfReader

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI

# --- CONFIGURATION STREAMLIT ---
st.set_page_config(page_title="Mes 10 Agents IA", page_icon="⚡", layout="wide")
st.title("⚡ Équipe Multi-Agents (Version Stabilisée)")

# --- RÉCUPÉRATION ET VERIFICATION DES CLÉS ---
google_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
mistral_key = st.secrets.get("MISTRAL_API_KEY", os.getenv("MISTRAL_API_KEY"))

with st.sidebar:
    st.header("🔑 État des connexions API")
    st.write("Google AI Studio :", "🟢 OK" if google_key else "🔴 Manquante")
    st.write("Mistral AI :", "🟢 OK" if mistral_key else "🔴 Manquante")

if not google_key or not mistral_key:
    st.error("Configure tes clés GOOGLE_API_KEY et MISTRAL_API_KEY dans Secrets (Manage app -> Settings -> Secrets).")
    st.stop()

# --- INITIALISATION DES MODÈLES D'ÉQUIPE ---
try:
    gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=google_key, temperature=0)
    mistral_text_llm = ChatMistralAI(model="mistral-small-latest", api_key=mistral_key, temperature=0)
    mistral_code_llm = ChatMistralAI(model="codestral-latest", api_key=mistral_key, temperature=0)
except Exception as e:
    st.error(f"Erreur d'initialisation des modèles : {e}")
    st.stop()

# --- EXTRACTION DE FICHIERS ---
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

# --- DÉFINITION DES ROLES D'AGENTS ---
AGENTS_ROLES = {
    "1. Superviseur": (gemini_llm, "Tu es le Superviseur. Analyse la demande globale et planifie le travail."),
    "2. Chercheur": (mistral_text_llm, "Tu es le Chercheur. Réponds avec tes connaissances actualisées de manière factuelle."),
    "3. Analyste": (gemini_llm, "Tu es l'Analyste Logique. Traite le problème étape par étape."),
    "4. Codeur": (mistral_code_llm, "Tu es le Codeur Senior. Écris du code propre et documenté."),
    "5. DataExcel": (mistral_text_llm, "Tu es l'Expert Data. Si un tableau est demandé, génère du code Python Pandas créant 'export_resultat.xlsx'."),
    "6. Rédacteur": (mistral_text_llm, "Tu es le Rédacteur. Soigne la structure et le style du texte en français."),
    "7. Critic": (gemini_llm, "Tu es le Relecteur Qualité. Détecte les erreurs et valide le travail."),
    "8. Traducteur": (mistral_text_llm, "Tu es le Traducteur Expert multilingue."),
    "9. Vision": (gemini_llm, "Tu es l'Expert Vision et Analyse documentaire."),
    "10. Formateur": (mistral_text_llm, "Tu es le Formateur Pédagogue. Explique simplement les concepts.")
}

# --- INTERFACE UTILISATEUR ---
user_prompt = st.text_area("Saisis ta demande :", placeholder="Exemple : Analyse ces données et rédige une synthèse...")
uploaded_file = st.file_uploader("Joindre un fichier (Optionnel)", type=["pdf", "xlsx", "csv", "txt", "py", "md"])

if st.button("🚀 Lancer l'analyse", type="primary"):
    if not user_prompt and not uploaded_file:
        st.warning("Merci de saisir un texte ou de déposer un fichier.")
    else:
        full_context = user_prompt + extract_file_content(uploaded_file)
        
        # Étape 1 : Analyse par le Superviseur
        with st.status("Traitement par l'équipe d'agents...", expanded=True) as status:
            st.write("🤖 **Analyse du Superviseur...**")
            try:
                sup_response = gemini_llm.invoke([
                    SystemMessage(content="Tu es le Superviseur. Choisis les 2 agents les plus pertinents parmi : Chercheur, Analyste, Codeur, DataExcel, Redacteur, Traducteur, Formateur."),
                    HumanMessage(content=full_context)
                ]).content
                st.markdown(f"**Plan du Superviseur :**\n{sup_response}")
                st.divider()
            except Exception as e:
                st.error(f"Erreur du Superviseur : {e}")
                st.stop()
            
            # Étape 2 : Traitement principal par le Rédacteur/Analyste
            st.write("🤖 **Exécution de la tâche...**")
            try:
                main_response = mistral_text_llm.invoke([
                    SystemMessage(content="Tu es l'agent principal. Traite la demande de manière complète et détaillée."),
                    HumanMessage(content=full_context)
                ]).content
                st.markdown(main_response)
            except Exception as e:
                st.error(f"Erreur d'exécution : {e}")
                st.stop()
                
            status.update(label="Analyse terminée !", state="complete")
            
            # Génération d'un fichier Excel si du code Pandas est détecté
            if "import pandas" in main_response and "export_resultat.xlsx" in main_response:
                try:
                    code_blocks = re.findall(r"```python(.*?)```", main_response, re.DOTALL)
                    if code_blocks:
                        exec(code_blocks[-1], globals())
                        if os.path.exists("export_resultat.xlsx"):
                            with open("export_resultat.xlsx", "rb") as f:
                                st.download_button("📥 Télécharger le fichier Excel", f, file_name="export_resultat.xlsx")
                except Exception as e:
                    st.error(f"Erreur d'export Excel : {e}")
