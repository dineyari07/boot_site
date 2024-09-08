import warnings
from fileinput import filename
warnings.filterwarnings("ignore")
import json
from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool,SerperDevTool
from  langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
import os
import pandas as pd  
from typing import Dict


api_key = os.environ.get("GROQ_API_KEY")

#llm = ChatOllama(model="llama3.1",base_url="http://localhost:11434")
llm_groq = ChatGroq(temperature=0,groq_api_key=api_key,model_name="llama-3.1-70b-versatile")

tool = SerperDevTool(
    country="bj",                 # Code ISO du Bénin
    locale="fr",                  # Langue principale utilisée au Bénin
    location="Bénin, afrique, africa",  # Localisation spécifique 
    n_results=100,                 # Augmenter le nombre de résultats pour obtenir plus d'informations
)

class SaveJSONTool(BaseTool):
    name: str = "SaveJSONTool"
    description: str = "Tool to save provided data into a JSON file."

    def _run(self, data: Dict, filename: str = "data_sites_touristiques.json") -> str:
        # Convertir les données en DataFrame (si les données sont déjà un dictionnaire, pas besoin de les convertir en liste)
        if "sites_touristiques" in data:
            df = pd.DataFrame(data["sites_touristiques"])
            # Sauvegarder les données en format JSON
            df.to_json(filename, orient='records', lines=True)
            return f"JSON saved as {filename}"
        else:
            raise ValueError("Les données ne contiennent pas la clé 'sites_touristiques'.")


# Collect agent
agent_collect = Agent(
    role="Tourism Research Specialist",
    goal="Collecter des informations complètes sur les sites touristiques du Bénin, y compris des descriptions textuelles, des liens vers des vidéos, et des images, puis les organiser sous forme de JSON.",
    backstory="En tant qu'expert en recherche touristique, ta mission est d'explorer diverses sources pour recueillir des informations détaillées sur les sites touristiques du Bénin. Tu devras compiler des descriptions, des vidéos et des images, et organiser toutes ces informations dans un format JSON structuré.",
    verbose=True,
    llm=llm_groq,
    tools = [tool, SaveJSONTool(result_as_answer=True)]
)

task = Task(
    description=(
       "Effectue une recherche approfondie sur les sites touristiques du Bénin. "
        "Les informations recherchées doivent inclure des descriptions textuelles, des liens vers des vidéos pertinentes, et des images représentatives. "
        "Organise ces informations dans un objet JSON structuré, où chaque site est détaillé avec son nom, une description, l'adresse, et les liens multimédias."
    ),
    expected_output=(
        "Un objet JSON avec la clé 'sites_touristiques'"
    ),
    agent=agent_collect
)

# # Validate agent
# agent_manage= Agent(
#     role="Information Validation Specialist",
#     goal="Valider la pertinence et l'exactitude des informations collectées sur les sites touristiques du Bénin.",
#     backstory="Tu es chargé de vérifier les informations obtenues par l'agent de recherche pour les sites touristiques du Bénin. Ta tâche est de confirmer leur exactitude, leur pertinence, et leur conformité aux exigences, en utilisant les outils appropriés pour garantir des données fiables et précises.",
#     verbose=True,
#     allow_delegation=True,
#     llm=llm_groq
# )

# manage_task = Task(
#     description=(
#        "Valider la pertinence et l'exactitude des informations collectées sur les sites touristiques du Bénin."
#        "Renvoyer le json contenant seulement les sites touristiques validés"
#     ),
#     expected_output=(
#         "Un objet JSON avec la clé 'sites_touristiques'"
#     ),
#     agent=agent_manage
# )





crew = Crew(
    agents= [agent_collect],
    tasks= [task],
    verbose=True,
)


result = crew.kickoff(inputs={"search_query":"Tous les sites touristiques du Bénin"})


print(result)