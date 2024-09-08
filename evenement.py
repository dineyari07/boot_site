import json
import warnings

from minio import Minio

warnings.filterwarnings("ignore")
from crewai import Agent, Task, Crew
from crewai_tools import BaseTool, SerperDevTool
from langchain_groq import ChatGroq
import os


os.environ["GROQ_API_KEY"] = "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"
os.environ["SERPER_API_KEY"] = "36d236523f043f3f653f3544471dc42883a5785e"


def load_json_data(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


json_data = load_json_data("evenements_touristiques.json")


serper_tool = SerperDevTool(
    country="bj",
    locale="fr",
    location="Benin, Afrique, Africa",
    n_results=100,
)


class SaveJSONTool(BaseTool):
    name: str = "SaveJSONTool"
    description: str = "Tool to save provided data into a JSON file."
    filename: str = "processed_data.json"

    def _run(self, data: dict) -> str:
        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return f"JSON saved as {self.filename}"


llm_groq = ChatGroq(
    temperature=0,
    groq_api_key="",
    model_name="llama-3.1-8b-instant"
)


agent = Agent(
    role="Tourist Event Research Specialist",
    goal="Chercher et Trouver les temoignages, les commentaires, et les liens des sites web sur les evenements touristiques du Benin",
    backstory="Tu es un expert en recherche d'informations sur les evenements du Bénin. Ton rôle est d'identifier les temoignages, les commentaires en plus des liens du site web sur les evenements touristiques du Bénin.",
    verbose=True,
    llm=llm_groq,
    tools=[serper_tool, SaveJSONTool(result_as_answer=True)],
)


task = Task(
    description=(
        "Lire les données en entrée : {input_data} sur lequel les evenements touristiques sont stockées. "
        "Recuperer les informations sur chaque evenement touristique du Benin et "
        "Collecter les temoignages et les commentaires des participants à ces evenements. "
        "Pour chaque evenement, trouver les temoignages, les commentaires et les liens du site web où les temoignages ont été collectés."
    ),
    expected_output="Un objet JSON",
    agent=agent,
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True,
)


result = crew.kickoff(inputs={"input_data": json_data})
# print(result)


class PythonMinIOUtils:
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool,
        bucket_name: str,
    ):
        # Initialize MinIO client
        self.minio_client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,  # Set to False for insecure connections (http instead of https)
        )

        # Define the bucket name
        self.bucket_name = bucket_name

    def upload_file_to_bucket(self, source_file_path: str, destination_file_name: str):
        found = self.minio_client.bucket_exists(self.bucket_name)

        if not found:
            self.minio_client.make_bucket(self.bucket_name)
            print("Created bucket:", self.bucket_name)
        else:
            print("Bucket", self.bucket_name, "already exists")

        self.minio_client.fput_object(
            self.bucket_name,
            destination_file_name,
            source_file_path,
        )
        print(
            source_file_path,
            "successfully uploaded as object",
            destination_file_name,
            "to bucket",
            self.bucket_name,
        )


utils = PythonMinIOUtils(
    endpoint="play.min.io",
    access_key="Q3AM3UQ867SPQQA43P2F",
    secret_key="zuf+tfteSlswRu7BJ86wekitnifILbZam1KYY3TG",
    secure=True,
    bucket_name="test",
)

utils.upload_file_to_bucket(
    "processed_data.json", "group_10_task_6_processed_data.json"
)
