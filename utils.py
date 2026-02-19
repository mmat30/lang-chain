import os
from dotenv import load_dotenv


def load_env():
    load_dotenv()

    # TODO: check if the following order
    # 0) based on the env var set ["OPENAI, AWS, OLLAMA"]
    # 1) grab the env and set the provider accordingally
    # 2) return a base provider that extends the base provider class
    # 3) then return the provider for use

def get_bedrock_api_key():
    return os.environ.get("BEDROCK_API_KEY")