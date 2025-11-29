from dotenv import load_dotenv
from pathlib import Path
import os
from custom_lib import console
# load environment variables
env_path = "./custom_lib/dotenv/.env"
load_dotenv(dotenv_path=env_path)
def is_openai_env_set():
    if not 'OPENAI_API_KEY' in os.environ:
        console.log(f"key: OPENAI_API_KEY has not been set. Code will exit")
        os._exit(1)