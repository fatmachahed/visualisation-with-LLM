import os
from dotenv import load_dotenv

# Charge le fichier .env
load_dotenv()

# Clés API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
