import google.generativeai as genai
import os
from dotenv import load_dotenv

def load_api_model(model_name: str):
    load_dotenv()
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    model = genai.GenerativeModel(model_name)
    return model