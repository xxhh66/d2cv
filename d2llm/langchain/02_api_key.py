import os
from dotenv import load_dotenv
import langchain

load_dotenv()

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
print(f'deepseek_api_key:{deepseek_api_key}')

