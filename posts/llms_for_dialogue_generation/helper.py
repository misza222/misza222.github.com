from enum import Enum
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Optional
import os

load_dotenv()


client = instructor.patch(
    #api-keys https://modelstudio.console.alibabacloud.com/ap-southeast-1/#/api-key
    OpenAI(
        # If the environment variable is not set, replace the following line with: api_key="sk-xxx"
        # API keys for the Singapore/Virginia and Beijing regions are different. To get an API key, see https://www.alibabacloud.com/help/en/model-studio/get-api-key
        api_key=os.getenv('QWEN_API_KEY'),
        # The following is the base_url for the Singapore region.
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",  
    ),
    mode=instructor.Mode.MD_JSON
)


def call_api(messages, response_model, model="qwen-flash"):
    response = client.chat.completions.create(
        model=model,  
        messages=messages,
        response_model=response_model,
        temperature=0.0,
        max_retries=3
    )
    
    return response
