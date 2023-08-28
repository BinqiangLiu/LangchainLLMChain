import streamlit as st
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.memory import StreamlitChatMessageHistory
import openai
from streamlit_chat import message
import numpy as np
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from streamlit.components.v1 import html
from langchain import HuggingFaceHub

import os
from dotenv import load_dotenv
load_dotenv()

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
HUGGINGFACEHUB_API_TOKEN = os.environ.get('HUGGINGFACEHUB_API_TOKEN')
repo_id="HuggingFaceH4/starchat-beta"
llm = HuggingFaceHub(repo_id=repo_id,
                     model_kwargs={"min_length":100,
                                   "max_new_tokens":1024, "do_sample":True,
                                   "temperature":0.1,
                                   "top_k":50,
                                   "top_p":0.95, "eos_token_id":49155})

user_query= input("Enter your query:")
prompt_template = "{user_query}"

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)

print("------Your entry:"+user_query+"------")
print()

response_1=llm_chain(user_query)
print(response_1)

response=llm_chain.run(user_query)
print(response)

temp_ai_response=response
final_ai_response=temp_ai_response.partition('<|end|>\n<|user|>\n')[0]
print(final_ai_response)

i_final_ai_response=final_ai_response.replace('<|end|>\n<|assistant|>\n', '') 
ii_final_ai_response=i_final_ai_response.replace('<|end|>\n<|system|>\n', '') 
print(i_final_ai_response)
print(ii_final_ai_response)

i_unique_responses = set(i_final_ai_response.split('\n'))
output_i_unique_responses = sorted(list(i_unique_responses), key=response.index)

ii_unique_responses = set(ii_final_ai_response.split('\n'))
output_ii_unique_responses = sorted(list(ii_unique_responses), key=response.index)

for item in output_i_unique_responses:
    print(item)

for item in output_ii_unique_responses:
    print(item)


