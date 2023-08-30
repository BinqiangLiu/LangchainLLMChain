import streamlit as st
from langchain import PromptTemplate, LLMChain
from langchain.memory import StreamlitChatMessageHistory
from streamlit_chat import message
import numpy as np
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from streamlit.components.v1 import html
from langchain import HuggingFaceHub

import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Open AI Chat Assistant", layout="wide")
st.subheader("Open AI Chat Assistant: Life Enhancing with AI!")

css_file = "main.css"
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
#HUGGINGFACEHUB_API_TOKEN = os.environ.get('HUGGINGFACEHUB_API_TOKEN')
repo_id = os.environ.get('repo_id')

repo_id="HuggingFaceH4/starchat-beta"
llm = HuggingFaceHub(repo_id=repo_id,
                     model_kwargs={"min_length":100,
                                   "max_new_tokens":1024, "do_sample":True,
                                   "temperature":0.1,
                                   "top_k":50,
                                   "top_p":0.95, "eos_token_id":49155})

user_query = st.text_input("Enter your query here:")
if user_query != "":
    prompt_template = "You are a helpful AI assistant. Please reponse to {user_query} in details and in a good manner."    
    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(prompt_template)
    )
    initial_response=llm_chain.run(user_query)
    temp_ai_response_1=initial_response.partition('<|end|>\n<|user|>\n')[0]
    temp_ai_response_2=temp_ai_response_1.replace('<|end|>\n<|assistant|>\n', '') 
    final_ai_response=temp_ai_response_2.replace('<|end|>\n<|system|>\n', '') 
    print("final_ai_response"+final_ai_response)
    st.write("AI Response:\n"+final_ai_response)
    


