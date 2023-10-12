from flask import Flask, request, jsonify
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

app = Flask(__name__)

# 初始化Chatbot
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
repo_id = os.environ.get('repo_id')
port = os.getenv('port')

llm = HuggingFaceHub(repo_id=repo_id,
                     model_kwargs={"min_length":1024,
                                   "max_new_tokens":5632, "do_sample":True,
                                   "temperature":0.1,
                                   "top_k":50,
                                   "top_p":0.95, "eos_token_id":49155}) 

prompt_template = """You are a very helpful AI assistant. Please response to the user's input question with as many details as possible.
Question: {user_question}
Helpful AI Repsonse:
"""  
llm_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))

# 定义API端点
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_query = data['query']

    # 调用Chatbot
    initial_response = llm_chain.run(user_query)

    return jsonify({'response': initial_response})

if __name__ == '__main__':    
    app.run(host='0.0.0.0', port=port)
