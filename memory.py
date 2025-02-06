import os
import streamlit as st
import tempfile

from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

chat_history = StreamlitChatMessageHistory()

__import__('pysqlite3')
import sys
sys.modules['splite3'] = sys.modules.pop('pysqlite3')

from langchain.vectorstores import Chroma
os.environ["OPENAI_API_KEY"] = "your api key"
