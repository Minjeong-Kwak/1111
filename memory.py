import os
import streamlit as st
import tempfile
import requests

from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import StreamlitChatMessageHistory

# Chroma 대신 FAISS 사용
from langchain.vectorstores import FAISS

def initialize_components(selected_model):
    # 디버그 출력
    st.write("===== DEBUG secrets =====")
    st.write(st.secrets)  # st.secrets에 어떤 키가 있는지 살펴보기

    if "OPENAI_API_KEY" not in st.secrets:
        st.write("❌ OPENAI_API_KEY not found in secrets.toml")
        return None

    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(_docs)
    # FAISS는 persist_directory 없이 in-memory로 생성할 수 있음
    vectorstore = FAISS.from_documents(
        split_docs, 
        OpenAIEmbeddings(model='text-embedding-ada-002')
    )
    return vectorstore

@st.cache_resource
def get_vectorstore(_docs):
    # FAISS는 간단히 in-memory 벡터스토어를 사용
    return create_vector_store(_docs)

@st.cache_resource
def download_pdf_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        temp_file_path = "downloaded_law.pdf"
        with open(temp_file_path, "wb") as f:
            f.write(response.content)
        return temp_file_path
    else:
        raise ValueError(f"파일을 다운로드할 수 없습니다. 상태 코드: {response.status_code}")

@st.cache_resource
def initialize_components(selected_model):
    github_url = "https://raw.githubusercontent.com/Minjeong-Kwak/1111/main/law.pdf"
    
    # GitHub에서 PDF 다운로드 후 로컬 경로 가져오기
    file_path = download_pdf_from_github(github_url)
    
    # PDF 로딩
    pages = load_and_split_pdf(file_path)
    vectorstore = get_vectorstore(pages)
    retriever = vectorstore.as_retriever()
    
    # 기존 코드 유지
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """Given a chat history and the latest user question 
                          which might reference context in the chat history, 
                          formulate a standalone question 
                          which can be understood without the chat history. 
                          Do not answer the question, just reformulate it 
                          if needed and otherwise return it as is."""),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are an assistant for question-answering tasks. 
                          Use the following pieces of retrieved context 
                          to answer the question. 
                          If you don't know the answer, just say that you don't know. 
                          Keep the answer perfect. 
                          Please use emoji with the answer. 
                          대답은 한국어로 하고, 존댓말을 써줘.
                          {context}"""),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(model=selected_model)
    # 아래 create_history_aware_retriever, create_stuff_documents_chain, create_retrieval_chain 
    # 함수들은 이미 정의되어 있다고 가정합니다.
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

st.header("헌법 Q&A 챗봇 💬 📚")
option = st.selectbox("Select GPT Model", ("gpt-4o", "gpt-3.5-turbo-0125"))
rag_chain = initialize_components(option)
chat_history = StreamlitChatMessageHistory(key="chat_messages")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role" : "assistant",
                                      "content" : "헌법에 대해 무엇이든 물어보세요!"}]

for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt_message := st.chat_input("Your question"):
    st.chat_message("human").write(prompt_message)
    response = rag_chain.invoke({"input": prompt_message, "history": chat_history.messages})
    answer = response.get("answer", "")
    st.chat_message("ai").write(answer)
    with st.expander("참고 문서 확인"):
        for doc in response.get("context", []):
            st.markdown(doc.metadata["source"], help=doc.page_content)
