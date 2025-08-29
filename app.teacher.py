import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os 

st.title("Streamlit 기본예제")
st.write("소득세에 관련된 모든것을 답변해 드립니다.")

load_dotenv()

if "message_list" not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def get_ai_messages(user_message):

    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    index_name = 'tax-index10'
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding,
    )

    llm = ChatOpenAI(model='gpt-4o')
    prompt = hub.pull("rlm/rag-prompt")
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
    )

    dictionary = ["사람을 나타내는 표현 -> 거주자"]

    prompt2 = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        사전 : {dictionary}

        질문: {{question}}

    """)

    dictionary_chain = prompt2 | llm | StrOutputParser()
    tax_chain = {"query": dictionary_chain} | qa_chain
    ai_message = tax_chain.invoke({"question": user_message})

    return ai_message['result']


if user_question := st.chat_input(placeholder='소득세에 관련된 궁금한 내용들을 말씀하세요.!'):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    ai_message = get_ai_messages(user_question)
    with st.chat_message("ai"):
        st.write(ai_message)
    st.session_state.message_list.append({"role": "ai", "content": ai_message})

  