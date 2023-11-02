import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from htmlTemplates import css, bot_template, user_template
import csv

import pandas as pd


def read_csv_file(file_path):
    '''
    Read csv file and return result as pandas dataframe.

    Parameters:
    file_path (str): The path of the csv file

    Returns:
    pd.DataFrame: pandas DataFrame of the csv file
    '''
    # Read the CSV data
    return pd.read_csv(file_path)


def parse_csv_data_to_sentence(data_frame):
    '''
    Parse csv data into sentences.

    Parameters:
    data_frame (pd.DataFrame): The pandas DataFrame from csv file

    Returns:
    list: Lists of sentences for each row
    '''
    # This will store sentences
    sentences = []

    # For each row
    for idx, row in data_frame.iterrows():
        # Create each sentence by joining the key-value pairs.
        sentence = ', '.join(f'{key}={val}' for key, val in row.items())

        # Add sentence to results.
        sentences.append(sentence)

    return sentences


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-4", temperature=1)


    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    print('hi')
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'")
        if st.button("Process"):
            with st.spinner("Processing"):
                # get csv text
                csv_data = read_csv_file(pdf_docs)

                sentences = parse_csv_data_to_sentence(csv_data)
                # get the text chunks
                # text_chunks = get_text_chunks(sentences)

                # create vector store
                vectorstore = get_vectorstore(sentences)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
