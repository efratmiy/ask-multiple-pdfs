import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import SeleniumURLLoader
from htmlTemplates import css, bot_template, user_template
from bs4 import BeautifulSoup
from selenium import webdriver
from typing import List
from time import sleep
import time
from selenium.webdriver.common.keys import Keys


def get_pdf_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf, strict=True)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# def scrape_webpage(url_list: List[str]):
#     """ Function to extract the body text of a list of web pages """
#
#     # Define webdriver options
#     options = webdriver.ChromeOptions()
#     options.add_argument('--headless')
#
#     # Initialize the Chrome webdriver
#     driver = webdriver.Chrome(options=options)
#     # Scroll down
#     driver.find_element_by_tag_name('body').send_keys(Keys.PAGE_DOWN)
#     time.sleep(2)  # Allow time for content to load
#     if type(url_list) is not list:
#         url_list = url_list.split('\n')
#
#     for url in url_list:
#         try:
#             # Open the URL
#             driver.get(url)
#             # Give some time to load the page and run JavaScript
#             sleep(5)
#             # Parse the page source through BeautifulSoup
#             soup = BeautifulSoup(driver.page_source, 'html.parser')
#
#             # Extracts all tags and joins them.
#             body_text = '\n'.join([tag.get_text(strip=True) for tag in soup.find_all()])
#
#             return body_text
#
#         except Exception as e:
#             print(f'Failed to scrape webpage {url}. Error: {str(e)}')


def get_url_text(urls):
    if type(urls) is not list:
        urls = urls.split('\n')
    loader = SeleniumURLLoader(urls=urls)
    doc = loader.load()
    txt = ' '.join([t.page_content for t in doc])
    return txt


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
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-4", temperature=1)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 1})
    # llm = model_bison

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
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
    st.set_page_config(page_title="Chat with multiple URLs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple URLs :books:")
    user_question = st.text_input("Ask a question about your web page:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        urls = st.text_area(label="past your URLs here and click on 'Process'")
        if st.button("Process"):
            with st.spinner("Processing"):
                # get  text
                # raw_text = scrape_webpage(urls)
                raw_text = get_url_text(urls)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
