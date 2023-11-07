from typing import List, Union
import re

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import HumanMessage, AgentAction, AgentFinish
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from htmlTemplates import css, bot_template, user_template


#
# def parse_csv_data_to_sentence(file_path):
#
#     data_frame = pd.read_csv(file_path)
#     sentences = []
#
#     # For each row
#     for idx, row in data_frame.iterrows():
#         # Create each sentence by joining the key-value pairs.
#         sentence = ', '.join(f'{key}={val}' for key, val in row.items())
#
#         # Add sentence to results.
#         sentences.append(sentence)
#
#     return sentences
#

def agent_define():
    csv_agent = create_csv_agent(
        ChatOpenAI(temperature=0),
        'wc-product-export-21-10-2023-1697892104000 (1).csv',
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    tools = [
        Tool(
            name="csv_agent",
            func=csv_agent.run,
            description="useful for when you need to answer questions about products in the store"
        )
    ]

    # Set up a prompt template
    class CustomPromptTemplate(BaseChatPromptTemplate):
        # The template to use
        template: str
        # The list of tools available
        tools: List[Tool]

        def format_messages(self, **kwargs) -> str:
            # Get the intermediate steps (AgentAction, Observation tuples)
            # Format them in a particular way
            intermediate_steps = kwargs.pop("intermediate_steps")
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\nObservation: {observation}\nThought: "
            # Set the agent_scratchpad variable to that value
            kwargs["agent_scratchpad"] = thoughts
            # Create a tools variable from the list of tools provided
            kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
            # Create a list of tool names for the tools provided
            kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
            formatted = self.template.format(**kwargs)
            return formatted

    template_with_history = """
    Thought: {{thought}}
    Action: {{action}}
    Action Input: {{action_input}}
    {% for name, result in action_outputs.items() %}
    Observation: {{result.observation}}
    {% endfor %}
    Final Answer: {{answer}}
    """

    prompt_with_history = CustomPromptTemplate(
        template=template_with_history,
        tools=[],
        input_variables=["input", "intermediate_steps", "history"]
    )

    llm = ChatOpenAI(model_name="gpt-4", temperature=1)

    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(llm=llm, prompt=prompt_with_history)

    tool_names = [tool.name for tool in tools]

    class CustomOutputParser(AgentOutputParser):

        def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:

            # Check if agent should finish
            if "Final Answer:" in llm_output:
                return AgentFinish(
                    # Return values is generally always a dictionary with a single `output` key
                    # It is not recommended to try anything else at the moment :)
                    return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                    log=llm_output,
                )

            # Parse out the action and action input
            regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
            match = re.search(regex, llm_output, re.DOTALL)

            # If it can't parse the output it raises an error
            # You can add your own logic here to handle errors in a different way i.e. pass to a human, give a canned response
            if not match:
                raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            action = match.group(1).strip()
            action_input = match.group(2)

            # Return the action and action input
            return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

    output_parser = CustomOutputParser()
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names,

    )
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
    return agent_executor


#
#
# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=500,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks
#
#
# def get_vectorstore(text_chunks):
#     embeddings = OpenAIEmbeddings()
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore
#

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-4", temperature=1, )

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    agent = st.session_state.agent
    response = agent.run({'history': st.session_state.chat_history, 'input': user_question})
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
    st.set_page_config(page_title="Chat with csv's",
                       page_icon=":chart:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "agent" not in st.session_state:
        st.session_state.agent = agent_define()

    st.header("Chat with multiple CSVs :chart:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(user_question)
    #
    # with st.sidebar:
    #     st.subheader("Your documents")
    #     docs = st.file_uploader(
    #         "Upload your CSV's here and click on 'Process'")
    #     if st.button("Process"):
    #         with st.spinner("Processing"):
    #             # get csv text
    #             sentences = parse_csv_data_to_sentence(docs)
    #             # get the text chunks
    #             # text_chunks = get_text_chunks(sentences)
    #
    #             # create vector store
    #             vectorstore = get_vectorstore(sentences)
    #
    #             # create conversation chain
    #             st.session_state.conversation = get_conversation_chain(
    #                 vectorstore)


if __name__ == '__main__':
    main()
