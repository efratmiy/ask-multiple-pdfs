import re
from typing import List, Union

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, load_tools
from langchain.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.tools.plugin import AIPluginTool

from htmlTemplates import css, bot_template, user_template


def agent_define():
    csv_agent = create_csv_agent(
        ChatOpenAI(temperature=0),
        'wc-product-export-21-10-2023-1697892104000 (1).csv',
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    # tool = AIPluginTool.from_plugin_url("https://www.klarna.com/.well-known/ai-plugin.json")


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

        def format_messages(self, **kwargs):
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
            return [HumanMessage(content=formatted)]

    # Set up a prompt template which can interpolate the history
    template_with_history = """

    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    1. ensure you have a tool named csv_agent that contains a df of data about products in a sport store 
    
    2. Your primary task is to assist customers in finding the products they are seeking. Engage in conversation with them and ask for specific details about the product they want, such as its name, description, or any other relevant information.
    
    3. If the question posed by the customer requires additional steps or if there is an error in the data provided, continue thinking and calculating internally without displaying any progress or intermediate results in the chat. Share only the final answer with the customer.
    
    4. To search for product specifications, ensure that you check both the "שם" (Name) and "תיאור קצר" (Short Description) columns in the CSV file. Look for matches or related information in these columns to quickly provide accurate answers to customer queries.
    
    5. Remember, your duty is to keep customers within the store's ecosystem and refrain from suggesting or directing them to search the web. Utilize the available information within the CSV file to address
    
    6. when using the tool csv_agent, the action should be a normal sentence that explains the query needed, and not the actual query. 
    
    7. the quetions and the answers will be in hebrew only
    
    Previous conversation history:
    {history}

    New question: {input}
    {agent_scratchpad}"""

    prompt_with_history = CustomPromptTemplate(
        template=template_with_history,
        tools=tools,
        input_variables=["input", "intermediate_steps", "history"]
    )

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

    llm = ChatOpenAI(model_name="gpt-4", temperature=1)

    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(llm=llm, prompt=prompt_with_history)
    tool_names = [tool.name for tool in tools]
    output_parser = CustomOutputParser()

    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names
    )
    # Initiate the memory with k=2 to keep the last two turns
    # Provide the memory to the agent
    memory = ConversationBufferWindowMemory(k=2)
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True)
    return agent_executor


def handle_userinput(user_question):
    agent = st.session_state.agent
    st.session_state.chat_history.append(user_question)

    intermediate_steps = None
    response = agent.run({'history': st.session_state.chat_history, 'input': user_question})
    st.session_state.chat_history.append(response)

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with csv's",
                       page_icon=":chart:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent" not in st.session_state:
        st.session_state.agent = agent_define()

    st.header("Chat with multiple CSVs :chart:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(user_question)


if __name__ == '__main__':
    main()
