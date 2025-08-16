import os
from dotenv import load_dotenv

from typing import TypedDict, Union, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages 
from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.runtime import Runtime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from llama_index.llms.google_genai import GoogleGenAI
from langchain_core.runnables.base import Runnable
from google import genai
from tavily import TavilyClient

from backend.rag.query import rag_query

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key = api_key)

tavily_api_key = os.getenv('TAVILY_API_KEY')
tavily_client = TavilyClient(api_key=tavily_api_key)

class State(TypedDict):
    messages: Annotated[list, add_messages]

memory = MemorySaver()

@tool
def llm_query_tool(query: str) -> str:
    """Query the LLM with a question."""
    llm = GoogleGenAI(
        model="gemini-2.0-flash",
        api_key=api_key,
    )
    return llm.complete(query).text

@tool
def rag_query_tool(query: str) -> str:
    """Query the RAG system with a question."""
    return rag_query(query)

@tool
def tavily_search_tool(query: str) -> str:
    """Search the web using Tavily."""
    response = tavily_client.search(query)
    return response

tool_map = {
    "llm_tool": llm_query_tool,
    "rag_tool": rag_query_tool,
    "search_tool": tavily_search_tool,
    }

tools = list(tool_map.values())
tool_node = ToolNode(tools)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    )

llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    input: str
    agent_out: Union[str, dict]

def tool_router(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

def agent(state: State):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    print(response)
    return {"messages": [response]}

builder = StateGraph(State)

builder.add_node("agent", agent)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tool_router, ["tools", END])
# builder.add_edge("tools", "agent")

graph = builder.compile(checkpointer=memory)

def stream_graph_updates(user_input: str):
    for event in graph.stream(
        {"messages": [("user", user_input)]},
        {"configurable": {"thread_id": "1"}},
    ):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        print("Error. Goodbye!")
        break

# https://github.com/lchavasse/langgraph-tutorial/blob/main/basic_rag_agent.py