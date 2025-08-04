import os
from dotenv import load_dotenv

from typing import TypedDict, Union
from typing import Literal
from typing import List, Optional, Any

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages 
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.runtime import Runtime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from llama_index.llms.google_genai import GoogleGenAI
from google import genai

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key = api_key)

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
    from backend.rag.vector_store import rag_tool
    return rag_tool(query)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=api_key,
    convert_system_message_to_human=True
    )

# response = llm.invoke([HumanMessage(content="What is the capital of Canada?")])
# print(response.content)
tool_map = {
    "llm_tool": llm_query_tool, 
    "rag_tool": rag_query_tool,
    }
tools = list(tool_map.values())
agent = create_react_agent(model=llm, tools=tools)

class AgentState(TypedDict):
    input
    agent_out: Union[str, dict]

# Define agent node
def agent_node(state: AgentState) -> AgentState:
    """Run the agent with the given state."""
    query = state.get("input", "")
    # messages = state.get("messages", [])
    response = agent.invoke({"input": query})
    # state["messages"] = add_messages(messages, response["messages"])
    return {"agent_out": response}

# Define tool call node
def tool_call_node(state: AgentState) -> AgentState:
    tool_call = state["agent_out"]
    tool_name = tool_call["tool"]
    tool_input = tool_call["tool_input"]

    if tool_name not in tool_map:
        return {"agent_out": f"[Error] Unknown tool: {tool_name}"}

    result = tool_map[tool_name].invoke(tool_input)
    return {"agent_out": result}

def should_call_tool(state: AgentState) -> bool:
    return isinstance(state["agent_out"], dict) and "tool" in state["agent_out"]

builder = StateGraph(AgentState)

builder = StateGraph(AgentState)
builder.set_entry_point("agent")

builder.add_node("agent", agent_node)
builder.add_node("tool_call", tool_call_node)

builder.add_conditional_edges(
    "agent",
    should_call_tool,
    path_map={
        True: "tool_call",
        False: END
    }
)

builder.add_edge("tool_call", END)

graph = builder.compile()

if __name__ == "__main__":
    while True:
        q = input("Ask: ")
        if q.strip().lower() in {"exit", "quit"}:
            break
        result = graph.invoke({"input": q})
        print("Final Answer:", result["agent_out"])

# response = graph.invoke({"input": "What is the capital of France?"})
# print(response["agent_out"])

# output = graph.invoke(
#     {
#         "messages": [
#             {"role": "user", "content": "What is llama?"},
#         ],
#     }
# )

# print(output["messages"][-1].text())