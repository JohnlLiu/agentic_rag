import os
from dotenv import load_dotenv

from typing import Annotated
from typing import Literal
from typing import List, Optional, Any
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.runtime import Runtime

from langchain_core.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from llama_index.llms.google_genai import GoogleGenAI
from google import genai

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key = api_key)


class CustomGoogleGenAI(BaseChatModel):
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        super().__init__()  # important to avoid the __slots__ error
        self._llm = GoogleGenAI(model=model, api_key=api_key)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> ChatResult:
        prompt = " ".join(
            m.content for m in messages if isinstance(m, HumanMessage)
        )

        response = self._llm.complete(prompt)

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=response.text)
                )
            ]
        )

    @property
    def _llm_type(self) -> str:
        return "custom_google_genai"

class CustomContext:
    tools: list[Literal["llm_query_tool"]]

@tool
def llm_query_tool(query: str) -> str:
    """Query the LLM with a question."""
    llm = GoogleGenAI(
        model="gemini-2.0-flash",
        api_key=api_key,
    )
    return llm.complete(query).text


llm = CustomGoogleGenAI(api_key=api_key)

# response = llm.invoke([HumanMessage(content="What is the capital of Canada?")])
# print(response.content)

agent = create_react_agent(model=llm, tools=[llm_query_tool])

# output = agent.invoke(
#     {
#         "messages": [
#             {"role": "user", "content": "What is the capital of France?"},
#         ],
#     },
#     context=CustomContext(tools=["llm_query_tool"]),
# )

# print(output["messages"][-1].text())