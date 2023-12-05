
from typing import Optional, Type
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMMathChain
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from pydantic import BaseModel, Field
from langchain.utilities import SerpAPIWrapper
from langchain.agents import AgentType, initialize_agent

llm = ChatOpenAI(temperature=0)
llm_math_chain = LLMMathChain(llm=llm, verbose=True)

search = SerpAPIWrapper()

class UsageTool(BaseTool):
    name = "search1"
    description = "查询业务区域利用率，设备昨日利用率"

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return "业务区域利用率: 0.53, 设备昨日利用率0.27"

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


class TestInfoTool(BaseTool):
    name = "search2"
    description = "查询极限压测满意度、丢包压测满意度"

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return "IPv4：极限压测满意度: 99.2%, 丢包压测满意度：99.2%。IPv4：极限压测满意度: 0%, 丢包压测满意度：0%。"

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


class NATTool(BaseTool):
    name = "search3"
    description = "查询设备"

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return "NAT类型：NAT0"

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")



class CalculatorInput(BaseModel):
    question: str = Field()


class CustomCalculatorTool(BaseTool):
    name = "Calculator"
    description = "useful for when you need to answer questions about math"
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return llm_math_chain.run(query)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Calculator does not support async")
    


if __name__ == "__main__":
    tools = [UsageTool(), TestInfoTool(), ]
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    agent.run(
        "业务区域利用率多少?"
    )
        