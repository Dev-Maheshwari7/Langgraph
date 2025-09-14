from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt

load_dotenv()

memory = MemorySaver()
model = ChatGroq(model="llama-3.1-8b-instant")

prompt1="""You are an helpful AI assistant, you need to act like a job interviewer, and based on what user gives you
           you need to ask questions according to that job, and based on his answers rate him on a scale of 1 to 10.
           If told you to recheck answers then use the tool1 and nothing else, and also print the tool call"""

class State(TypedDict):
    messages:Annotated[list, add_messages]

@tool
def tool1(query: str):
    """use this tool when the user ask to recheck the answer he gave"""
    human_response = interrupt({"query": query})
    return human_response["data"]

tools=[tool1]

agent= create_react_agent(
    model=model,
    tools=tools,
    prompt=prompt1
)


def should_continue(state: State): 
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls: 
        return "end"
    else:
        return "continue"


graph = StateGraph(State)
graph.add_node("agent", agent)
graph.set_entry_point("agent")


tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "agent")

app=graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}


events = app.stream(
     {"messages": [{"role": "user", "content": "i would make a temporarry variable and put one value in it, can you recheck the answers"}]},
     config,
     stream_mode="values",
 )
for event in events:
    if "messages" in event and event["messages"]:
        event["messages"][-1].pretty_print()
    else:
        print("Non-message event:", event)


human_response = (
    "yes you are right, yipee"
)

human_command = Command(resume={"data": human_response})

events = app.stream(human_command, config, stream_mode="values")
for event in events:
    if "messages" in event and event["messages"]:
        event["messages"][-1].pretty_print()
   
