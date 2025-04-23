from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize HuggingFace LLM
llm = HuggingFaceEndpoint(
    repo_id="microsoft/DialoGPT-medium",
    task="text-generation",
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    max_length=512,
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
)

# Wrap with LangChain Chat interface
chat_model = ChatHuggingFace(llm=llm)

# Define state structure for LangGraph
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

# Create graph builder with defined state
graph_builder = StateGraph(ChatState)

# Define core chatbot logic
def chatbot_node(state: ChatState):
    response = chat_model.invoke(state["messages"])
    return {"messages": [response]}

# Add chatbot node to graph
graph_builder.add_node("chatbot", chatbot_node)

# Connect nodes and define flow
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")

# Compile the graph
graph = graph_builder.compile()

# Function to handle streaming chatbot responses
def stream_graph_response(user_input: str):
    initial_state = {"messages": [{"role": "user", "content": user_input}]}
    for event in graph.stream(initial_state):
        for value in event.values():
            response = value["messages"][-1].content
            print("Assistant:", response)

# Run chatbot loop
def run_chat():
    print("Chatbot ready! Type 'quit' to exit.")
    while True:
        try:
            user_input = input("User: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_response(user_input)
        except Exception as e:
            print(f"Error occurred: {e}")
            break

if __name__ == "__main__":
    run_chat()
