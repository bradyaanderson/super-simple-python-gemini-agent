import os
import sys
import logging
from typing import List
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import Graph
from langchain_core.runnables import chain
from typing import TypedDict, Annotated

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatState(TypedDict):
    messages: List[HumanMessage | AIMessage]

class GeminiChat:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.error("GEMINI_API_KEY not found in environment variables")
            sys.exit(1)

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=self.api_key,
            temperature=0.7
        )
        
        # Initialize message history
        self.message_history: List[HumanMessage | AIMessage] = []

        # Define the chat function
        @chain
        def chat(state: Annotated[ChatState, "The chat state"]) -> ChatState:
            messages = state["messages"]
            response = self.llm.invoke(messages)
            new_messages = messages + [response]
            return {"messages": new_messages}

        # Create the graph
        self.workflow = Graph()
        self.workflow.add_node("chat", chat)
        self.workflow.set_entry_point("chat")
        self.workflow.set_finish_point("chat")
        
        # Compile the graph
        self.app = self.workflow.compile()

    def get_response(self, user_input: str) -> str:
        try:
            # Add the new user message to history
            new_message = HumanMessage(content=user_input)
            self.message_history.append(new_message)
            
            # Create state with full message history
            state = {"messages": self.message_history}
            
            # Run the graph
            result = self.app.invoke(state)
            
            # Update message history with AI's response
            self.message_history = result["messages"]
            
            # Return the last message (the AI's response)
            return result["messages"][-1].content
        except Exception as e:
            logger.error("Error getting response: %s", e)
            return f"Error: {str(e)}"

def main():
    chat = GeminiChat()
    print("Welcome to Gemini Chat! (Type 'quit' to exit)")

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() == 'quit':
                break
            response = chat.get_response(user_input)
            print(f"\nGemini: {response}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error("Unexpected error: %s", e)
            print("An error occurred. Please try again.")

if __name__ == "__main__":
    main()