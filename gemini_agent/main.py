import os
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

class GeminiChat:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.7
        )
        self.message_history: List[HumanMessage | AIMessage] = []

    def get_response(self, user_input: str) -> str:
        try:
            # Add user message to history
            self.message_history.append(HumanMessage(content=user_input))

            # Get response from Gemini
            response = self.llm.invoke(self.message_history)
            self.message_history.append(response)

            return response.content
        except Exception as e:
            return f"Error: {str(e)}"

def main():
    chat = GeminiChat()
    print("Welcome to Super Simple Gemini Agent! (Type 'quit' to exit)")

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

if __name__ == "__main__":
    main()
