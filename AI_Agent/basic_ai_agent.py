import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Load AI Model from Ollama
llm = OllamaLLM(model="phi")

# initializing Memory
# chat_history = ChatMessageHistory()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

#Define AI Chat Prompt
prompt = PromptTemplate(
    input_variables=['chat_history', 'question'],
    template="Previous Conversion: {chat_history}\n User: {question}\n AI:",
)

#Function To run AI Chat with Memory
def run_chain(question):
    #retrieve chat history
    chat_history_text="\n".join([f"{msg.type.capitalize()}: {msg.content} " for msg in st.session_state.chat_history.messages])
    
    #Run the AI response generation
    response = llm.invoke(prompt.format(chat_history=chat_history_text, question=question))
    
    #store new user input and AI response in memory
    st.session_state.chat_history.add_user_message(question)
    st.session_state.chat_history.add_ai_message(response)
    
    return response


# Streamlit UI
st.title("AI Chatbot 🤖")
st.write("Ask me anything!")

user_input = st.text_input("Your Action:")
if user_input:
    response= run_chain(user_input)
    st.write(f"**👱🏻You:** {user_input}")
    st.write(f"**🤖AI:** {response}")


#show full chat history
st.subheader("Chat History")
for msg in st.session_state.chat_history.messages:
    st.write(f"**{msg.type.capitalize()}:** {msg.content}")

# while True:
#     user_input= input("You: ")
#     if user_input.lower()=="exit":
#         print("Goodbye!!")
#         break
    
#     ai_response = run_chain(user_input)
#     print(f"AI: {ai_response}")