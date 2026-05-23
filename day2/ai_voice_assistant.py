import streamlit as st
import speech_recognition as sr
import pyttsx3
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Load AI Model
llm = OllamaLLM(model="phi")

# initialize Memory (Langchain)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory() # Stores user_ai conversation
    
#Initialise Text-to-Speech Engine
engine = pyttsx3.init()
engine.setProperty("rate", 160)  #speaking speed

# Speech Recognition
recognizer = sr.Recognizer()

# Function to Speak AI Response
def speak(text):
    engine.say(text)
    engine.runAndWait()
    
# Function to listen to Speak AI Response
def listen():
    with sr.Microphone() as source:
        st.write(" Listening...!")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        st.write(f" You said: {query}")
        return query.lower()
    except st.UnknownValueError:
        st.write("Sorry, I couldn't understand. Try Again!")
        return ""
    except st.RequestError:
        st.write("Speech Recognition Service Unavailable ")
        return ""
    
# AI chat Prompt
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="Previous Conversation: {chat_history}\n\nUser: {question}\nAI:"
)

# Function to process AI responses
def run_chain(question):
    # Retrieve chat history
    chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in st.session_state.chat_history.messages])
    
    # Run AI response Generation
    response = llm.invoke(prompt.format(chat_history=chat_history_text, question=question))
    
    # store the question and response in chat history
    st.session_state.chat_history.add_user_message(question)
    st.session_state.chat_history.add_ai_message(response)
    return response

# Streamlit UI
st.title("AI Voice Assistant")
st.write("Click the button and speak to the AI assistant.")

# Button to Record voice input
if st.button("🎤 Speak"):
    user_query = listen()
    if user_query:
        ai_response = run_chain(user_query)
        st.write(f"👱🏻 You: {user_query}")
        st.write(f"🤖 AI: {ai_response}")
        speak(ai_response)
        
# Display chat history
st.subheader("Chat History")
for msg in st.session_state.chat_history.messages:
    if msg.type == "user":
        st.write(f"👱🏻 You: {msg.content}")
    else:
        st.write(f"🤖 AI: {msg.content}")
