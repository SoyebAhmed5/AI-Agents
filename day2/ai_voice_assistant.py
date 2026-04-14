import speech_recognition as sr
import pyttsx3
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Load AI Model 
llm = OllamaLLM(model="phi")

# Initialize the speech recognizer and text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 160)  # Set speech rate

# speech recognition
recognizer = sr.Recognizer()

# Function to Speak
def speak(text):
    engine.say(text)
    engine.runAndWait()
    
# Function to Listen
def listen():
    with sr.Microphone() as source:
        print("🎤Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            
            query = recognizer.recognize_google(audio)
            print(f"👱🏻You said: {query}")
            return query.lower()
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return None
        except sr.RequestError:
            print("Sorry, my speech service is down.")
            return None
        
# AI chat Prompt
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="Previous Conversion: {chat_history}\n\nUser: {question}\nAI:"
)

# Function to process AI responses
def run_chain(question):
    
    # Retrieve chat history
    chat_history_text = "\n".join([f"{msg.type.value.capitalize()}: {msg.content}" for msg in chat_history.messages])
    
    # Run AI response Generation
    response = llm.revoke(prompt.format(chat_history=chat_history_text, question=question))
    
    # store the question and response in chat history
    chat_history.add_user_message(question)
    chat_history.add_ai_message(response)
    return response

# Main Loop
speak("Hello! I am your AI voice assistant. How can I help you today?")
while True:
    query = listen()
    if "exit" in query or "quit" in query or "goodbye" in query or "stop" in query:
        speak("Goodbye!")
        break
    if query:
        response = run_chain(query)
        print(f"🤖AI: {response}")
        speak(response)