import speech_recognition as sr
import pyttsx3
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate

from langchain_ollama import OllamaLLM

# Load the Ollma(AI) model
llm = OllamaLLM(model="mistral")    

# Initialize Memory (Langchain )
chat_history = ChatMessageHistory()

# Intialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

# Initialize Speech Recognizer
recognizer = sr.Recognizer()

# Function to speak 
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to listen and recognize speech
def listen():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        print(f"You said: {query}")
        return query.lower()
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
        return None
    except sr.RequestError:
        print("Could not request results; check your network connection.")
        return None
# AI chat prompt

prompt =   PromptTemplate(
    input_variables=["chat_history", "question"],
    template = "Previous Converstation : {chat_history}\nUser: {question}\nAI:"
)

# Function to Process AI Response
def run_chain(question):
    # Retrieve chat history 
    chat_history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history.messages])

    # Run AI Response generation 
    response = llm.invoke(prompt.format(chat_history = chat_history_text, question=question))
     
    # Store new user input
    chat_history.add_user_message(question)
    chat_history.add_ai_message(response)

    return response

# Main Loop
speak("Hello, how can I assist you today?")
while True:
    query = listen()
    if query:  # Only proceed if query is not None
        if "exit" in query or "stop" in query:
            speak("Goodbye!")
            break
        response = run_chain(query)
        print(f"AI: {response}")
        speak(response)