import streamlit as st
import speech_recognition as sr
import pyttsx3
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate

from langchain_ollama import OllamaLLM

# Load AI Model
llm = OllamaLLM(model="mistral")
# Initialize Memory (Langchain )
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory() # Stores user_AI_Converstation

# Intialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

# Speech Recognizer
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

prompt = PromptTemplate(
    input_variables=["chat_history", "question"],   
    template="Previous Converstation : {chat_history}\nUser: {question}\nAI:"
)

# Function to Process AI Response
def run_chain(question):
    # Retrieve chat history 
    chat_history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in st.session_state.chat_history.messages])

    # Run AI Response generation 
    response = llm.invoke(prompt.format(chat_history=chat_history_text, question=question))
     
    # Store new user input
    st.session_state.chat_history.add_user_message(question)
    st.session_state.chat_history.add_ai_message(response)

    return response

# Streamlit UI
st.title("AI Voice Assistant")
st.write("Click the button and speak to the AI")

# Button to start listening
if st.button("Start Listening"):
    user_input = listen()
    if user_input:
        ai_response = run_chain(user_input)
        st.write(f"**You** {user_input}")
        st.write(f"**AI:** {ai_response}")
        speak(ai_response)

# Display the full chat history

st.subheader("Chat History")
for msg in st.session_state.chat_history.messages:
    st.write(f"**{msg.type}: {msg.content}**")