# app.py (Fixed & Upgraded Version)

import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI   # ✅ fixed
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_tavily import TavilySearch   # ✅ updated to modern tool
from langchain.memory import ConversationBufferWindowMemory

# --- 1. SETUP & CONFIGURATION ---

# Load environment variables
load_dotenv()

# Initialize the LLM
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.6)
except Exception as e:
    st.error(f"Error initializing the AI model: {e}", icon="🚨")
    st.stop()

# Initialize tools
search_tool = TavilySearch(max_results=2)
tools = [search_tool]

# --- 2. MEMORY SETUP ---

@st.cache_resource
def get_memory():
    return ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

memory = get_memory()

# --- 3. AGENT PROMPT & CREATION ---

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a wise and practical old mentor. Your goal is to give insightful, actionable advice.
            If a user asks about a modern topic, you MUST use your search tool to get up-to-date information.
            Always ground your advice in reality. Your tone should be calming, experienced, and deeply wise."""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=False)

# --- 4. UI ENHANCEMENTS & APP LAYOUT ---

st.title("🤖 Wise Mentor AI")
st.caption("Your AI-powered guide, now with memory and streaming.")

with st.sidebar:
    st.header("About")
    st.markdown("""
    This is your Wise Mentor AI, built with LangChain and Streamlit.
    It can search the web for current information and remembers the last few turns of your conversation.
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        memory = get_memory()  # reset memory
        st.rerun()

# --- 5. CHAT HISTORY & INTERACTION ---

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_prompt := st.chat_input("What is your question, my friend?"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()

    full_response = ""
    try:
        # Streaming agent response
        for chunk in agent_executor.stream({"input": user_prompt}):
            if "output" in chunk:
                content = chunk["output"]
                full_response += content
                response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)
    except Exception as e:
        full_response = f"I apologize, an error occurred: {e}"
        response_placeholder.error(full_response, icon="🚨")

    st.session_state.messages.append({"role": "assistant", "content": full_response})
