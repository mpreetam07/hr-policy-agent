import streamlit as st
import uuid

# IMPORT GRAPH APP
from agent import app


# CACHE GRAPH
@st.cache_resource
def load_app():
    return app

graph_app = load_app()


# SESSION STATE INIT
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# SIDEBAR
with st.sidebar:
    st.title("HR Policy Bot 🤖")

    st.markdown("""
### 📌 About
This AI assistant answers HR policy queries using:
- Retrieval (ChromaDB)
- Memory (thread_id)
- Tools (date/time)
- Self-evaluation

### 📚 Topics Covered
- Leave Policy  
- Attendance  
- Salary  
- Resignation  
- Work From Home  
""")

    if st.button("🔄 New Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()


# MAIN UI
st.title("💼 HR Policy Assistant")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# USER INPUT
user_input = st.chat_input("Ask your HR question...")

if user_input:

    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Call Agent
    result = graph_app.invoke(
        {"question": user_input},
        config={"configurable": {"thread_id": st.session_state.thread_id}}
    )

    answer = result.get("answer", "Error generating response.")

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })