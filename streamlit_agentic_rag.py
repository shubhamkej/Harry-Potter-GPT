import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client, Client

load_dotenv()

# -------------------------------------------------------
# PAGE STYLE
# -------------------------------------------------------
st.set_page_config(
    page_title="Hogwarts Library Assistant",
    page_icon="🪄",
    layout="wide",
)

hp_css = """
<style>
.stApp {
    background: radial-gradient(circle at top, #2b2838 0, #050509 60%);
    color: #f5f2e9;
    font-family: "Georgia", "serif";
}
.block-container {
    max-width: 900px;
    padding-top: 2rem;
}
.hp-title {
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
    color: #fecd57;
    text-shadow: 0 0 12px rgba(251,192,45,0.7);
}
.hp-subtitle {
    text-align: center;
    opacity: 0.9;
    margin-bottom: 2rem;
}
[data-testid="stChatMessage"] {
    border-radius: 18px;
    padding: 0.75rem 1rem;
}
[data-testid="stChatMessage"]:has(> div svg[data-testid="user-avatar"]) {
    background: rgba(176,55,55,0.18);
}
[data-testid="stChatMessage"]:has(> div svg[data-testid="assistant-avatar"]) {
    background: linear-gradient(135deg,
        rgba(143,97,220,0.22),
        rgba(255,215,128,0.18)
    );
    border: 1px solid rgba(254,205,87,0.5);
}
</style>
"""
st.markdown(hp_css, unsafe_allow_html=True)

# -------------------------------------------------------
# SUPABASE CONNECTION
# -------------------------------------------------------
supabase_url = os.environ.get("SUPABASE_URL") or st.secrets["SUPABASE_URL"]
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY") or st.secrets["SUPABASE_SERVICE_KEY"]

supabase: Client = create_client(supabase_url, supabase_key)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)

# -------------------------------------------------------
# LLM
# -------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# -------------------------------------------------------
# RETRIEVE TOOL
# -------------------------------------------------------
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve structured passages from the Harry Potter books."""
    retrieved_docs = vector_store.similarity_search(query, k=4)

    parts = []
    for i, doc in enumerate(retrieved_docs, start=1):
        meta = doc.metadata or {}
        page = meta.get("page_label") or meta.get("page") or "unknown page"
        parts.append(
            f"--- Passage {i} (page {page}) ---\n{doc.page_content.strip()}"
        )

    context_text = "\n\n".join(parts)
    return context_text, retrieved_docs


# -------------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the **Hogwarts Library Assistant**, and you operate strictly within
the world of the Harry Potter **BOOKS** by J.K. Rowling.

✔ You MAY interpret:
- personalities (“Snape’s personality”)
- motives (“Why did Dumbledore…?”)
- relationships & themes
- character arcs across all books

✔ You MAY combine:
(a) the retrieved passages AND  
(b) your overall understanding of the Harry Potter books

❌ You MUST NOT:
- discuss anything outside Harry Potter (Elon Musk, Marvel, science, etc.)
  → reply: "I can only answer questions about the Harry Potter books."
- hallucinate quotes or page numbers
- pretend a retrieved passage contains something it does not

Honesty rules:
1. If the user asks about a specific moment (e.g., “first time Voldemort is mentioned”),
   and it is NOT in the retrieved context:
   → Say: “The retrieved text does not show this exact moment.”
   → Then answer from your knowledge of the books.
2. When using retrieved text, reference it:
   “In Passage 2, we see that…”

Answer style:
- Warm, clear, faithful to the books
- Structured in short paragraphs or bullet points
"""
        ),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

tools = [retrieve]
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# -------------------------------------------------------
# UI HEADER
# -------------------------------------------------------
st.markdown('<h1 class="hp-title">Hogwarts Library Assistant 🪄</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hp-subtitle">Ask anything about the Harry Potter books — characters, spells, motives, events, and themes.</p>',
    unsafe_allow_html=True,
)

# -------------------------------------------------------
# CHAT SESSION
# -------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.markdown(msg.content)

user_input = st.chat_input("Ask me anything about the Harry Potter books")

if user_input:
    st.session_state.messages.append(HumanMessage(user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    result = executor.invoke({"input": user_input, "chat_history": st.session_state.messages})
    output = result["output"]

    st.session_state.messages.append(AIMessage(output))
    with st.chat_message("assistant"):
        st.markdown(output)
