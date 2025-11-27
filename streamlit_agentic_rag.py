import os
from dotenv import load_dotenv
import streamlit as st
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain.agents import create_tool_calling_agent
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from supabase.client import Client, create_client

load_dotenv()  


# supabase_url = SUPABASE_URL
# supabase_key = SUPABASE_SERVICE_KEY
supabase_url = os.environ.get("SUPABASE_URL") or st.secrets("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY") or st.secrets("SUPABASE_SERVICE_KEY")
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

supabase: Client = create_client(supabase_url, supabase_key)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)
 
llm = ChatOpenAI(model="gpt-4o",temperature=0)

prompt = hub.pull("hwchase17/openai-functions-agent")

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools = [retrieve]

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🦜")
st.title("🦜 Agentic RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)


user_question = st.chat_input("Name everyone in Ron's family")

if user_question:

    with st.chat_message("user"):
        st.markdown(user_question)

        st.session_state.messages.append(HumanMessage(user_question))


    result = agent_executor.invoke({"input": user_question, "chat_history":st.session_state.messages})

    ai_message = result["output"]

    with st.chat_message("assistant"):
        st.markdown(ai_message)

        st.session_state.messages.append(AIMessage(ai_message))