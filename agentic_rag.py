import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from supabase.client import Client, create_client
from langchain_core.tools import tool

load_dotenv()  


# supabase_url = SUPABASE_URL
# supabase_key = SUPABASE_SERVICE_KEY
# supabase_url = os.environ.get("SUPABASE_URL")
# supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
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

llm = ChatOpenAI(temperature=0) # temperature = 0 means the model will always give the same output for the same input

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

response = agent_executor.invoke({"input": "Name everyone in Ron's family"})

print(response["output"])