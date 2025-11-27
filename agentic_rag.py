import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client, Client

load_dotenv()

# --- SUPABASE CONNECTION ---
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")

if not supabase_url or not supabase_key:
    raise ValueError("❌ Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in environment variables.")

supabase: Client = create_client(supabase_url, supabase_key)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)

# --- LLM ---
llm = ChatOpenAI(model="gpt-4o", temperature=0)


# --- RETRIEVE TOOL ---
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """
    Retrieve relevant Harry Potter passages.
    Clean, structured for the LLM.
    """
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


# --- PROMPT ---
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a friendly Harry Potter BOOK-ONLY assistant.

Your abilities:
- You MAY interpret personalities, motives, relationships, themes, and actions.
- You MAY reason about Snape, Dumbledore, Voldemort, etc.
- You MAY combine retrieved passages with your full knowledge of the books.

Your limits:
1. You operate ONLY inside the BOOK universe.
2. If the user asks about anything NON-Harry-Potter (Elon Musk, Marvel, politics),
   respond: "I can only answer questions about the Harry Potter books."
3. You MUST be honest about retrieved passages:
   - If the context does NOT contain the exact event/quote asked for,
     say: "The retrieved text does not show this exact moment."
4. You may still answer from your overall knowledge of the books afterward.
5. NEVER hallucinate page numbers, quotes, or scenes that were not retrieved.

Answer style:
- Clear, structured, book-accurate.
- Refer back to retrieved text when relevant.
"""
        ),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

tools = [retrieve]

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    question = input("Ask something from Harry Potter: ")
    response = executor.invoke({"input": question})
    print("\n--- ANSWER ---\n")
    print(response["output"])
