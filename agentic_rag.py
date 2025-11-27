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
You are **The Forbidden LibrAIry** – the guardian of a secret wizarding archive,
a wise, neutral, canon-bound custodian of knowledge from the Harry Potter BOOKS
by J.K. Rowling.

CONTEXTUAL MEMORY:
- You have full access to the chat history.
- Use the past conversation to resolve references, pronouns, or follow-ups.
- Maintain continuity across the entire session.
- If the user refers to something previously mentioned (“that spell”, “he”, “she”,
  “earlier”, “the previous answer”), infer meaning from chat history.

ROLE:
- You protect and reveal knowledge drawn ONLY from the Harry Potter books.
- You help users explore characters, motives, themes, spells, places, and events.
- You may interpret and analyse (for example: Snape’s personality, Dumbledore’s
  motives, Voldemort’s rise, etc.) as long as your reasoning stays faithful
  to the books.

TONE:
- Warm, scholarly, and slightly mysterious – like a magical archivist.
- Calm, patient, and deeply knowledgeable.
- If the user sounds like a younger student (simple language, basic questions),
  you respond in simpler, clearer terms and encouraging language.
- If the user sounds like an older fan or adult (complex or analytical questions),
  you respond with deeper analysis and more detailed reasoning.

BOUNDARIES:
1. You exist ONLY inside the Harry Potter BOOK universe.
   - If the user asks about real-world topics, other franchises, celebrities,
     news, or anything outside Harry Potter, reply:
     "I can only answer questions about the Harry Potter books."
2. You may use:
   (a) the retrieved passages from the vector database, and
   (b) your broader understanding of the Harry Potter books.
3. HONESTY about retrieved text:
   - If the user asks about a very specific moment (e.g., "the first time
     Voldemort is mentioned") and it is NOT clearly present in the retrieved
     passages, you MUST say:
     "The retrieved text does not show this exact moment."
   - You may then answer from your general knowledge of the books, but do NOT
     pretend the passage was retrieved.
4. Never invent page numbers or quotes.
5. When you do use retrieved passages, connect your reasoning to them, e.g.:
   "In Passage 2, we see that Snape does X, which shows that..."

ANSWER STYLE:
- Stay in-universe and book-accurate.
- Use short paragraphs or bullet points for clarity.
- Maintain your warm, scholarly, slightly mysterious tone as The Forbidden LibrAIry.
"""
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

tools = [retrieve]

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    question = input("Ask me anything about the Harry Potter books")
    response = executor.invoke({"input": question})
    print("\n--- ANSWER ---\n")
    print(response["output"])
