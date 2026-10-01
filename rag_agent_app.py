# import streamlit as st
# import merged_rag_agent


# # ============================================================
# # PAGE CONFIGURATION
# # ============================================================

# st.set_page_config(
#     page_title="Unified RAG AI Agent",
#     page_icon="🤖",
#     layout="wide"
# )


# # ============================================================
# # CUSTOM HEADER
# # ============================================================

# st.markdown(
#     """
#     <style>

#     .main-title {
#         font-size: 38px;
#         font-weight: 700;
#         margin-bottom: 5px;
#     }

#     .subtitle {
#         font-size: 17px;
#         color: #777777;
#         margin-bottom: 25px;
#     }

#     </style>
#     """,
#     unsafe_allow_html=True
# )


# st.markdown(
#     '<div class="main-title">🤖 Unified RAG + Web Search AI Agent</div>',
#     unsafe_allow_html=True
# )

# st.markdown(
#     '<div class="subtitle">'
#     'Ask questions from your local PDF, Word, Excel files or the web.'
#     '</div>',
#     unsafe_allow_html=True
# )


# # ============================================================
# # INITIALIZE AGENT
# # ============================================================

# @st.cache_resource
# def initialize_agent():

#     # Load all local documents
#     local_documents = merged_rag_agent.load_all_local_documents()

#     # Create FAISS vector database
#     vectorstore = merged_rag_agent.create_vector_database(
#         local_documents
#     )

#     return vectorstore, len(local_documents)


# try:

#     vectorstore, total_documents = initialize_agent()

# except Exception as e:

#     st.error("❌ Failed to initialize the RAG agent.")

#     st.exception(e)

#     st.stop()


# # ============================================================
# # SIDEBAR
# # ============================================================

# with st.sidebar:

#     st.title("⚙️ Agent Info")

#     st.divider()

#     st.subheader("📚 Local Sources")

#     st.write("📄 PDF")
#     st.write("📝 Word")
#     st.write("📊 Excel")

#     st.divider()

#     st.subheader("🌐 Online Sources")

#     st.write("🔎 DuckDuckGo Web Search")

#     st.divider()

#     st.subheader("🧠 AI Components")

#     st.write("🔹 FAISS Vector Search")
#     st.write("🔹 Sentence Transformers")
#     st.write("🔹 Groq LLM")

#     st.divider()

#     st.metric(
#         "Local Document Sections",
#         total_documents
#     )

#     st.divider()

#     if st.button(
#         "🗑️ Clear Chat",
#         use_container_width=True
#     ):

#         st.session_state.messages = []

#         st.rerun()


# # ============================================================
# # CHAT HISTORY
# # ============================================================

# if "messages" not in st.session_state:

#     st.session_state.messages = []


# # ============================================================
# # WELCOME MESSAGE
# # ============================================================

# if len(st.session_state.messages) == 0:

#     st.info(
#         "👋 Welcome! Ask me something about your local documents "
#         "or ask a question that requires current web information."
#     )


# # ============================================================
# # DISPLAY CHAT HISTORY
# # ============================================================

# for message in st.session_state.messages:

#     with st.chat_message(message["role"]):

#         st.markdown(message["content"])

#         # Show sources if available
#         if message["role"] == "assistant":

#             local_sources = message.get(
#                 "local_sources",
#                 []
#             )

#             web_sources = message.get(
#                 "web_sources",
#                 []
#             )

#             if local_sources:

#                 with st.expander(
#                     "📚 Local Sources Used"
#                 ):

#                     seen_sources = set()

#                     for doc in local_sources:

#                         source = doc.metadata.get(
#                             "source_file",
#                             "Unknown"
#                         )

#                         page = doc.metadata.get(
#                             "page",
#                             ""
#                         )

#                         sheet = doc.metadata.get(
#                             "sheet",
#                             ""
#                         )

#                         source_text = source

#                         if page:
#                             source_text += (
#                                 f" — Page {page}"
#                             )

#                         if sheet:
#                             source_text += (
#                                 f" — Sheet: {sheet}"
#                             )

#                         if source_text not in seen_sources:

#                             st.write(
#                                 f"• {source_text}"
#                             )

#                             seen_sources.add(
#                                 source_text
#                             )

#             if web_sources:

#                 with st.expander(
#                     "🌐 Web Sources Used"
#                 ):

#                     for result in web_sources:

#                         title = result.get(
#                             "title",
#                             "Untitled"
#                         )

#                         url = result.get(
#                             "url",
#                             ""
#                         )

#                         st.write(
#                             f"• {title}"
#                         )

#                         if url:

#                             st.markdown(
#                                 f"[Open Source]({url})"
#                             )


# # ============================================================
# # USER INPUT
# # ============================================================

# user_query = st.chat_input(
#     "Ask your question..."
# )


# # ============================================================
# # PROCESS USER QUERY
# # ============================================================

# if user_query:

#     # --------------------------------------------------------
#     # SHOW USER MESSAGE
#     # --------------------------------------------------------

#     st.session_state.messages.append(
#         {
#             "role": "user",
#             "content": user_query
#         }
#     )

#     with st.chat_message("user"):

#         st.markdown(user_query)


#     # --------------------------------------------------------
#     # GENERATE ASSISTANT RESPONSE
#     # --------------------------------------------------------

#     with st.chat_message("assistant"):

#         with st.spinner(
#             "🤔 Analyzing your query..."
#         ):

#             try:

#                 # =================================================
#                 # LOCAL RAG SEARCH
#                 # =================================================

#                 local_results = (
#                     merged_rag_agent.search_local_documents(
#                         vectorstore,
#                         user_query,
#                         merged_rag_agent.TOP_K_LOCAL
#                     )
#                 )


#                 # =================================================
#                 # WEB SEARCH DECISION
#                 # =================================================

#                 web_results = []

#                 if merged_rag_agent.needs_web_search(
#                     user_query
#                 ):

#                     web_results = (
#                         merged_rag_agent.duckduckgo_search(
#                             user_query,
#                             merged_rag_agent.TOP_K_WEB
#                         )
#                     )


#                 # =================================================
#                 # GENERATE FINAL ANSWER
#                 # =================================================

#                 answer = (
#                     merged_rag_agent.generate_answer(
#                         user_query,
#                         local_results,
#                         web_results
#                     )
#                 )


#                 # =================================================
#                 # DISPLAY ANSWER
#                 # =================================================

#                 st.markdown(answer)


#                 # =================================================
#                 # LOCAL SOURCES
#                 # =================================================

#                 if local_results:

#                     with st.expander(
#                         "📚 Local Sources Used"
#                     ):

#                         seen_sources = set()

#                         for doc in local_results:

#                             source = doc.metadata.get(
#                                 "source_file",
#                                 "Unknown"
#                             )

#                             page = doc.metadata.get(
#                                 "page",
#                                 ""
#                             )

#                             sheet = doc.metadata.get(
#                                 "sheet",
#                                 ""
#                             )

#                             source_text = source

#                             if page:

#                                 source_text += (
#                                     f" — Page {page}"
#                                 )

#                             if sheet:

#                                 source_text += (
#                                     f" — Sheet: {sheet}"
#                                 )

#                             if source_text not in seen_sources:

#                                 st.write(
#                                     f"• {source_text}"
#                                 )

#                                 seen_sources.add(
#                                     source_text
#                                 )


#                 # =================================================
#                 # WEB SOURCES
#                 # =================================================

#                 if web_results:

#                     with st.expander(
#                         "🌐 Web Sources Used"
#                     ):

#                         for result in web_results:

#                             title = result.get(
#                                 "title",
#                                 "Untitled"
#                             )

#                             url = result.get(
#                                 "url",
#                                 ""
#                             )

#                             st.write(
#                                 f"• {title}"
#                             )

#                             if url:

#                                 st.markdown(
#                                     f"[Open Source]({url})"
#                                 )


#                 # =================================================
#                 # SAVE CHAT
#                 # =================================================

#                 st.session_state.messages.append(
#                     {
#                         "role": "assistant",
#                         "content": str(answer),
#                         "local_sources": local_results,
#                         "web_sources": web_results
#                     }
#                 )


#             except Exception as e:

#                 error_message = (
#                     "❌ An error occurred while processing "
#                     "your question.\n\n"
#                     f"`{str(e)}`"
#                 )

#                 st.error(error_message)

#                 st.session_state.messages.append(
#                     {
#                         "role": "assistant",
#                         "content": error_message
#                     }
#                 )






import os

import streamlit as st

# NOTE: `merged_rag_agent` is deliberately NOT imported here.
# It pulls in torch / sentence-transformers / faiss / fitz, which takes a long
# time on a cold Streamlit Cloud container. Importing it at the top would block
# the whole page (blank screen + endless "Stop" button). It is imported lazily
# inside initialize_agent() below, after the UI has already been drawn.


# ============================================================
# PAGE CONFIGURATION (must be the first Streamlit call)
# ============================================================

st.set_page_config(
    page_title="Unified RAG AI Agent",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# SECRETS -> ENVIRONMENT
# (Streamlit Cloud has no local .env file; keys live in
#  App Settings -> Secrets. Copy them into os.environ so
#  merged_rag_agent can read them the same way as locally.)
# ============================================================

try:
    for _key in ("GROQ_API_KEY",):
        if _key in st.secrets and not os.environ.get(_key):
            os.environ[_key] = st.secrets[_key]
except Exception:
    # No secrets file (e.g. running locally with a .env) - that's fine.
    pass


# ============================================================
# CUSTOM HEADER (rendered immediately, before any heavy work)
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 17px;
        color: #777777;
        margin-bottom: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">🤖 Unified RAG + Web Search AI Agent</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Ask questions from your local PDF, Word, Excel files or the web."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# SIDEBAR (static part - shown while the agent is still loading)
# ============================================================

with st.sidebar:
    st.title("⚙️ Agent Info")
    st.divider()

    st.subheader("📚 Local Sources")
    st.write("📄 PDF")
    st.write("📝 Word")
    st.write("📊 Excel")
    st.divider()

    st.subheader("🌐 Online Sources")
    st.write("🔎 DuckDuckGo Web Search")
    st.divider()

    st.subheader("🧠 AI Components")
    st.write("🔹 FAISS Vector Search")
    st.write("🔹 Sentence Transformers")
    st.write("🔹 Groq LLM")
    st.divider()

    metric_slot = st.empty()  # filled in after the agent has loaded
    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# INITIALIZE AGENT (lazy import + cached)
# ============================================================

@st.cache_resource(
    show_spinner="⏳ Loading AI models and indexing documents "
    "(first start can take a few minutes)..."
)
def initialize_agent():
    # Heavy import happens HERE, after the page is already visible.
    import merged_rag_agent

    # Load all local documents
    local_documents = merged_rag_agent.load_all_local_documents()

    # Create FAISS vector database
    vectorstore = merged_rag_agent.create_vector_database(local_documents)

    return merged_rag_agent, vectorstore, len(local_documents)


try:
    merged_rag_agent, vectorstore, total_documents = initialize_agent()
except Exception as e:
    st.error("❌ Failed to initialize the RAG agent.")
    st.exception(e)
    st.stop()

metric_slot.metric("Local Document Sections", total_documents)


# ============================================================
# HELPERS
# ============================================================

def render_local_sources(local_sources):
    """Show unique local sources in an expander."""
    if not local_sources:
        return

    with st.expander("📚 Local Sources Used"):
        seen_sources = set()

        for doc in local_sources:
            metadata = getattr(doc, "metadata", {}) or {}

            source = metadata.get("source_file", "Unknown")
            page = metadata.get("page", "")
            sheet = metadata.get("sheet", "")

            source_text = str(source)

            if page not in ("", None):
                source_text += f" — Page {page}"

            if sheet:
                source_text += f" — Sheet: {sheet}"

            if source_text not in seen_sources:
                st.write(f"• {source_text}")
                seen_sources.add(source_text)


def render_web_sources(web_sources):
    """Show web sources in an expander."""
    if not web_sources:
        return

    with st.expander("🌐 Web Sources Used"):
        for result in web_sources:
            title = result.get("title", "Untitled")
            url = result.get("url", "")

            st.write(f"• {title}")

            if url:
                st.markdown(f"[Open Source]({url})")


# ============================================================
# WELCOME MESSAGE
# ============================================================

if len(st.session_state.messages) == 0:
    st.info(
        "👋 Welcome! Ask me something about your local documents "
        "or ask a question that requires current web information."
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            render_local_sources(message.get("local_sources", []))
            render_web_sources(message.get("web_sources", []))


# ============================================================
# USER INPUT
# ============================================================

user_query = st.chat_input("Ask your question...")


# ============================================================
# PROCESS USER QUERY
# ============================================================

if user_query:

    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing your query..."):
            try:
                # Local RAG search
                local_results = merged_rag_agent.search_local_documents(
                    vectorstore,
                    user_query,
                    merged_rag_agent.TOP_K_LOCAL,
                )

                # Web search decision
                web_results = []

                if merged_rag_agent.needs_web_search(user_query):
                    web_results = merged_rag_agent.duckduckgo_search(
                        user_query,
                        merged_rag_agent.TOP_K_WEB,
                    )

                # Final answer
                answer = merged_rag_agent.generate_answer(
                    user_query,
                    local_results,
                    web_results,
                )

                st.markdown(answer)

                render_local_sources(local_results)
                render_web_sources(web_results)

                # Save chat
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": str(answer),
                        "local_sources": local_results,
                        "web_sources": web_results,
                    }
                )

            except Exception as e:
                error_message = (
                    "❌ An error occurred while processing "
                    "your question.\n\n"
                    f"`{str(e)}`"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )
