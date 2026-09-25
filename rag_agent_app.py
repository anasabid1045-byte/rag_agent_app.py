import streamlit as st
import merged_rag_agent


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Unified RAG AI Agent",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CUSTOM HEADER
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
    unsafe_allow_html=True
)


st.markdown(
    '<div class="main-title">🤖 Unified RAG + Web Search AI Agent</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions from your local PDF, Word, Excel files or the web.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# INITIALIZE AGENT
# ============================================================

@st.cache_resource
def initialize_agent():

    # Load all local documents
    local_documents = merged_rag_agent.load_all_local_documents()

    # Create FAISS vector database
    vectorstore = merged_rag_agent.create_vector_database(
        local_documents
    )

    return vectorstore, len(local_documents)


try:

    vectorstore, total_documents = initialize_agent()

except Exception as e:

    st.error("❌ Failed to initialize the RAG agent.")

    st.exception(e)

    st.stop()


# ============================================================
# SIDEBAR
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

    st.metric(
        "Local Document Sections",
        total_documents
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


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

        # Show sources if available
        if message["role"] == "assistant":

            local_sources = message.get(
                "local_sources",
                []
            )

            web_sources = message.get(
                "web_sources",
                []
            )

            if local_sources:

                with st.expander(
                    "📚 Local Sources Used"
                ):

                    seen_sources = set()

                    for doc in local_sources:

                        source = doc.metadata.get(
                            "source_file",
                            "Unknown"
                        )

                        page = doc.metadata.get(
                            "page",
                            ""
                        )

                        sheet = doc.metadata.get(
                            "sheet",
                            ""
                        )

                        source_text = source

                        if page:
                            source_text += (
                                f" — Page {page}"
                            )

                        if sheet:
                            source_text += (
                                f" — Sheet: {sheet}"
                            )

                        if source_text not in seen_sources:

                            st.write(
                                f"• {source_text}"
                            )

                            seen_sources.add(
                                source_text
                            )

            if web_sources:

                with st.expander(
                    "🌐 Web Sources Used"
                ):

                    for result in web_sources:

                        title = result.get(
                            "title",
                            "Untitled"
                        )

                        url = result.get(
                            "url",
                            ""
                        )

                        st.write(
                            f"• {title}"
                        )

                        if url:

                            st.markdown(
                                f"[Open Source]({url})"
                            )


# ============================================================
# USER INPUT
# ============================================================

user_query = st.chat_input(
    "Ask your question..."
)


# ============================================================
# PROCESS USER QUERY
# ============================================================

if user_query:

    # --------------------------------------------------------
    # SHOW USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    with st.chat_message("user"):

        st.markdown(user_query)


    # --------------------------------------------------------
    # GENERATE ASSISTANT RESPONSE
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🤔 Analyzing your query..."
        ):

            try:

                # =================================================
                # LOCAL RAG SEARCH
                # =================================================

                local_results = (
                    merged_rag_agent.search_local_documents(
                        vectorstore,
                        user_query,
                        merged_rag_agent.TOP_K_LOCAL
                    )
                )


                # =================================================
                # WEB SEARCH DECISION
                # =================================================

                web_results = []

                if merged_rag_agent.needs_web_search(
                    user_query
                ):

                    web_results = (
                        merged_rag_agent.duckduckgo_search(
                            user_query,
                            merged_rag_agent.TOP_K_WEB
                        )
                    )


                # =================================================
                # GENERATE FINAL ANSWER
                # =================================================

                answer = (
                    merged_rag_agent.generate_answer(
                        user_query,
                        local_results,
                        web_results
                    )
                )


                # =================================================
                # DISPLAY ANSWER
                # =================================================

                st.markdown(answer)


                # =================================================
                # LOCAL SOURCES
                # =================================================

                if local_results:

                    with st.expander(
                        "📚 Local Sources Used"
                    ):

                        seen_sources = set()

                        for doc in local_results:

                            source = doc.metadata.get(
                                "source_file",
                                "Unknown"
                            )

                            page = doc.metadata.get(
                                "page",
                                ""
                            )

                            sheet = doc.metadata.get(
                                "sheet",
                                ""
                            )

                            source_text = source

                            if page:

                                source_text += (
                                    f" — Page {page}"
                                )

                            if sheet:

                                source_text += (
                                    f" — Sheet: {sheet}"
                                )

                            if source_text not in seen_sources:

                                st.write(
                                    f"• {source_text}"
                                )

                                seen_sources.add(
                                    source_text
                                )


                # =================================================
                # WEB SOURCES
                # =================================================

                if web_results:

                    with st.expander(
                        "🌐 Web Sources Used"
                    ):

                        for result in web_results:

                            title = result.get(
                                "title",
                                "Untitled"
                            )

                            url = result.get(
                                "url",
                                ""
                            )

                            st.write(
                                f"• {title}"
                            )

                            if url:

                                st.markdown(
                                    f"[Open Source]({url})"
                                )


                # =================================================
                # SAVE CHAT
                # =================================================

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": str(answer),
                        "local_sources": local_results,
                        "web_sources": web_results
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
                    {
                        "role": "assistant",
                        "content": error_message
                    }
                )