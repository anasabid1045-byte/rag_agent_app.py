import os
import re
import glob
import pandas as pd
import fitz
import pytesseract
from PIL import Image
from langchain_core.documents import Document
from dotenv import load_dotenv
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

# Identify web requests
os.environ.setdefault(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/153.0 Safari/537.36"
)

# ============================================================
# LANGCHAIN IMPORTS
# ============================================================

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    WebBaseLoader,
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = BASE_DIR

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Change this model only if your installed Groq account/model
# requires another currently available model.
GROQ_MODEL = "openai/gpt-oss-20b"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

TOP_K_LOCAL = 10
TOP_K_WEB = 5

# ============================================================
# CHECK GROQ API KEY
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("\nERROR: GROQ_API_KEY is not set.")
    print("Set it in your .env file or PowerShell environment.")
    print()
    print("Example .env:")
    print("GROQ_API_KEY=your_api_key_here")
    raise SystemExit(1)

# ============================================================
# INITIALIZE LLM
# ============================================================

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0.2,
    api_key=GROQ_API_KEY,
)

# ============================================================
# EMBEDDINGS
# ============================================================

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

print("Embedding model loaded successfully.")

# ============================================================
# TEXT SPLITTER
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        " ",
        "",
    ],
)

# ============================================================
# FILE LOADERS
# ============================================================

def load_pdf(file_path):
    """Load PDF text. Use cached OCR so scanned PDFs are not OCR'd every time."""

    try:
        print(f"\nLoading PDF: {os.path.basename(file_path)}")

        # ---------------------------------------------------------
        # 1. CHECK OCR CACHE FIRST
        # ---------------------------------------------------------
        cache_file = file_path + ".ocr.txt"

        if os.path.exists(cache_file):
            print("  OCR cache found.")
            print("  Loading saved OCR text...")
            
            ocr_docs = []

            with open(cache_file, "r", encoding="utf-8") as f:
                content = f.read()

            pages = content.split("\n\n===== PAGE ")

            for page_data in pages:
                if not page_data.strip():
                    continue

                try:
                    page_number_text, text = page_data.split(" =====\n", 1)

                    page_number = int(
                        page_number_text.strip()
                    )

                    text = text.strip()

                    if text:
                        ocr_docs.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source_type": "PDF_OCR",
                                    "source_file": os.path.basename(file_path),
                                    "page": page_number
                                }
                            )
                        )

                except Exception:
                    continue

            print(
                f"  Loaded {len(ocr_docs)} pages from OCR cache."
            )

            return ocr_docs

        # ---------------------------------------------------------
        # 2. TRY NORMAL PDF TEXT EXTRACTION
        # ---------------------------------------------------------
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        total_text = sum(
            len(doc.page_content.strip())
            for doc in docs
        )

        print(
            f"  Normal PDF text extracted: "
            f"{total_text} characters"
        )

        if total_text >= 10000:

            for doc in docs:
                doc.metadata["source_type"] = "PDF"
                doc.metadata["source_file"] = os.path.basename(file_path)

            print(
                f"  PDF loaded normally: {len(docs)} pages"
            )

            return docs

        # ---------------------------------------------------------
        # 3. OCR REQUIRED
        # ---------------------------------------------------------
        print("  Very little text found.")
        print("  This appears to be a scanned/image-based PDF.")
        print("  Starting OCR...")

        ocr_docs = []

        pdf = fitz.open(file_path)
        total_pages = len(pdf)

        print(f"  Total pages to OCR: {total_pages}")

        for page_number in range(total_pages):

            page = pdf[page_number]

            matrix = fitz.Matrix(2, 2)

            pix = page.get_pixmap(
                matrix=matrix,
                alpha=False
            )

            image = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )

            text = pytesseract.image_to_string(
                image,
                lang="eng",
                config="--oem 3 --psm 6"
            )

            text = text.strip()

            if text:
                ocr_docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source_type": "PDF_OCR",
                            "source_file": os.path.basename(file_path),
                            "page": page_number + 1
                        }
                    )
                )

            print(
                f"  OCR page {page_number + 1}/{total_pages}"
            )

        pdf.close()

        # ---------------------------------------------------------
        # 4. SAVE OCR CACHE
        # ---------------------------------------------------------
        print("\n  Saving OCR cache...")

        with open(
            cache_file,
            "w",
            encoding="utf-8"
        ) as f:

            for doc in ocr_docs:

                page = doc.metadata.get("page", 0)

                f.write(
                    f"\n===== PAGE {page} =====\n"
                )

                f.write(
                    doc.page_content
                )

                f.write("\n")

        print(
            f"  OCR cache saved: "
            f"{os.path.basename(cache_file)}"
        )

        print(
            f"  OCR completed. "
            f"Useful pages: {len(ocr_docs)}/{total_pages}"
        )

        total_ocr_text = sum(
            len(doc.page_content)
            for doc in ocr_docs
        )

        print(
            f"  OCR extracted: "
            f"{total_ocr_text} characters"
        )

        return ocr_docs

    except Exception as e:

        print(
            f"Could not load PDF: "
            f"{os.path.basename(file_path)}"
        )

        print(f"Reason: {e}")

        return []


def load_docx(file_path):
    """Load Word document safely."""

    try:
        loader = Docx2txtLoader(file_path)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source_type"] = "WORD"
            doc.metadata["source_file"] = os.path.basename(file_path)

        return docs

    except Exception as e:
        print(f"Could not load Word file: {os.path.basename(file_path)}")
        print(f"Reason: {e}")
        return []


def load_excel(file_path):
    """
    Load Excel files using pandas.

    Supports .xlsx, .xls and .xlsm.
    Each worksheet becomes a LangChain Document.
    """

    documents = []

    try:
        excel_file = pd.ExcelFile(file_path)

        for sheet_name in excel_file.sheet_names:

            try:
                df = pd.read_excel(
                    file_path,
                    sheet_name=sheet_name
                )

                if df.empty:
                    continue

                # Convert NaN to empty strings
                df = df.fillna("")

                # Convert dataframe into readable text
                rows = []

                for _, row in df.iterrows():

                    values = []

                    for column in df.columns:
                        value = row[column]

                        if str(value).strip():
                            values.append(
                                f"{column}: {value}"
                            )

                    if values:
                        rows.append(" | ".join(values))

                if not rows:
                    continue

                content = (
                    f"Excel File: {os.path.basename(file_path)}\n"
                    f"Sheet: {sheet_name}\n\n"
                    + "\n".join(rows)
                )

                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source_type": "EXCEL",
                            "source_file": os.path.basename(file_path),
                            "sheet": sheet_name,
                        },
                    )
                )

            except Exception as sheet_error:
                print(
                    f"Could not read sheet '{sheet_name}' "
                    f"in {os.path.basename(file_path)}"
                )
                print(f"Reason: {sheet_error}")

        return documents

    except Exception as e:
        print(f"Could not load Excel: {os.path.basename(file_path)}")
        print(f"Reason: {e}")
        return []


# ============================================================
# LOAD ALL LOCAL FILES
# ============================================================

def load_all_local_documents():

    all_documents = []

    print("\n========== LOADING LOCAL DOCUMENTS ==========\n")

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    pdf_files = glob.glob(
        os.path.join(DATA_DIR, "*.pdf")
    )

    for file_path in pdf_files:


        docs = load_pdf(file_path)

        if docs:
            all_documents.extend(docs)
            print(
                f"  Loaded {len(docs)} pages successfully."
            )

    # --------------------------------------------------------
    # WORD
    # --------------------------------------------------------

    word_files = []

    word_files.extend(
        glob.glob(os.path.join(DATA_DIR, "*.docx"))
    )

    word_files.extend(
        glob.glob(os.path.join(DATA_DIR, "*.doc"))
    )

    for file_path in word_files:

        print(f"Loading Word: {os.path.basename(file_path)}")

        # docx2txt only supports .docx
        if file_path.lower().endswith(".docx"):

            docs = load_docx(file_path)

            if docs:
                all_documents.extend(docs)

                print(
                    f"  Loaded successfully."
                )

        else:

            print(
                "  Skipped .doc file. "
                "Please convert it to .docx."
            )

    # --------------------------------------------------------
    # EXCEL
    # --------------------------------------------------------

    excel_files = []

    excel_files.extend(
        glob.glob(os.path.join(DATA_DIR, "*.xlsx"))
    )

    excel_files.extend(
        glob.glob(os.path.join(DATA_DIR, "*.xls"))
    )

    excel_files.extend(
        glob.glob(os.path.join(DATA_DIR, "*.xlsm"))
    )

    for file_path in excel_files:

        print(f"Loading Excel: {os.path.basename(file_path)}")

        docs = load_excel(file_path)

        if docs:
            all_documents.extend(docs)

            print(
                f"  Loaded {len(docs)} sheets successfully."
            )

    return all_documents


# ============================================================
# WEB PAGE LOADER
# ============================================================

def load_web_page(url):

    try:

        print(f"\nLoading web page:\n{url}")

        loader = WebBaseLoader(
            web_paths=(url,)
        )

        docs = loader.load()

        for doc in docs:

            doc.metadata["source_type"] = "WEB"
            doc.metadata["source_url"] = url

        return docs

    except Exception as e:

        print(f"Could not load web page.")
        print(f"Reason: {e}")

        return []


# ============================================================
# CREATE VECTOR DATABASE
# ============================================================

def create_vector_database(documents):

    if not documents:
        print("\nNo local documents found.")
        return None

    print(
        f"\nSplitting {len(documents)} document sections..."
    )

    chunks = text_splitter.split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    print(
        "\nCreating FAISS vector database..."
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    print(
        "FAISS vector database created successfully."
    )

    return vectorstore


# ============================================================
# DUCKDUCKGO SEARCH
# ============================================================

def duckduckgo_search(query, max_results=5):

    try:

        from ddgs import DDGS

    except ImportError:

        print(
            "\nERROR: DDGS package is not installed."
        )

        print(
            "Run:"
        )

        print(
            "pip install -U ddgs"
        )

        return []

    results = []

    try:

        with DDGS() as ddgs:

            search_results = ddgs.text(
                query,
                max_results=max_results
            )

            for item in search_results:

                title = item.get("title", "")
                body = item.get("body", "")
                href = item.get("href", "")

                if title or body:

                    results.append(
                        {
                            "title": title,
                            "body": body,
                            "url": href,
                        }
                    )

        return results

    except Exception as e:

        print(
            f"Web search failed: {e}"
        )

        return []


# ============================================================
# QUERY CLASSIFICATION
# ============================================================

def needs_web_search(query):

    query_lower = query.lower()

    web_keywords = [

        # Current information
        "latest",
        "today",
        "current",
        "currently",
        "recent",
        "recently",
        "news",
        "update",
        "updates",

        # Time-sensitive
        "2026",
        "2027",

        # Web-oriented
        "website",
        "online",
        "internet",
        "search",
        "google",
        "price",
        "weather",

        # Universities / admissions
        "admission",
        "admissions",
        "deadline",
        "merit",
        "fee",
        "fees",

        # Jobs
        "job",
        "jobs",
        "vacancy",
        "vacancies",

        # Explicit web request
        "on the web",
        "search online",
    ]

    return any(
        keyword in query_lower
        for keyword in web_keywords
    )


# ============================================================
# LOCAL RAG SEARCH
# ============================================================

def search_local_documents(
    vectorstore,
    query,
    k=TOP_K_LOCAL
):
    """
    Professional hybrid local retrieval.

    Retrieval layers:
    1. FAISS semantic search
    2. Exact phrase matching
    3. Keyword matching
    4. Definition-pattern matching
    5. Source-aware ranking

    This improves retrieval for questions such as:
    "What is artificial intelligence according to the textbook?"
    """

    if vectorstore is None:
        return []

    try:
        # =====================================================
        # STEP 1: SEMANTIC SEARCH
        # =====================================================

        semantic_k = max(k * 4, 40)

        semantic_results = vectorstore.similarity_search(
            query,
            k=semantic_k
        )

        # =====================================================
        # STEP 2: GET ALL STORED CHUNKS
        # =====================================================

        all_docs = list(
            vectorstore.docstore._dict.values()
        )

          # =====================================================
        # STEP 3: CLEAN QUERY
        # =====================================================

        query_lower = query.lower().strip()

        # Remove common question words
        stop_words = {
            "what", "is", "are", "the", "a", "an",
            "of", "in", "on", "to", "for",
            "according", "my", "textbook",
            "define", "definition", "explain",
            "does", "do", "how", "why",
            "can", "could", "would",
            "please", "tell", "me",
            "from"
        }

        query_words = re.findall(
            r"\b[a-zA-Z]{3,}\b",
            query_lower
        )

        keywords = [
            word
            for word in query_words
            if word not in stop_words
        ]

        # =====================================================
        # STEP 4: DETECT IMPORTANT PHRASES
        # =====================================================

        phrases = []

        for i in range(len(keywords) - 1):

            phrase = (
                keywords[i]
                + " "
                + keywords[i + 1]
            )

            phrases.append(phrase)

        # =====================================================
        # STEP 5: DETECT DEFINITION QUESTION
        # =====================================================

        is_definition_question = any(
            phrase in query_lower
            for phrase in [
                "what is",
                "what are",
                "define",
                "definition of",
                "meaning of",
                "what does"
            ]
        )

        # =====================================================
        # STEP 6: SCORE ALL DOCUMENTS
        # =====================================================

        scored_documents = []

        for doc in all_docs:

            text = doc.page_content.lower()

            score = 0

            # -------------------------------------------------
            # A. EXACT PHRASE MATCH
            # -------------------------------------------------

            for phrase_index, phrase in enumerate(phrases):

                occurrences = len(
                    re.findall(
                        rf"\b{re.escape(phrase)}\b",
                        text
                    )
                )

                if occurrences:

                    # The first important phrase in the query
                    # receives the strongest priority.
                    if phrase_index == 0:
                        phrase_weight = 80
                    else:
                        phrase_weight = 20

                    score += min(occurrences, 5) * phrase_weight

            # -------------------------------------------------
            # B. INDIVIDUAL KEYWORD MATCH
            # -------------------------------------------------

            for keyword in keywords:

                occurrences = len(
                    re.findall(
                        rf"\b{re.escape(keyword)}\b",
                        text
                    )
                )

                if occurrences:

                    score += min(occurrences, 5) * 2

            # -------------------------------------------------
            # C. FULL QUERY MATCH
            # -------------------------------------------------

            if query_lower in text:

                score += 100

                        # -------------------------------------------------
            # D. DEFINITION PATTERNS
            # -------------------------------------------------

            if is_definition_question:

                definition_patterns = [

                    r"\bis\s+an?\b",

                    r"\bis\s+the\b",

                    r"\brefers\s+to\b",

                    r"\bdefined\s+as\b",

                    r"\bcan\s+be\s+defined\s+as\b",

                    r"\bmeans\b",

                    r"\bis\s+known\s+as\b",

                    r"\bis\s+called\b",

                ]

                # Only give a strong definition bonus when
                # the definition pattern occurs near the
                # main subject of the question.

                subject_phrase = ""

                if "artificial intelligence" in query_lower:

                    subject_phrase = "artificial intelligence"

                elif "computer science" in query_lower:

                    subject_phrase = "computer science"

                subject_found = False

                if subject_phrase:

                    subject_positions = [
                        match.start()
                        for match in re.finditer(
                            rf"\b{re.escape(subject_phrase)}\b",
                            text
                        )
                    ]

                    for position in subject_positions:

                        nearby_text = text[
                            max(0, position - 250):
                            position + 500
                        ]

                        for pattern in definition_patterns:

                            if re.search(
                                pattern,
                                nearby_text
                            ):

                                subject_found = True
                                break

                        if subject_found:
                            break

                if subject_found:

                    score += 100
            # -------------------------------------------------
            # E. TEXTBOOK SOURCE PRIORITY
            # -------------------------------------------------

            source_type = doc.metadata.get(
                "source_type",
                ""
            )

            if source_type == "PDF_OCR":

                score += 2

            elif source_type == "PDF":

                score += 2

            # -------------------------------------------------
            # KEEP RELEVANT DOCUMENTS
            # -------------------------------------------------

            if score > 0:

                scored_documents.append(
                    (score, doc)
                )

        # =====================================================
        # STEP 7: SORT LEXICAL RESULTS
        # =====================================================

        scored_documents.sort(
            key=lambda item: item[0],
            reverse=True
        )

        lexical_results = [
            doc
            for score, doc
            in scored_documents[:semantic_k]
        ]

               # =====================================================
        # STEP 8: COMBINE RESULTS
        # =====================================================

        combined = {}

        def document_key(doc):

            source = doc.metadata.get(
                "source_file",
                ""
            )

            page = doc.metadata.get(
                "page",
                ""
            )

            sheet = doc.metadata.get(
                "sheet",
                ""
            )

            content_start = doc.page_content[:150]

            return (
                source,
                str(page),
                str(sheet),
                content_start
            )

        # -----------------------------------------------------
        # Semantic results
        # -----------------------------------------------------

        for rank, doc in enumerate(
            semantic_results,
            start=1
        ):

            key = document_key(doc)

            combined[key] = {
                "doc": doc,
                "semantic_score": semantic_k - rank + 1,
                "lexical_score": 0
            }

        # -----------------------------------------------------
        # Lexical results
        # -----------------------------------------------------

        lexical_scores = {}

        for score, doc in scored_documents[:semantic_k]:

            key = document_key(doc)

            lexical_scores[key] = score

        for doc in lexical_results:

            key = document_key(doc)

            if key not in combined:

                combined[key] = {
                    "doc": doc,
                    "semantic_score": 0,
                    "lexical_score": 0
                }

            combined[key]["lexical_score"] = (
                lexical_scores.get(key, 0)
            )

               # =====================================================
        # STEP 9: FINAL HYBRID RANKING
        # =====================================================

        ranked_results = []

        for item in combined.values():

            doc = item["doc"]

            text = doc.page_content.lower()

            final_score = (
                item["semantic_score"]
                +
                item["lexical_score"] * 5
            )

            # -------------------------------------------------
            # EXACT SUBJECT PHRASE BOOST
            # -------------------------------------------------

            if "artificial intelligence" in query_lower:

                if re.search(
                    r"\bartificial\s+intelligence\b",
                    text
                ):

                    final_score += 300

            # -------------------------------------------------
            # DEFINITION CONTEXT BOOST
            # -------------------------------------------------

            if is_definition_question:

                if (
                    "artificial intelligence" in query_lower
                    and
                    "artificial intelligence" in text
                ):

                    nearby_match = re.search(
                        r"artificial\s+intelligence.{0,500}"
                        r"(refers\s+to|is\s+|means|defined)",
                        text
                    )

                    if nearby_match:

                        final_score += 500

            ranked_results.append(
                (
                    final_score,
                    doc
                )
            )

        ranked_results.sort(
            key=lambda item: item[0],
            reverse=True
        )

        # =====================================================
        # STEP 10: RETURN TOP K
        # =====================================================

        final_results = [
            doc
            for score, doc
            in ranked_results[:k]
        ]
        # =====================================================
        # DEBUG INFORMATION
        # =====================================================

        print("\nTop local retrieval results:")

        for index, doc in enumerate(
            final_results,
            start=1
        ):

            source = doc.metadata.get(
                "source_file",
                "Unknown"
            )

            page = doc.metadata.get(
                "page",
                ""
            )

            preview = (
                doc.page_content
                .replace("\n", " ")
                .strip()
            )

            preview = preview[:180]

            print(
                f"  {index}. {source}"
                f" | Page: {page}"
            )

            print(
                f"     {preview}..."
            )

        return final_results

    except Exception as e:

        print(
            f"Local document search failed: {e}"
        )

        return []


# ============================================================
# FORMAT LOCAL SOURCES
# ============================================================

def format_local_context(documents):

    if not documents:
        return "No relevant local documents were found."

    context_parts = []

    for index, doc in enumerate(documents, 1):

        source_type = doc.metadata.get(
            "source_type",
            "UNKNOWN"
        )

        source_file = doc.metadata.get(
            "source_file",
            "Unknown source"
        )

        sheet = doc.metadata.get(
            "sheet",
            ""
        )

        source_label = source_file

        if sheet:
            source_label += f" | Sheet: {sheet}"

        context_parts.append(
            f"""
--- LOCAL SOURCE {index} ---
Type: {source_type}
Source: {source_label}

Content:
{doc.page_content}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# FORMAT WEB SOURCES
# ============================================================

def format_web_context(results):

    if not results:
        return "No web search results were found."

    context_parts = []

    for index, result in enumerate(results, 1):

        context_parts.append(
            f"""
--- WEB SOURCE {index} ---
Title: {result.get("title", "")}
URL: {result.get("url", "")}

Content:
{result.get("body", "")}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# UNIFIED ANSWER GENERATION
# ============================================================

def generate_answer(
    query,
    local_documents,
    web_results
):

    local_context = format_local_context(
        local_documents
    )

    web_context = format_web_context(
        web_results
    )

    prompt = f"""
You are a professional Unified RAG and Web Search AI Agent.

You have access to TWO knowledge sources:

1. LOCAL DOCUMENTS
   - PDF
   - Word
   - Excel

2. WEB SEARCH
   - DuckDuckGo search results

Your job is to answer the user's question using the
most relevant information from BOTH sources when useful.

IMPORTANT RULES:

- Do not invent facts.
- Do not make up information that is not present in the sources.
- If local documents contain the answer, use them.
- If current information is required, use web results.
- If both sources are relevant, combine them.
- Clearly distinguish information from local documents and web sources
  when necessary.
- If sources disagree, mention the disagreement instead of hiding it.
- Give a direct answer first.
- Then give a concise explanation.
- For web information, include the source URL when available.
- Do not say that you searched the internet unless web results
  were actually provided.
- If there is insufficient information, say so clearly.

USER QUESTION:
{query}

============================================================
LOCAL DOCUMENT CONTEXT
============================================================

{local_context}

============================================================
WEB SEARCH CONTEXT
============================================================

{web_context}

============================================================
FINAL ANSWER
============================================================
"""

    try:

        response = llm.invoke(prompt)

        return response.content

    except Exception as e:

        return (
            "I could not generate the answer because "
            f"the AI model returned an error:\n{e}"
        )


# ============================================================
# UNIFIED AGENT
# ============================================================

def unified_agent(
    query,
    vectorstore
):

    print(
        "\nAnalyzing your query..."
    )

    # --------------------------------------------------------
    # ALWAYS SEARCH LOCAL DOCUMENTS
    # --------------------------------------------------------

    local_results = search_local_documents(
        vectorstore,
        query,
        TOP_K_LOCAL
    )

    # --------------------------------------------------------
    # DECIDE WHETHER WEB SEARCH IS NEEDED
    # --------------------------------------------------------

    web_results = []

    if needs_web_search(query):

        print(
            "Web search required → searching DuckDuckGo..."
        )

        web_results = duckduckgo_search(
            query,
            TOP_K_WEB
        )

    else:

        print(
            "No current-web requirement detected."
        )

    # --------------------------------------------------------
    # SHOW SOURCES USED
    # --------------------------------------------------------

    print(
        f"\nLocal results: {len(local_results)}"
    )

    print(
        f"Web results: {len(web_results)}"
    )

    # --------------------------------------------------------
    # GENERATE ONE COMBINED ANSWER
    # --------------------------------------------------------

    answer = generate_answer(
        query,
        local_results,
        web_results
    )

    print(
        "\n============================================================"
    )

    print(
        "                    FINAL ANSWER"
    )

    print(
        "============================================================\n"
    )

    print(answer)

    # --------------------------------------------------------
    # SOURCE LIST
    # --------------------------------------------------------

    if local_results:

        print(
            "\n------------------------------------------------------------"
        )

        print(
            "LOCAL SOURCES USED"
        )

        print(
            "------------------------------------------------------------"
        )

        seen = set()

        for doc in local_results:

            source = doc.metadata.get(
                "source_file",
                "Unknown"
            )

            if source not in seen:

                print(
                    f"- {source}"
                )

                seen.add(source)

    if web_results:

        print(
            "\n------------------------------------------------------------"
        )

        print(
            "WEB SOURCES"
        )

        print(
            "------------------------------------------------------------"
        )

        for result in web_results:

            print(
                f"- {result.get('title', 'Untitled')}"
            )

            print(
                f"  {result.get('url', '')}"
            )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print(
        "\n============================================================"
    )

    print(
        "        UNIFIED RAG + WEB SEARCH AI AGENT"
    )

    print(
        "============================================================"
    )

    print(
        "\nSupported sources:"
    )

    print(
        "  ✓ PDF"
    )

    print(
        "  ✓ Word (.docx)"
    )

    print(
        "  ✓ Excel (.xlsx / .xls / .xlsm)"
    )

    print(
        "  ✓ Web pages"
    )

    print(
        "  ✓ DuckDuckGo Search"
    )

    print(
        "  ✓ FAISS Vector Search"
    )

    print(
        "  ✓ Groq LLM"
    )

    print(
        "\nAll sources are handled by ONE unified agent."
    )

    # --------------------------------------------------------
    # LOAD LOCAL FILES
    # --------------------------------------------------------

    local_documents = load_all_local_documents()

    print(
        f"\nTotal local document sections loaded: "
        f"{len(local_documents)}"
    )

    # --------------------------------------------------------
    # CREATE VECTOR STORE
    # --------------------------------------------------------

    vectorstore = create_vector_database(
        local_documents
    )

    # --------------------------------------------------------
    # MAIN CHAT LOOP
    # --------------------------------------------------------

    print(
        "\n============================================================"
    )

    print(
        "              AGENT READY"
    )

    print(
        "============================================================"
    )

    print(
        "\nAsk anything."
    )

    print(
        "The agent will combine local RAG + web search when needed."
    )

    print(
        "Type 'exit' to close the program."
    )

    while True:

        try:

            query = input(
                "\nYou: "
            ).strip()

        except KeyboardInterrupt:

            print(
                "\n\nProgram closed."
            )

            break

        except EOFError:

            print(
                "\n\nProgram closed."
            )

            break

        if not query:
            continue

        if query.lower() in {
            "exit",
            "quit",
            "3",
        }:

            print(
                "\nGoodbye!"
            )

            break

        unified_agent(
            query,
            vectorstore
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()