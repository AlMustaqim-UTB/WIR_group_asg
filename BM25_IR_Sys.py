"""
Prerequisite: Install python 3.8+
Language: Python 3.8+
How to run:
    python bm25_retrieval.py
No external dependencies required (uses standard library only).
"""

from collections import Counter, defaultdict  
from html.parser import HTMLParser
import math  
import re  
import json  
import os  

# UTILITY FUNCTIONS FOR DATA LOADING
def load_stop_words():
    """
    Loads common English stop words from a JSON file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
    json_path = os.path.join(script_dir, "stop_words_english.json")  # Construct file path
    with open(json_path, "r", encoding="utf-8") as f:
        return set(json.load(f))  # Convert JSON list to set for efficient membership testing


# Global variable storing stop words for reuse across the program
STOP_WORDS = load_stop_words()

class ArticleTextParser(HTMLParser):
    """Extract visible text from the main article content of an HTML page."""

    CONTENT_ID = "mw-content-text"
    IGNORED_TAGS = {"script", "style", "noscript", "template", "svg"}
    BLOCK_TAGS = {
        "article",
        "br",
        "div",
        "figcaption",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "p",
        "section",
        "table",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.content_depth = 0
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)

        if self.content_depth == 0 and (
            tag == "div" and attributes.get("id") == self.CONTENT_ID
        ):
            self.content_depth = 1
            return

        if self.content_depth == 0:
            return

        if tag in self.IGNORED_TAGS:
            self.ignored_depth += 1
            return

        if self.ignored_depth == 0 and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if self.content_depth > 0 and self.ignored_depth == 0 and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.content_depth == 0:
            return

        if tag in self.IGNORED_TAGS and self.ignored_depth > 0:
            self.ignored_depth -= 1
            return

        if self.ignored_depth == 0 and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

        if tag == "div" and self.content_depth == 1:
            self.content_depth = 0

    def handle_data(self, data: str) -> None:
        if self.content_depth > 0 and self.ignored_depth == 0:
            text = " ".join(data.split())
            if text:
                self.parts.append(text)

    def get_text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self.parts)).strip()


def extract_html_text(html: str) -> str:
    """Extract searchable article text from one crawled HTML document."""
    parser = ArticleTextParser()
    parser.feed(html)
    parser.close()
    return parser.get_text()

def load_corpus() -> dict[str, str]:
    """
    Loads and extracts searchable text from HTML files in crawled_documents.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    documents_dir = os.path.join(script_dir, "crawled_documents")

    if not os.path.isdir(documents_dir):
        raise FileNotFoundError(
            f"Crawled documents directory was not found: {documents_dir}"
        )

    corpus: dict[str, str] = {}

    for filename in sorted(os.listdir(documents_dir)):
        if not filename.lower().endswith((".html", ".htm")):
            continue

        document_path = os.path.join(documents_dir, filename)

        with open(document_path, "r", encoding="utf-8") as file:
            text = extract_html_text(file.read())

        if not text:
            print(f"Skipping {filename}: no searchable article text found.")
            continue

        document_id = os.path.splitext(filename)[0]
        corpus[document_id] = text

    if not corpus:
        raise ValueError(
            f"No HTML documents were found in {documents_dir}"
        )

    return corpus


def preprocess(text: str) -> list[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())  # Extract word tokens and lowercase
    return [token for token in tokens if token not in STOP_WORDS]  # Remove stop words


# BM25 INFORMATION RETRIEVAL INDEX
class BM25Index:

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """
        Initializes BM25 index with tuning parameters.
        """
        self.k1 = k1
        self.b = b
        # Maps doc_id -> original unprocessed text
        self.corpus: dict[str, str] = {}
        # Maps doc_id -> token count (after preprocessing)            
        self.doc_lengths: dict[str, int] = {}  
        # Average document length across entire corpus (used for normalization)      
        self.avg_doc_len: float = 0.0  
        # Total number of documents (N) in corpus              
        self.doc_count: int = 0   
        # Structure: term -> {doc_id: term_frequency}                   
        self.inverted_index: defaultdict = defaultdict(dict)  
        # Maps each term -> its IDF score (ln-based weighting)
        self.idf: dict[str, float] = {}              

    def fit(self, documents: dict[str, str]) -> None:
        """
        Builds the BM25 index from a corpus of documents.
        
        1. Build the inverted index (term -> {doc_id: frequency})
        2. Calculate document length statistics
        3. Compute IDF weights for all terms
        """
        self.corpus = documents  # Store raw documents for later snippet generation
        self.doc_count = len(documents)  # Total number of documents (N)

        total_length = 0  # Accumulator for average document length calculation
        df: Counter = Counter()  # Tracks document frequency (how many docs contain each term)

        # STEP 1: Process all documents and build inverted index
        for doc_id, text in documents.items():
            tokens = preprocess(text)  # Tokenize and remove stop words
            self.doc_lengths[doc_id] = len(tokens)  # Store token count for this document
            total_length += len(tokens)  # Accumulate for average calculation

            # Count term frequencies within this document
            tf = Counter(tokens)  # {term: frequency_in_doc}
            for term, freq in tf.items():
                self.inverted_index[term][doc_id] = freq  # Store term freq for this doc
                df[term] += 1  # Increment document frequency counter

        # Calculate average document length (used for length normalization in ranking)
        self.avg_doc_len = total_length / self.doc_count if self.doc_count > 0 else 0.0

        # STEP 2: Calculate IDF (Inverse Document Frequency) for each term
        for term, doc_freq in df.items():
            # Robertson-Spärck Jones IDF formula (standard BM25)
            self.idf[term] = math.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5))

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float, str]]:
        """
        Searches the index using a query and returns top-k ranked documents.
        
        BM25 Scoring Process:
        1. Preprocess query to extract terms
        2. For each query term, retrieve its IDF and posting list (docs containing it)
        3. For each matching document, calculate BM25 score component
        4. Aggregate scores across all query terms
        5. Return top-k documents with highest scores
        """
        query_terms = preprocess(query)  # Tokenize query and remove stop words
        scores: defaultdict = defaultdict(float)  # Will accumulate BM25 scores per document

        # Process each term in the query
        for term in query_terms:
            if term not in self.inverted_index:
                # Skip query terms not in corpus (they don't help ranking)
                continue

            term_idf = self.idf[term]  # Get precomputed IDF weight for this term
            postings = self.inverted_index[term]  # Get {doc_id: term_freq} for this term

            # Score this term across all documents that contain it
            for doc_id, tf in postings.items():
                # Length Normalization Factor
                doc_len = self.doc_lengths[doc_id]
                len_norm = 1.0 - self.b + self.b * (doc_len / self.avg_doc_len)

                # Term Frequency Component with Saturation
                tf_component = (tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm)

                # Accumulate the BM25 component score for this term in this document
                scores[doc_id] += term_idf * tf_component

        # Sort documents by descending BM25 score and keep only top_k
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

        return ranked


# MAIN PROGRAM 

if __name__ == "__main__":
    # Load the document corpus
    sample_corpus = load_corpus()

    test_queries = [
    ("Q1", "minecraft game java"),
    ("Q2", "smelting iron crafting the item"),
    ("Q3", "mob enemy skeleton and ender dragon"),

]

    index = BM25Index(k1=1.2, b=0.75)
    
    # Build the index from the corpus
    # This populates: inverted index, document lengths, IDF weights
    index.fit(sample_corpus)

    # Execute each test query and display results
    for query_index, query in test_queries:
        print("-" * 80)
        print(f"QUERY INDEX: {query_index}")
        print(f"QUERY      : '{query}'")
        print("-" * 80)

        # Search for top 10 most relevant documents
        results = index.search(query, top_k=10)

        # Display ranked results
        for rank, (doc_id, score) in enumerate(results, start=1):
            print(f"Rank {rank}:")
            print(f"  Doc ID  : {doc_id}")
            print(f"  Score   : {score:.4f}")
        print()