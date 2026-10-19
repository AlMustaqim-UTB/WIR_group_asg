"""
Prerequisite: Install python 3.8+
Language: Python 3.8+
How to run:
    python bm25_retrieval.py
No external dependencies required (uses standard library only).
"""

from collections import Counter, defaultdict  
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

def load_corpus() -> dict[str, str]:
    """
    Loads the document corpus from a JSON file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
    corpus_path = os.path.join(script_dir, "doc_corpus.json")  # Construct file path

    with open(corpus_path, "r", encoding="utf-8") as file:
        corpus = json.load(file)

    # Validate that corpus is a dictionary with string keys and string values
    if not isinstance(corpus, dict) or not all(
        isinstance(doc_id, str) and isinstance(text, str)
        for doc_id, text in corpus.items()
    ):
        raise ValueError(
            "doc_corpus.json must contain an object mapping document IDs to text."
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

        # Build result list with document ID, score, and snippet
        results = []
        for doc_id, score in ranked:
            snippet = self._generate_snippet(self.corpus[doc_id], query_terms)  # Extract snippet around query terms
            results.append((doc_id, score, snippet))

        return results

    def _generate_snippet(self, text: str, query_terms: list[str] = None) -> str:
        #Generates a snippet/preview of the document for display in search results.
        return text.strip()  


# MAIN PROGRAM 

if __name__ == "__main__":
    # Load the document corpus
    sample_corpus = load_corpus()

    test_queries = [
        "boolean index logic",  
        "retrieval-augmented generation question answering",  
        "visual recognition photo search",  
        "systematic literature survey classic search assessment", 
        "learning models",  
        "quick brown fox dog"  
    ]

    index = BM25Index(k1=1.2, b=0.75)
    
    # Build the index from the corpus
    # This populates: inverted index, document lengths, IDF weights
    index.fit(sample_corpus)

    # Execute each test query and display results
    for query in test_queries:
        print("-" * 80)  
        print(f"QUERY: '{query}'")
        print("-" * 80)

        # Search for top 3 most relevant documents
        results = index.search(query, top_k=3)
        
        # Display ranked results
        for rank, (doc_id, score, snippet) in enumerate(results, start=1):
            print(f"Rank {rank}:")
            print(f"  Doc ID  : {doc_id}") 
            print(f"  Score   : {score:.4f}") 
            print(f"  Snippet : {snippet}") 
        print() 