"""
preprocessing.py
----------------
Reusable preprocessing pipeline for the Domain-Adaptive Research Paper
Summarization System.

Covers:
    - Text cleaning (regex + spaCy)
    - Section extraction (regex heuristics)
    - Sentence segmentation (spaCy)
    - Keyword extraction (KeyBERT + YAKE)
    - Stopword removal (NLTK + spaCy)
    - Named Entity Recognition (spaCy NER)

All functions are stateless and accept plain strings.
Call build_pipeline() once to load heavy models, then pass the returned
pipeline object into functions that need it.
"""

import re
import string
from typing import Optional

import nltk
import spacy
import yake
from keybert import KeyBERT
from nltk.corpus import stopwords

# ── Ensure NLTK data is available ─────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)

# ── Section header patterns ────────────────────────────────────────────────────
# Matches common section headings in research papers (case-insensitive)
SECTION_PATTERNS = {
    "abstract"    : r"\b(abstract)\b",
    "introduction": r"\b(introduction|background)\b",
    "methodology" : r"\b(method(ology)?s?|approach|proposed\s+method|materials?\s+and\s+methods?)\b",
    "results"     : r"\b(results?|findings?|experiments?|experimental\s+results?)\b",
    "discussion"  : r"\b(discussion|analysis|interpretation)\b",
    "conclusion"  : r"\b(conclusion[s]?|summary|closing\s+remarks?|future\s+work)\b",
}

# ── LaTeX / noise patterns to strip ───────────────────────────────────────────
LATEX_PATTERNS = [
    r"\$[^$]*\$",               # inline math: $...$
    r"\$\$[^$]*\$\$",           # display math: $$...$$
    r"\\[a-zA-Z]+\{[^}]*\}",   # LaTeX commands: \command{...}
    r"\\[a-zA-Z]+",             # bare LaTeX commands: \command
    r"\{[^}]*\}",               # leftover braces
    r"https?://\S+",            # URLs
    r"\[\d+\]",                 # citation markers: [1], [23]
    r"\(\d{4}\)",               # year markers: (2021)
    r"et\s+al\.",               # "et al."
    r"fig\.\s*\d+",             # figure references
    r"table\s*\d+",             # table references
    r"equation\s*\d+",          # equation references
]

COMPILED_LATEX = [re.compile(p, re.IGNORECASE) for p in LATEX_PATTERNS]


# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline loader
# ══════════════════════════════════════════════════════════════════════════════

class PreprocessingPipeline:
    """
    Container for all heavy NLP models.
    Load once with build_pipeline(), reuse across many documents.

    Attributes:
        nlp: spaCy language model (en_core_web_sm).
        kw_model: KeyBERT model for embedding-based keyword extraction.
        yake_extractor: YAKE keyword extractor instance.
        stop_words: Set of English stopwords from NLTK.
    """

    def __init__(self, nlp, kw_model, yake_extractor, stop_words):
        self.nlp            = nlp
        self.kw_model       = kw_model
        self.yake_extractor = yake_extractor
        self.stop_words     = stop_words


def build_pipeline(keybert_model: str = "all-MiniLM-L6-v2") -> PreprocessingPipeline:
    """
    Load all NLP models and return a PreprocessingPipeline instance.

    Call this once at the start of your script or notebook. Reuse the
    returned object — do not call this inside a loop.

    Args:
        keybert_model (str): Sentence-transformers model name for KeyBERT.

    Returns:
        PreprocessingPipeline: Loaded pipeline ready for use.
    """
    print("Loading spaCy model ...")
    nlp = spacy.load("en_core_web_sm")

    print("Loading KeyBERT model ...")
    kw_model = KeyBERT(model=keybert_model)

    print("Loading YAKE extractor ...")
    yake_extractor = yake.KeywordExtractor(
        lan="en",
        n=2,          # max n-gram size
        dedupLim=0.7, # deduplication threshold
        top=10,       # top K keywords
    )

    stop_words = set(stopwords.words("english"))

    print("Pipeline ready.")
    return PreprocessingPipeline(nlp, kw_model, yake_extractor, stop_words)


# ══════════════════════════════════════════════════════════════════════════════
#  1. Text Cleaning
# ══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """
    Clean raw text by removing LaTeX, URLs, citations, and normalising whitespace.

    Steps:
        1. Strip LaTeX math and commands
        2. Remove citation markers, URLs, figure/table references
        3. Remove non-ASCII characters
        4. Normalise whitespace and punctuation

    Args:
        text (str): Raw input text (e.g. from PDF or dataset field).

    Returns:
        str: Cleaned text string.
    """
    if not text or not text.strip():
        return ""

    # Apply all LaTeX / noise regex patterns
    for pattern in COMPILED_LATEX:
        text = pattern.sub(" ", text)

    # Remove non-ASCII characters
    text = text.encode("ascii", errors="ignore").decode("ascii")

    # Collapse multiple spaces / newlines
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


# ══════════════════════════════════════════════════════════════════════════════
#  2. Section Extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_sections(text: str) -> dict:
    """
    Extract named sections from a research paper using regex heuristics.

    Looks for common section headings and slices the text between them.
    Returns whatever sections are found; missing sections get an empty string.

    Args:
        text (str): Cleaned full paper text.

    Returns:
        dict: Keys are section names (abstract, introduction, methodology,
              results, discussion, conclusion). Values are extracted text strings.
    """
    sections = {name: "" for name in SECTION_PATTERNS}

    # Find all heading positions in the text
    heading_positions = []
    for section_name, pattern in SECTION_PATTERNS.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            # Only treat as heading if it appears at start of line or after newline
            start = match.start()
            preceding = text[max(0, start - 2): start]
            if preceding.strip() == "" or start == 0:
                heading_positions.append((start, section_name))

    # Sort by position in the document
    heading_positions.sort(key=lambda x: x[0])

    # Slice text between consecutive headings
    for i, (pos, name) in enumerate(heading_positions):
        next_pos = (
            heading_positions[i + 1][0]
            if i + 1 < len(heading_positions)
            else len(text)
        )
        # Skip the heading word itself (move past it)
        content_start = pos + len(name) + 1
        sections[name] = text[content_start:next_pos].strip()

    # If no sections found, treat entire text as the body
    if all(v == "" for v in sections.values()):
        sections["introduction"] = text

    return sections


def get_body_text(sections: dict, preferred: Optional[list] = None) -> str:
    """
    Return the most informative section text for summarization input.

    Tries sections in priority order and returns the first non-empty one,
    falling back to concatenating all sections.

    Args:
        sections (dict): Output of extract_sections().
        preferred (list, optional): Priority order of section names.
                                    Defaults to methodology → results → introduction.

    Returns:
        str: Best available body text for summarization.
    """
    if preferred is None:
        preferred = ["methodology", "results", "introduction", "discussion", "conclusion"]

    for name in preferred:
        if sections.get(name, "").strip():
            return sections[name]

    # Fallback: concatenate all non-empty sections
    return " ".join(v for v in sections.values() if v.strip())


# ══════════════════════════════════════════════════════════════════════════════
#  3. Sentence Segmentation
# ══════════════════════════════════════════════════════════════════════════════

def segment_sentences(text: str, pipeline: PreprocessingPipeline) -> list:
    """
    Split text into sentences using spaCy's sentence boundary detection.

    Args:
        text (str): Cleaned input text.
        pipeline (PreprocessingPipeline): Loaded pipeline from build_pipeline().

    Returns:
        list[str]: List of sentence strings.
    """
    if not text.strip():
        return []

    # spaCy has a token limit; truncate very long texts to avoid memory issues
    MAX_CHARS = 100_000
    doc = pipeline.nlp(text[:MAX_CHARS])
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


# ══════════════════════════════════════════════════════════════════════════════
#  4. Stopword Removal
# ══════════════════════════════════════════════════════════════════════════════

def remove_stopwords(text: str, pipeline: PreprocessingPipeline) -> str:
    """
    Remove English stopwords from text using NLTK stopword list.

    Args:
        text (str): Input text.
        pipeline (PreprocessingPipeline): Loaded pipeline from build_pipeline().

    Returns:
        str: Text with stopwords removed.
    """
    tokens = text.lower().split()
    filtered = [
        t for t in tokens
        if t not in pipeline.stop_words
        and t not in string.punctuation
        and len(t) > 1
    ]
    return " ".join(filtered)


# ══════════════════════════════════════════════════════════════════════════════
#  5. Keyword Extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_keywords_keybert(
    text: str,
    pipeline: PreprocessingPipeline,
    top_n: int = 10,
    keyphrase_ngram_range: tuple = (1, 2),
) -> list:
    """
    Extract keywords using KeyBERT (sentence-transformer embeddings).

    KeyBERT finds keywords by measuring cosine similarity between
    candidate n-gram embeddings and the document embedding.

    Args:
        text (str): Input text (cleaned, ideally without stopwords).
        pipeline (PreprocessingPipeline): Loaded pipeline from build_pipeline().
        top_n (int): Number of keywords to return.
        keyphrase_ngram_range (tuple): Min and max n-gram size for candidates.

    Returns:
        list[tuple]: List of (keyword, score) tuples, sorted by relevance.
    """
    if not text.strip():
        return []

    keywords = pipeline.kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=keyphrase_ngram_range,
        stop_words="english",
        top_n=top_n,
    )
    return keywords


def extract_keywords_yake(
    text: str,
    pipeline: PreprocessingPipeline,
    top_n: int = 10,
) -> list:
    """
    Extract keywords using YAKE (Yet Another Keyword Extractor).

    YAKE is a statistical, unsupervised method — no embeddings needed.
    Lower score = more relevant in YAKE's scoring scheme.

    Args:
        text (str): Input text.
        pipeline (PreprocessingPipeline): Loaded pipeline from build_pipeline().
        top_n (int): Number of keywords to return.

    Returns:
        list[tuple]: List of (keyword, score) tuples.
    """
    if not text.strip():
        return []

    keywords = pipeline.yake_extractor.extract_keywords(text)
    return keywords[:top_n]


def extract_keywords_combined(
    text: str,
    pipeline: PreprocessingPipeline,
    top_n: int = 10,
) -> list:
    """
    Combine KeyBERT and YAKE keywords into a single deduplicated list.

    KeyBERT contributes semantic relevance; YAKE contributes statistical
    salience. Union gives broader coverage.

    Args:
        text (str): Input text.
        pipeline (PreprocessingPipeline): Loaded pipeline from build_pipeline().
        top_n (int): Total keywords to return from combined list.

    Returns:
        list[str]: Deduplicated list of keyword strings.
    """
    kb_keywords   = [kw for kw, _ in extract_keywords_keybert(text, pipeline, top_n=top_n)]
    yake_keywords = [kw for kw, _ in extract_keywords_yake(text, pipeline, top_n=top_n)]

    # Deduplicate while preserving order (KeyBERT first)
    seen = set()
    combined = []
    for kw in kb_keywords + yake_keywords:
        kw_lower = kw.lower()
        if kw_lower not in seen:
            seen.add(kw_lower)
            combined.append(kw)

    return combined[:top_n]


# ══════════════════════════════════════════════════════════════════════════════
#  6. Named Entity Recognition
# ══════════════════════════════════════════════════════════════════════════════

def extract_entities(text: str, pipeline: PreprocessingPipeline) -> dict:
    """
    Extract named entities from text using spaCy NER.

    Entity types of interest for research papers:
        - ORG:     organisations, institutions
        - PERSON:  author names, researchers
        - GPE:     countries, cities
        - DATE:    years, time periods
        - NORP:    nationalities, groups
        - PRODUCT: model names, tools

    Args:
        text (str): Cleaned input text.
        pipeline (PreprocessingPipeline): Loaded pipeline from build_pipeline().

    Returns:
        dict: Keys are entity type labels, values are lists of entity strings.
    """
    if not text.strip():
        return {}

    MAX_CHARS = 100_000
    doc = pipeline.nlp(text[:MAX_CHARS])

    entities: dict = {}
    for ent in doc.ents:
        label = ent.label_
        if label not in entities:
            entities[label] = []
        if ent.text.strip() not in entities[label]:
            entities[label].append(ent.text.strip())

    return entities


# ══════════════════════════════════════════════════════════════════════════════
#  7. Full Preprocessing — single entry point
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(text: str, pipeline: PreprocessingPipeline) -> dict:
    """
    Run the full preprocessing pipeline on a single document.

    Steps in order:
        1. Clean text
        2. Extract sections
        3. Segment sentences
        4. Remove stopwords (on cleaned text)
        5. Extract keywords (KeyBERT + YAKE combined)
        6. Extract named entities

    Args:
        text (str): Raw input text (article field from dataset).
        pipeline (PreprocessingPipeline): Loaded pipeline from build_pipeline().

    Returns:
        dict with keys:
            - cleaned_text  (str)
            - sections      (dict)
            - sentences     (list[str])
            - no_stopwords  (str)
            - keywords      (list[str])
            - entities      (dict)
    """
    cleaned    = clean_text(text)
    sections   = extract_sections(cleaned)
    sentences  = segment_sentences(cleaned, pipeline)
    no_sw      = remove_stopwords(cleaned, pipeline)
    keywords   = extract_keywords_combined(cleaned, pipeline, top_n=10)
    entities   = extract_entities(cleaned, pipeline)

    return {
        "cleaned_text" : cleaned,
        "sections"     : sections,
        "sentences"    : sentences,
        "no_stopwords" : no_sw,
        "keywords"     : keywords,
        "entities"     : entities,
    }