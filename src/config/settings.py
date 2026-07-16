"""Central configuration values for ScholarAI."""

from dotenv import load_dotenv

load_dotenv()

# How large each text chunk should be (in characters).
CHUNK_SIZE = 4000

# How much overlap to keep between chunks so context is not lost at boundaries.
CHUNK_OVERLAP = 200

# Maximum characters indexed for the vector database (longer PDFs are trimmed).
INDEXING_CHAR_LIMIT = 50000

# Maximum characters sent to Gemini for full-paper analysis.
MAX_ANALYSIS_CHARS = 100000

# How many of the most relevant chunks to retrieve for each question.
TOP_K_CHUNKS = 4

APP_MODEL = "Gemini 2.5 Flash"
