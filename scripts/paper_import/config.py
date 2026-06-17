"""Constants and configuration for the batch paper importer.

All magic values (endpoints, defaults, retry budgets, source/conference
mapping tables) live here so the rest of the package reads as logic, not data.
"""

from pathlib import Path

# Repo root is three levels up: paper_import/ -> scripts/ -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# --- External endpoints -----------------------------------------------------
ARXIV_API_URL = "http://export.arxiv.org/api/query"
OPENREVIEW_NOTES_URL = "https://api2.openreview.net/notes"
ACL_BASE_URL = "https://aclanthology.org"
CVF_BASE_URL = "https://openaccess.thecvf.com"
NEURIPS_BASE_URL = "https://proceedings.neurips.cc"
PMLR_BASE_URL = "https://proceedings.mlr.press"
AAAI_BASE_URL = "https://ojs.aaai.org"
AAAI_ARCHIVE_URL = f"{AAAI_BASE_URL}/index.php/AAAI/issue/archive"
IJCAI_BASE_URL = "https://www.ijcai.org"
ICLR_BASE_URL = "https://iclr.cc"
COLM_BASE_URL = "https://colmweb.org"

# --- HTTP behaviour ---------------------------------------------------------
DEFAULT_USER_AGENT = "knowledge-base-paper-importer/1.0"
REQUEST_TIMEOUT = 60
REQUEST_RETRY_TOTAL = 3
REQUEST_RETRY_BACKOFF = 1.0

# --- Search / import retry budgets ------------------------------------------
SOURCE_SEARCH_RETRIES = 2
SOURCE_SEARCH_RETRY_SLEEP = 2.0
PDF_DOWNLOAD_VALIDATION_RETRIES = 1
DEFAULT_IMPORT_RETRIES = 2
DEFAULT_IMPORT_BATCH_SIZE = 5

# --- Output -----------------------------------------------------------------
DEFAULT_IMPORT_ROOT = PROJECT_ROOT / "output" / "batch_imports"

# --- CLI source choices -----------------------------------------------------
CLI_SOURCE_CHOICES = [
    "all",
    "arxiv",
    "url_file",
    "openreview",
    "acl",
    "cvf",
    "neurips",
    "icml",
    "aaai",
    "ijcai",
]

# Default conference years used when --year is not supplied.
DEFAULT_SOURCE_YEARS = {
    "neurips": 2024,
    "icml": 2024,
    "aaai": 2025,
    "ijcai": 2024,
}

# Order (and membership) of sources searched in the multi-source `all` mode.
DEFAULT_MULTI_SOURCE_ORDER = ["arxiv", "neurips", "icml", "aaai", "ijcai"]

# Sources whose searcher takes a `year` keyword argument.
YEAR_BASED_SOURCES = {"neurips", "icml", "aaai", "ijcai", "iclr_virtual", "colm_official"}

# Conference shorthands accepted by --conference.
CONFERENCE_SOURCE_CHOICES = [
    "neurips",
    "icml",
    "aaai",
    "ijcai",
    "cvpr",
    "iccv",
    "eccv",
    "wacv",
    "acl",
    "emnlp",
    "naacl",
    "iclr",
    "colm",
]

# Known ICML year -> PMLR volume mapping (cache; discovery fills the rest).
ICML_YEAR_TO_VOLUME = {
    2024: 235,
    2025: 267,
}

# Sources that require an extra CLI field before they can run directly.
DIRECT_SOURCE_REQUIRED_FIELDS = {
    "url_file": "url_file",
    "openreview": "openreview_venueid",
    "acl": "acl_event",
    "cvf": "cvf_event",
}

# Conference shorthand -> underlying source + parameter templates.
CONFERENCE_TARGET_RULES = {
    "NeurIPS": {"source": "neurips", "uses_year": True},
    "ICML": {"source": "icml", "uses_year": True},
    "AAAI": {"source": "aaai", "uses_year": True},
    "IJCAI": {"source": "ijcai", "uses_year": True},
    "CVPR": {"source": "cvf", "params": {"event": "CVPR{year}"}},
    "ICCV": {"source": "cvf", "params": {"event": "ICCV{year}"}},
    "ECCV": {"source": "cvf", "params": {"event": "ECCV{year}"}},
    "WACV": {"source": "cvf", "params": {"event": "WACV{year}"}},
    "ACL": {"source": "acl", "params": {"event": "acl-{year}"}},
    "EMNLP": {"source": "acl", "params": {"event": "emnlp-{year}"}},
    "NAACL": {"source": "acl", "params": {"event": "naacl-{year}"}},
    "ICLR": {"source": "iclr_virtual", "uses_year": True},
    "COLM": {"source": "colm_official", "uses_year": True},
}

# Maps a source to (searcher_kwarg_name, argparse_attr_name) for option fields.
SOURCE_OPTION_FIELDS = {
    "openreview": ("venueid", "openreview_venueid"),
    "acl": ("event", "acl_event"),
    "cvf": ("event", "cvf_event"),
}
