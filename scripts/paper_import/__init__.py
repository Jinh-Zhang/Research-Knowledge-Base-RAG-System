"""Paper-import package.

Splits the original monolithic ``batch_import_papers.py`` into focused modules:

- ``config``       constants, endpoints, source/conference mapping tables
- ``text_utils``   pure string helpers (whitespace, HTML, query matching)
- ``http_session`` requests.Session construction + network-error classification
- ``metadata``     paper-record building and metadata enrichment
- ``crawlers``     one module per source, registered via a decorator
- ``pdf``          PDF download + validation
- ``manifest``     resume manifest read/write and paths
- ``pipeline``     import execution, download staging, batch import loop
- ``collect``      source resolution and paper collection orchestration
- ``cli``          argparse parser + validation
"""
