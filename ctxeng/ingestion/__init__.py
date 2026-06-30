"""Multi-modal file ingestion.

* ``TextIngestor`` / ``MarkdownIngestor`` — chunked text and markdown files
* ``ImageIngestor`` — metadata extraction + optional BLIP captioning
* ``CSVIngestor`` / ``JSONIngestor`` — structured data to natural-language descriptions
* ``PDFIngestor`` — PDF text extraction via PyPDF2
* ``FileIngestor`` — unified dispatcher by file extension (with optional python-magic MIME detection)
"""
