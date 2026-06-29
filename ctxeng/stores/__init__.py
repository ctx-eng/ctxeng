"""Pluggable context store backends.

Provides a ``ContextStore`` abstract base class and three
concrete implementations:

* ``InMemoryStore`` — ephemeral, in-process storage
* ``SQLiteStore``   — file-backed persistent storage via sqlite3
* ``VectorStore``   — embedding-powered storage via ChromaDB
"""
