"""Context assembly engine.

* ``ContextAssembler`` — retrieve, deduplicate, prioritize, render, trim to token budget
* ``Prioritizer`` — trigram dedup + MMR diversity ranking
* ``PromptTemplate`` — slot-based template with Jinja2 / str-format fallback
"""
