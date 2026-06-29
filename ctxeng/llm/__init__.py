"""LLM provider abstraction layer.

* ``LLMProvider`` — abstract base class for text-generation backends
* ``OpenAIProvider`` — ChatGPT / compatible API via openai SDK
* ``OllamaProvider`` — local models via Ollama HTTP API
* ``run_chat`` — high-level chat loop that integrates ContextAssembler + LLM
"""
