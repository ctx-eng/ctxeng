from __future__ import annotations

import re
from typing import Dict, List, Optional

DEFAULT_SYSTEM_TEMPLATE = """You are CtxEng.
User profile:
{profile}

Relevant memories:
{memories}

Conversation history:
{history}

Tool outputs:
{tool_outputs}

Current request:
{query}"""


class PromptTemplate:
    def __init__(self, template: str, name: str = "default") -> None:
        self.name = name
        self._template = template

    @property
    def slots(self) -> List[str]:
        return re.findall(r"\{(\w+)\}", self._template)

    def render(self, **kwargs: str) -> str:
        has_jinja = False
        try:
            from jinja2 import Template
            has_jinja = True
        except ImportError:
            pass

        if has_jinja:
            from jinja2 import Template
            return Template(self._template).render(**kwargs)

        missing = [k for k in self.slots if k not in kwargs]
        if missing:
            raise KeyError(f"Missing template slots: {missing}")
        return self._template.format(**kwargs)


TEMPLATE_REGISTRY: Dict[str, PromptTemplate] = {
    "default": PromptTemplate(DEFAULT_SYSTEM_TEMPLATE, name="default"),
}


def register_template(name: str, template: str) -> PromptTemplate:
    pt = PromptTemplate(template, name=name)
    TEMPLATE_REGISTRY[name] = pt
    return pt


def get_template(name: str = "default") -> Optional[PromptTemplate]:
    return TEMPLATE_REGISTRY.get(name)
