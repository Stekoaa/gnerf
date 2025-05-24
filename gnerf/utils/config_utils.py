from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple, Type
from rich.console import Console

CONSOLE = Console()

# Pretty printing class
class PrintableConfig:
    """Printable Config defining str function"""

    def __str__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if isinstance(val, Tuple):
                flattened_val = "["
                for item in val:
                    flattened_val += str(item) + "\n"
                flattened_val = flattened_val.rstrip("\n")
                val = flattened_val + "]"
            lines += f"{key}: {str(val)}".split("\n")
        return "\n    ".join(lines)


# Base instantiate configs
@dataclass
class InstantiateConfig(PrintableConfig):
    """Config class for instantiating an the class specified in the _target attribute."""

    _target: Type

    def setup(self, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self._target(self, **kwargs)
    

def convert_markup_to_ansi(markup_string: str) -> str:
    """Convert rich-style markup to ANSI sequences for command-line formatting.

    Args:
        markup_string: Text with rich-style markup.

    Returns:
        Text formatted via ANSI sequences.
    """
    with CONSOLE.capture() as out:
        CONSOLE.print(markup_string, soft_wrap=True)
    return out.get()