from enum import Enum

class SpecialTokenDefinition(Enum):
    SLIDE_PLACEHOLDER = ("slide_placeholder_token", "<slide_placeholder>")

    def __init__(self, token_name: str, token_string: str):
        # Using different internal names to avoid conflict if property names are the same
        self._defined_token_name = token_name
        self._defined_token_string = token_string

    @property
    def name(self) -> str:
        """The logical name for the token (e.g., 'slide_placeholder_token')."""
        return self._defined_token_name

    @property
    def string(self) -> str:
        """The actual string representation of the token (e.g., '<slide_placeholder>')."""
        return self._defined_token_string
    
    @staticmethod
    def to_tokenizer_dict(token_definitions: list['SpecialTokenDefinition']) -> dict[str, str]:
        """
        Converts a list of SpecialTokenDefinition enum members into the
        dictionary format required by Hugging Face tokenizers for extra_special_tokens.

        Args:
            token_definitions: A list of SpecialTokenDefinition enum members.

        Returns:
            A dictionary where keys are token names and values are token strings.
        """
        if not token_definitions:
            return {}
        return {st_def.name: st_def.string for st_def in token_definitions}
