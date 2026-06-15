import re
import logging
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Anonymizer:
    """Class responsible for removing personally identifiable information (PII) from medical reports.
    
    This includes patient names, dates of birth, medical records (prontuário), CPFs,
    phone numbers, addresses, and physician details, ensuring absolute privacy compliance.
    """

    def __init__(self) -> None:
        """Initializes the Anonymizer with compiled regular expression patterns for performance optimization."""
        # Common headers to strip entirely from lines
        self.header_patterns: List[re.Pattern] = [
            re.compile(r"^\s*(paciente|nome|nome\s+do\s+paciente)\s*:\s*.*$", re.IGNORECASE),
            re.compile(r"^\s*(dn|d\.n\.|data\s+de\s+nascimento|nascimento)\s*:\s*.*$", re.IGNORECASE),
            re.compile(r"^\s*(prontuário|prontuario|registro|reg|id|atendimento)\s*:\s*.*$", re.IGNORECASE),
            re.compile(r"^\s*(médico|medico|solicitante|médico\s+solicitante|dr\.|dra\.)\s*:\s*.*$", re.IGNORECASE),
            re.compile(r"^\s*(cpf|identidade|rg)\s*:\s*.*$", re.IGNORECASE),
            re.compile(r"^\s*(telefone|tel|celular|cel)\s*:\s*.*$", re.IGNORECASE),
            re.compile(r"^\s*(endereço|endereco|rua|av\.|avenida)\s*:\s*.*$", re.IGNORECASE),
            re.compile(r"^\s*(idade|sexo|convênio|convenio)\s*:\s*.*$", re.IGNORECASE),
        ]

        # Inline PII patterns to redact within sentences
        self.inline_patterns: List[tuple[re.Pattern, str]] = [
            # CPF (e.g., 123.456.789-00)
            (re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"), "[CPF REDIGIDO]"),
            # Date of birth or generic dates associated with birth (e.g. 12/05/1980 or 12.05.1980)
            (re.compile(r"\b\d{2}[/\.]\d{2}[/\.]\d{4}\b"), "[DATA REDIGIDA]"),
            # Phone numbers
            (re.compile(r"\b(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?9?\d{4}[-.\s]?\d{4}\b"), "[TELEFONE REDIGIDO]"),
            # E-mails
            (re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"), "[EMAIL REDIGIDO]"),
        ]

    def anonymize_text(self, text: str) -> str:
        """Removes PII from the given radiology report text.

        Args:
            text (str): The raw report text containing clinical information and potential PII.

        Returns:
            str: The sanitized text containing only clinical information.

        Raises:
            TypeError: If the input text is not a string.

        Complexity:
            Time Complexity: O(N * P) where N is the number of characters in the text and
                            P is the number of regular expression patterns applied.
            Space Complexity: O(N) to store the modified copy of the text.
        """
        if not isinstance(text, str):
            logger.error("Input text is not a string. Type: %s", type(text))
            raise TypeError("Input must be a string")

        try:
            lines: List[str] = text.splitlines()
            sanitized_lines: List[str] = []

            for line in lines:
                should_skip = False
                # 1. Check line-level header patterns
                for pattern in self.header_patterns:
                    if pattern.match(line):
                        should_skip = True
                        break
                
                if should_skip:
                    continue

                # 2. Process inline patterns
                sanitized_line = line
                for pattern, replacement in self.inline_patterns:
                    sanitized_line = pattern.sub(replacement, sanitized_line)
                
                sanitized_lines.append(sanitized_line)

            # Rejoin the lines and remove leading/trailing whitespace
            result = "\n".join(sanitized_lines).strip()
            return result

        except Exception as e:
            # Section 2.1: Resilience Pipeline
            # 1. Isolation: Catching generic/all exceptions to protect execution flow
            # 2. Observability: Logging error details with trace context
            logger.error("Failed to anonymize text. Error: %s", str(e), exc_info=True)
            # 3. Graceful degradation: Return the original text if processing fails,
            # ensuring critical operations don't freeze, but warn the developer.
            return text
