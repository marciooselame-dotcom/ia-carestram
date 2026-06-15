import unittest
from src.core.anonymizer import Anonymizer

class TestAnonymizer(unittest.TestCase):
    """Unit test suite for validating anonymizer regular expressions and PII removal."""

    def setUp(self) -> None:
        """Initializes the Anonymizer class before each test case."""
        self.anonymizer = Anonymizer()

    def test_header_anonymization(self) -> None:
        """Tests removing whole lines containing patient metadata headers."""
        raw_text = (
            "Paciente: João da Silva Santos\n"
            "Data de Nascimento: 15/08/1974\n"
            "Prontuário: 9876543-A\n"
            "Médico Solicitante: Dr. Marcos Silveira\n"
            "Idade: 51 anos\n"
            "Sexo: Masculino\n"
            "Exame realizado de forma adequada.\n"
            "Tendão patelar íntegro."
        )
        expected = (
            "Exame realizado de forma adequada.\n"
            "Tendão patelar íntegro."
        )
        sanitized = self.anonymizer.anonymize_text(raw_text)
        self.assertEqual(sanitized, expected)

    def test_inline_anonymization(self) -> None:
        """Tests redacting inline PII such as CPFs, emails, and phone numbers within sentences."""
        raw_text = (
            "O paciente apresentou CPF 123.456.789-00 no guichê.\n"
            "Contato realizado pelo telefone (11) 99888-7766 ou email teste@hospital.com em 12/04/2026."
        )
        sanitized = self.anonymizer.anonymize_text(raw_text)
        
        self.assertIn("[CPF REDIGIDO]", sanitized)
        self.assertIn("[TELEFONE REDIGIDO]", sanitized)
        self.assertIn("[EMAIL REDIGIDO]", sanitized)
        self.assertIn("[DATA REDIGIDA]", sanitized)
        self.assertNotIn("123.456.789-00", sanitized)
        self.assertNotIn("99888-7766", sanitized)
        self.assertNotIn("teste@hospital.com", sanitized)

    def test_empty_and_invalid_inputs(self) -> None:
        """Tests the robustness of anonymizer when receiving empty strings or invalid data types."""
        self.assertEqual(self.anonymizer.anonymize_text(""), "")
        
        # Test incorrect type raises TypeError
        with self.assertRaises(TypeError):
            self.anonymizer.anonymize_text(12345)  # type: ignore

if __name__ == "__main__":
    unittest.main()
