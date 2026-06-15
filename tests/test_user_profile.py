import unittest
from src.core.user_profile import UserProfile

class TestUserProfile(unittest.TestCase):
    """Unit test suite for validating user profile operations and style learning features."""

    def setUp(self) -> None:
        """Initializes the UserProfile with an in-memory database to isolate test side-effects."""
        self.profile = UserProfile(db_path=":memory:")

    def tearDown(self) -> None:
        """Closes the test database connection."""
        self.profile.close()

    def test_default_templates(self) -> None:
        """Tests that default templates are loaded correctly for all supported joints."""
        joints = ["joelho", "ombro", "quadril", "tornozelo", "punho", "cotovelo"]
        for joint in joints:
            template = self.profile.get_template(joint)
            self.assertTrue(len(template) > 0)
            self.assertIn(joint.upper(), template.upper())

    def test_update_template(self) -> None:
        """Tests updating joint templates with custom content."""
        new_template = "TEMPLATE PERSONALIZADO DE TESTE DE JOELHO"
        success = self.profile.update_template("joelho", new_template)
        self.assertTrue(success)
        self.assertEqual(self.profile.get_template("joelho"), new_template)

    def test_vocabulary_learning(self) -> None:
        """Tests comparing AI output against final physician edits to extract vocabulary choices."""
        ai_text = "O laudo indica uma ruptura do ligamento cruzado anterior."
        final_text = "O laudo indica uma rotura do ligamento cruzado anterior."
        
        # Initial learn execution
        learned = self.profile.learn_from_edit(ai_text, final_text)
        self.assertEqual(len(learned), 1)
        self.assertEqual(learned[0], ("ruptura", "rotura"))

        # Verify learning was persisted in DB
        cursor = self.profile.conn.cursor()
        cursor.execute("SELECT preferred_word, frequency FROM learned_vocabulary WHERE original_word = 'ruptura'")
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "rotura")
        self.assertEqual(row[1], 1)

        # Trigger second learn of same preference to verify frequency increments
        self.profile.learn_from_edit(ai_text, final_text)
        cursor.execute("SELECT frequency FROM learned_vocabulary WHERE original_word = 'ruptura'")
        row = cursor.fetchone()
        self.assertEqual(row[0], 2)

    def test_style_prompt_generation(self) -> None:
        """Tests that learned preferences are properly compiled into the prompt guidelines."""
        # Clean state has no preferences
        empty_prompt = self.profile.get_style_preferences_prompt()
        self.assertEqual(empty_prompt, "Nenhuma preferência personalizada registrada ainda.")

        # Simulate learned words with frequency >= 2
        ai_text = "Detectamos uma ruptura e um derrame."
        final_text = "Detectamos uma rotura e um derrame."
        self.profile.learn_from_edit(ai_text, final_text)
        self.profile.learn_from_edit(ai_text, final_text) # freq = 2

        # Verify prompt details
        prompt = self.profile.get_style_preferences_prompt()
        self.assertIn("ruptura", prompt)
        self.assertIn("rotura", prompt)

if __name__ == "__main__":
    unittest.main()
