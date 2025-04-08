import unittest
# Assuming OpenAlexService is importable from app
# Adjust the import path if OpenAlexService is in a different module within backend
from app import OpenAlexService

class TestOpenAlexService(unittest.TestCase):
    def setUp(self):
        # Provide a dummy email for testing purposes
        self.service = OpenAlexService(email="test@example.com")

    def test_reconstruct_abstract_from_inverted_index(self):
        # Example from OpenAlex documentation or known structure
        inverted_index = {"Hello": [0, 2], "World": [1]}
        expected = "Hello World Hello"
        # Accessing the method, assuming it's intended for internal use but testable
        result = self.service._reconstruct_abstract_from_inverted_index(inverted_index)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main() 