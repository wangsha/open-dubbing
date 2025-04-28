# Copyright 2024 Jordi Mas i Hernàndez <jmas@softcatala.org>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from unittest.mock import MagicMock, patch

from dotenv import load_dotenv

from open_dubbing.translation_openai import TranslationOpenAI

load_dotenv()

MODEL = "gpt-4o"


class TestTranslationOpenAI:

    @classmethod
    def setup_class(cls):
        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key is not None, "OPENAI_API_KEY not set in environment variables"
        cls.translator = TranslationOpenAI(api_key=api_key, model_name=MODEL)
        cls.translator.load_model()

    @patch("openai.OpenAI")
    def test_translate_text(self, mock_openai):
        # Setup mock response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Hola, mundo"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_client.chat.completions.create.return_value = mock_response

        # Set the mock client
        self.translator.client = mock_client

        # Test translation
        translated_text = self.translator._translate_text(
            source_language="eng", target_language="spa", text="Hello, world"
        )

        # Verify the API was called with correct parameters
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        assert call_args["model"] == MODEL

        # Check that the messages contain the correct prompt
        messages = call_args["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "English" in messages[1]["content"]
        assert "Spanish" in messages[1]["content"]
        assert "Hello, world" in messages[1]["content"]

        # Verify the result
        assert translated_text == "Hola, mundo"

    @patch("openai.OpenAI")
    def test_custom_prompt_template(self, mock_openai):
        # Setup mock response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Bonjour le monde"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_client.chat.completions.create.return_value = mock_response

        # Create translator with custom prompt template
        custom_prompt = "Custom prompt: translate '{text}' from {source_language} to {target_language}"
        translator = TranslationOpenAI(
            api_key="test_key", model_name=MODEL, prompt_template=custom_prompt
        )
        translator.client = mock_client

        # Test translation
        translated_text = translator._translate_text(
            source_language="eng", target_language="fra", text="Hello world"
        )

        # Verify the API was called with the custom prompt
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        messages = call_args["messages"]
        assert "Custom prompt" in messages[1]["content"]
        assert "Hello world" in messages[1]["content"]

        # Verify the result
        assert translated_text == "Bonjour le monde"

    def test_empty_text_translation(self):
        # Test that empty text returns empty string without making API call
        result = self.translator._translate_text(
            source_language="eng", target_language="spa", text="   "  # Just whitespace
        )
        assert result == ""

    def test_get_language_pairs(self):
        # Test that get_language_pairs returns a non-empty set of language pairs
        pairs = self.translator.get_language_pairs()
        assert isinstance(pairs, set)
        assert len(pairs) > 0

        # Check that each pair is a tuple of two strings
        for pair in pairs:
            assert isinstance(pair, tuple)
            assert len(pair) == 2
            assert isinstance(pair[0], str)
            assert isinstance(pair[1], str)

        # Check that some common language pairs are included
        assert ("eng", "spa") in pairs  # English to Spanish
        assert ("fra", "eng") in pairs  # French to English
        assert ("deu", "ita") in pairs  # German to Italian

    def test_get_language_name(self):
        # Test conversion of ISO 639-3 codes to language names
        assert self.translator._get_language_name("eng") == "English"
        assert self.translator._get_language_name("spa") == "Spanish"
        assert self.translator._get_language_name("fra") == "French"
        assert self.translator._get_language_name("deu") == "German"

        # Test fallback for unknown language code
        unknown_code = "xyz"
        assert self.translator._get_language_name(unknown_code) == unknown_code
