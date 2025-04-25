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
from unittest.mock import patch, MagicMock

from dotenv import load_dotenv

from open_dubbing.speech_to_text_openai_whisper import SpeechToTextOpenAIWhisperTransformers
load_dotenv()

MODEL= "whisper-1"
class TestSpeechToTextOpenAIWhisperTransformers:

    @classmethod
    def setup_class(cls):
        cls.data_dir = os.path.dirname(os.path.realpath(__file__))
        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key is not None, "OPENAI_API_KEY not set in environment variables"
        cls.stt = SpeechToTextOpenAIWhisperTransformers(api_key=api_key, model_name=MODEL)
        cls.stt.load_model()

    @patch('openai.OpenAI')
    def test_transcribe(self, mock_openai):
        # Setup mock response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "This is a test."
        mock_response.language = "english"
        mock_client.audio.transcriptions.create.return_value = mock_response
        
        # Set the mock client
        self.stt.client = mock_client
        
        filename = os.path.join(self.data_dir, "data/this_is_a_test.mp3")
        text = self.stt._transcribe(
            vocals_filepath=filename, source_language_iso_639_1="en"
        )
        
        # Verify the API was called with correct parameters
        mock_client.audio.transcriptions.create.assert_called_once()
        call_args = mock_client.audio.transcriptions.create.call_args[1]
        assert call_args["model"] == MODEL
        assert call_args["language"] == "en"
        
        # Verify the result
        assert text.strip() == "This is a test."

    @patch('openai.OpenAI')
    def test_detect_language(self, mock_openai):
        # Setup mock response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.language = "english"
        mock_client.audio.transcriptions.create.return_value = mock_response

        # Set the mock client
        self.stt.client = mock_client

        filename = os.path.join(self.data_dir, "data/this_is_a_test.mp3")
        language = self.stt.detect_language(filename)

        # Verify the API was called
        mock_client.audio.transcriptions.create.assert_called_once()

        # Verify the result is converted to ISO 639-3
        assert language == "eng"

    def test_get_languages(self):
        languages = self.stt.get_languages()
        assert len(languages) == 100
        assert "eng" in languages