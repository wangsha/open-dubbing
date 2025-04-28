# Copyright 2025 Jordi Mas i Hernàndez <jmas@softcatala.org>
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

from typing import List

from iso639 import Lang
from openai import OpenAI

from open_dubbing import logger
from open_dubbing.text_to_speech import TextToSpeech, Voice


# Documentation: https://platform.openai.com/docs/guides/text-to-speech
class TextToSpeechOpenAI(TextToSpeech):

    def __init__(self, device="cpu", server="", api_key=""):
        super().__init__()
        self.client = OpenAI(api_key=api_key)

    def get_available_voices(self, language_code: str) -> List[Voice]:

        voices_names = {
            "alloy": self._SSML_MALE,
            "echo": self._SSML_MALE,
            "fable": self._SSML_MALE,
            "onyx": self._SSML_MALE,
            "nova": self._SSML_FEMALE,
            "shimmer": self._SSML_FEMALE,
        }

        voices = []
        for voice_name in voices_names.keys():
            gender = voices_names[voice_name]

            voice = Voice(
                name=voice_name,
                gender=gender,
                region="",
            )
            voices.append(voice)

        logger().debug(
            f"text_to_speech_openai.get_available_voices: {voices} for language {language_code}"
        )

        return voices

    def _does_voice_supports_speeds(self):
        return False

    def _convert_text_to_speech(
        self,
        *,
        assigned_voice: str,
        target_language: str,
        output_filename: str,
        text: str,
        speed: float,
    ) -> str:

        logger().debug(
            f"text_to_speech_openai._convert_text_to_speech: assigned_voice: {assigned_voice}, output_filename: '{output_filename}'"
        )
        with self.client.with_streaming_response.audio.speech.create(
            model="tts-1",
            voice=assigned_voice,
            input=text,
        ) as response:
            response.stream_to_file(output_filename)

        return output_filename

    def _get_iso_639_3(self, iso_639_1: str):
        if iso_639_1 == "jw":
            iso_639_1 = "jv"

        o = Lang(iso_639_1)
        iso_639_3 = o.pt3
        return iso_639_3

    def get_languages(self):
        from transformers.models.whisper.tokenization_whisper import LANGUAGES

        languages = []
        for language in LANGUAGES:
            pt3 = self._get_iso_639_3(language)
            languages.append(pt3)
        logger().debug(f"text_to_speech_openai.get_languages: {languages}")
        return sorted(languages)
