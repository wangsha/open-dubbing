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

import time

from openai import OpenAI

from open_dubbing import logger
from open_dubbing.translation import Translation


class TranslationOpenAI(Translation):
    """
    Translation implementation using OpenAI's API with a prompt-based approach.
    """

    def __init__(
        self, device="cpu", model_name="gpt-4o-mini", api_key="", prompt_template=None
    ):
        """
        Initialize the TranslationOpenAI class.

        Args:
            device: The device to use (not used for API-based translation)
            model_name: The OpenAI model to use for translation
            api_key: The OpenAI API key
            prompt_template: Custom prompt template for translation. If None, a default template is used.
        """
        super().__init__(device)
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

        # Default prompt template if none is provided
        self.prompt_template = prompt_template or (
            "Translate the following text from {source_language} to {target_language}. "
            "Maintain the original meaning, tone, and style. "
            "Only return the translated text without explanations or notes.\n\n"
            "Text to translate: {text}"
        )

    def load_model(self):
        """
        No model to load as we're using the OpenAI API.
        """
        pass

    def _translate_text(
        self, source_language: str, target_language: str, text: str
    ) -> str:
        """
        Translate text using OpenAI's API with a prompt-based approach.

        Args:
            source_language: The ISO 639-3 code of the source language
            target_language: The ISO 639-3 code of the target language
            text: The text to translate

        Returns:
            The translated text
        """
        if not text.strip():
            return ""

        start_time = time.time()

        # Format the prompt with the source and target languages and the text to translate
        prompt = self.prompt_template.format(
            source_language=self._get_language_name(source_language),
            target_language=self._get_language_name(target_language),
            text=text,
        )

        logger().debug(f"translation_openai._translate_text. Prompt: {prompt}")

        # Call the OpenAI API to translate the text
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional translator.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,  # Lower temperature for more consistent translations
                )

                translated_text = response.choices[0].message.content.strip()

                execution_time = time.time() - start_time
                logger().debug(
                    f"translation_openai._translate_text. Translation completed in {execution_time:.2f} seconds."
                )
                logger().debug(
                    f"translation_openai._translate_text. Result: {translated_text}"
                )

                return translated_text

            except Exception as e:
                if attempt == max_retries:
                    logger().error(
                        f"translation_openai._translate_text. Max retries reached. Could not complete translation API call: {str(e)}"
                    )
                    raise
                else:
                    logger().warning(
                        f"translation_openai._translate_text. Could not complete translation API call, retrying attempt {attempt}: {str(e)}"
                    )
                    time.sleep(5)  # Wait before retrying

    def get_language_pairs(self):
        """
        Return all possible language pairs supported by OpenAI.
        OpenAI supports most languages, so we return a comprehensive list of common language pairs.

        Returns:
            A set of tuples containing source and target language ISO 639-3 codes
        """
        # Common languages supported by OpenAI
        languages = [
            "eng",  # English
            "spa",  # Spanish
            "fra",  # French
            "deu",  # German
            "ita",  # Italian
            "por",  # Portuguese
            "rus",  # Russian
            "jpn",  # Japanese
            "zho",  # Chinese
            "ara",  # Arabic
            "hin",  # Hindi
            "ben",  # Bengali
            "cat",  # Catalan
            "nld",  # Dutch
            "kor",  # Korean
            "tur",  # Turkish
            "vie",  # Vietnamese
            "tha",  # Thai
            "ind",  # Indonesian
            "swe",  # Swedish
            "nor",  # Norwegian
            "fin",  # Finnish
            "dan",  # Danish
            "pol",  # Polish
            "ukr",  # Ukrainian
            "heb",  # Hebrew
            "ell",  # Greek
            "hun",  # Hungarian
            "ces",  # Czech
            "ron",  # Romanian
        ]

        # Generate all possible language pairs
        pairs = set()
        for source in languages:
            for target in languages:
                if source != target:
                    pairs.add((source, target))

        return pairs

    def _get_language_name(self, iso_639_3_code: str) -> str:
        """
        Convert ISO 639-3 language code to full language name for better prompting.

        Args:
            iso_639_3_code: The ISO 639-3 language code

        Returns:
            The full language name
        """
        language_map = {
            "eng": "English",
            "spa": "Spanish",
            "fra": "French",
            "deu": "German",
            "ita": "Italian",
            "por": "Portuguese",
            "rus": "Russian",
            "jpn": "Japanese",
            "zho": "Chinese",
            "ara": "Arabic",
            "hin": "Hindi",
            "ben": "Bengali",
            "cat": "Catalan",
            "nld": "Dutch",
            "kor": "Korean",
            "tur": "Turkish",
            "vie": "Vietnamese",
            "tha": "Thai",
            "ind": "Indonesian",
            "swe": "Swedish",
            "nor": "Norwegian",
            "fin": "Finnish",
            "dan": "Danish",
            "pol": "Polish",
            "ukr": "Ukrainian",
            "heb": "Hebrew",
            "ell": "Greek",
            "hun": "Hungarian",
            "ces": "Czech",
            "ron": "Romanian",
        }

        return language_map.get(iso_639_3_code, iso_639_3_code)
