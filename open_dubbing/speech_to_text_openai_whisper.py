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

import array
import os

from openai import OpenAI
from transformers.models.whisper.tokenization_whisper import LANGUAGES, TO_LANGUAGE_CODE

from open_dubbing import logger
from open_dubbing.pydub_audio_segment import AudioSegment
from open_dubbing.speech_to_text import SpeechToText


class SpeechToTextOpenAIWhisperTransformers(SpeechToText):

    def __init__(
        self,
        device="cpu",
        cpu_threads=0,
        model_name="gpt-4o-mini-transcribe",
        api_key="",
    ):
        assert model_name in [
            "gpt-4o-mini-transcribe",
            "gpt-4o-transcribe",
            "whisper-1",
        ], f"Invalid model name, supported models are: gpt-4o-mini-transcribe, gpt-4o-transcribe, whisper-1, got {model_name}"
        super().__init__(device=device, model_name=model_name, cpu_threads=cpu_threads)
        self.client = OpenAI(api_key=api_key)

    def load_model(self):
        # No model to load as we're using the OpenAI API
        pass

    def _transcribe(
        self,
        *,
        vocals_filepath: str,
        source_language_iso_639_1: str,
    ) -> str:
        logger().debug(
            f"speech_to_text_openai_whisper._transcribe. file: {vocals_filepath}, language: {source_language_iso_639_1}"
        )

        # Open the audio file
        with open(vocals_filepath, "rb") as audio_file:
            # Call the OpenAI API to transcribe the audio
            response = self.client.audio.transcriptions.create(
                model=self.model_name,
                file=audio_file,
                language=source_language_iso_639_1,
                response_format="verbose_json",
                timestamp_granularities="segment",
            )
        data = response.model_dump()
        transcription = response.text

        logger().debug(
            f"speech_to_text_openai_whisper._transcribe. transcription: {transcription}, file {vocals_filepath}"
        )
        return transcription

    def _get_audio_language(self, audio: array.array) -> str:
        # Save the audio array to a temporary file
        temp_file = "temp_audio_for_language_detection.wav"
        audio_segment = AudioSegment(
            data=audio,
            sample_width=2,  # 16-bit audio
            frame_rate=16000,
            channels=1,
        )
        audio_segment.export(temp_file, format="wav")

        try:
            # Open the audio file
            with open(temp_file, "rb") as audio_file:
                # Call the OpenAI API to detect the language
                response = self.client.audio.transcriptions.create(
                    model=self.model_name,
                    file=audio_file,
                    response_format="verbose_json",
                )

            # Extract language from response
            detected_language = response.language

            # Convert to ISO 639-3 if needed
            if detected_language:
                detected_language = TO_LANGUAGE_CODE[detected_language]
                detected_language = self._get_iso_639_3(detected_language)

            logger().debug(
                f"speech_to_text_openai_whisper._get_audio_language. Detected language: {detected_language}"
            )

            return detected_language
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def get_languages(self):
        iso_639_3 = []
        for language in LANGUAGES:
            pt3 = self._get_iso_639_3(language)
            iso_639_3.append(pt3)
        return iso_639_3
