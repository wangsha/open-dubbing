# Copyright 2024 Google LLC
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
import subprocess
import tempfile

from collections import namedtuple
from unittest.mock import MagicMock

import pytest

from faster_whisper import WhisperModel

from open_dubbing.speech_to_text_faster_whisper import SpeechToTextFasterWhisper


class TestTranscribe:

    def _generate_silence(self, *, output_file, silence_duration):
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-t",
            str(silence_duration),
            "-q:a",
            "9",
            output_file,
        ]
        subprocess.run(command, check=True)

    @pytest.fixture(autouse=True)
    def setup(self):
        """Fixture to create and clean up the temporary audio file."""
        self.silence_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        SILENCE_DURATION = 5

        self._generate_silence(
            output_file=self.silence_audio.name, silence_duration=SILENCE_DURATION
        )

        yield self.silence_audio.name
        os.remove(self.silence_audio.name)

    def test_transcribe(self):
        mock_model = MagicMock(spec=WhisperModel)
        Segment = namedtuple("Segment", ["text"])
        mock_model.transcribe.return_value = [Segment(text="Test.")], None
        spt = SpeechToTextFasterWhisper()
        spt.model = mock_model
        transcribed_text = spt._transcribe(
            vocals_filepath=self.silence_audio,
            source_language_iso_639_1="eng",
        )
        assert transcribed_text == "Test."

    @pytest.mark.parametrize(
        "no_dubbing_phrases, expected_for_dubbing",
        [
            (["goodbye"], True),
            ([], True),
        ],
    )
    def test_transcribe_chunks(self, no_dubbing_phrases, expected_for_dubbing):
        mock_model = MagicMock(spec=WhisperModel)
        Segment = namedtuple("Segment", ["text"])
        mock_model.transcribe.return_value = [
            Segment(text="hello world this is a test")
        ], None
        utterance_metadata = [dict(path=self.silence_audio, start=0.0, end=5.0)]
        source_language = "en"
        spt = SpeechToTextFasterWhisper()
        spt.model = mock_model
        transcribed_audio_chunks = spt.transcribe_audio_chunks(
            utterance_metadata=utterance_metadata,
            source_language=source_language,
            no_dubbing_phrases=no_dubbing_phrases,
        )
        expected_result = [
            dict(
                path=self.silence_audio,
                start=0.0,
                end=5.0,
                text="hello world this is a test",
                for_dubbing=expected_for_dubbing,
            )
        ]
        assert transcribed_audio_chunks == expected_result

    def test_transcribe_chunks_exception(self):
        no_dubbing_phrases = ["goodbye"]
        mock_model = MagicMock(spec=WhisperModel)
        mock_model.side_effect = Exception("This is an error")
        utterance_metadata = [dict(path=self.silence_audio, start=0.0, end=5.0)]
        spt = SpeechToTextFasterWhisper()
        spt.model = mock_model

        transcribed_audio_chunks = spt.transcribe_audio_chunks(
            utterance_metadata=utterance_metadata,
            source_language="en",
            no_dubbing_phrases=no_dubbing_phrases,
        )
        assert transcribed_audio_chunks[0]["text"] == ""
        assert not transcribed_audio_chunks[0]["for_dubbing"]


class TestAddSpeakerInfo:

    def test_add_speaker_info(self):
        utterance_metadata = [
            {
                "text": "Hello",
                "start": 0.0,
                "end": 1.0,
                "path": "path/to/file.mp3",
            },
            {
                "text": "world",
                "start": 1.0,
                "end": 2.0,
                "path": "path/to/file.mp3",
            },
        ]
        speaker_info = [("speaker1", "male"), ("speaker2", "female")]
        expected_result = [
            {
                "text": "Hello",
                "start": 0.0,
                "end": 1.0,
                "speaker_id": "speaker1",
                "gender": "male",
                "path": "path/to/file.mp3",
            },
            {
                "text": "world",
                "start": 1.0,
                "end": 2.0,
                "speaker_id": "speaker2",
                "gender": "female",
                "path": "path/to/file.mp3",
            },
        ]
        result = SpeechToTextFasterWhisper().add_speaker_info(
            utterance_metadata, speaker_info
        )
        assert result == expected_result

    def test_add_speaker_info_unequal_lengths(self):
        utterance_metadata = [
            {"text": "Hello", "start": 0.0, "stop": 1.0},
            {"text": "world", "start": 1.0, "stop": 2.0},
        ]
        speaker_info = [("speaker1", "male")]
        with pytest.raises(
            Exception,
            match="The length of 'utterance_metadata' and 'speaker_info' must be the same.",
        ):
            SpeechToTextFasterWhisper().add_speaker_info(
                utterance_metadata, speaker_info
            )

    def test_get_unique_speakers_largest_audio(self):
        test_data = [
            {
                "start": 110.73471875000001,
                "end": 111.74721875,
                "speaker_id": "SPEAKER_01",
                "path": "chunk_111.mp3",
            },
            {
                "start": 113.73471875000001,
                "end": 114.74721875,
                "speaker_id": "SPEAKER_01",
                "path": "chunk_120.mp3",
            },
            {
                "start": 113.73471875000001,
                "end": 120,
                "speaker_id": "SPEAKER_01",
                "path": "chunk_114.mp3",
            },
        ]
        result = SpeechToTextFasterWhisper()._get_unique_speakers_largest_audio(
            test_data
        )

        assert [("SPEAKER_01", "chunk_114.mp3")] == result
