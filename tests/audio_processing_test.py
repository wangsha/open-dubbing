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

"""Tests for utility functions in audio_processing.py."""

import os
import subprocess
import tempfile

from unittest.mock import MagicMock

import pytest

from pyannote.audio import Pipeline
from pydub import AudioSegment

from open_dubbing import audio_processing


class TestAudioProcessing:

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

    def test_create_timestamps_with_silence(self):
        SILENCE_DURATION = 10
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary_file:
            self._generate_silence(
                output_file=temporary_file.name, silence_duration=SILENCE_DURATION
            )
            mock_pipeline = MagicMock(spec=Pipeline)
            mock_pipeline.return_value.itertracks.return_value = [
                (MagicMock(start=0.0, end=SILENCE_DURATION), None, "SPEAKER_00")
            ]
            timestamps = audio_processing.create_pyannote_timestamps(
                audio_file=temporary_file.name,
                pipeline=mock_pipeline,
            )
            assert timestamps == [{"start": 0.0, "end": 10, "speaker_id": "SPEAKER_00"}]

    def test_cut_and_save_audio_no_clone(self):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temporary_file:
            self._generate_silence(output_file=temporary_file.name, silence_duration=10)
            with tempfile.TemporaryDirectory() as output_directory:
                audio = AudioSegment.from_file(temporary_file.name)
                audio_processing._cut_and_save_audio(
                    audio=audio,
                    utterance=dict(start=0.1, end=0.2),
                    prefix="chunk",
                    output_directory=output_directory,
                )
                expected_file = os.path.join(output_directory, "chunk_0.1_0.2.mp3")
                assert os.path.exists(expected_file)

    def test_run_cut_and_save_audio(self):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temporary_file:
            self._generate_silence(output_file=temporary_file.name, silence_duration=10)
            utterance_metadata = [{"start": 0.0, "end": 5.0}]
            with tempfile.TemporaryDirectory() as output_directory:
                # TODO: Add asert on results on this method
                _ = audio_processing.run_cut_and_save_audio(
                    utterance_metadata=utterance_metadata,
                    audio_file=temporary_file.name,
                    output_directory=output_directory,
                )
                expected_file = os.path.join(output_directory, "chunk_0.0_5.0.mp3")
                _ = {
                    "path": os.path.join(output_directory, "chunk_0.0_5.0.mp3"),
                    "start": 0.0,
                    "end": 5.0,
                }
                assert os.path.exists(expected_file)

    @pytest.mark.parametrize(
        "for_dubbing, expected_file_size",
        [
            (False, 20524),
            (True, 160749),
        ],
    )
    def test_insert_audio_at_timestamps(self, for_dubbing, expected_file_size):

        with tempfile.TemporaryDirectory() as temporary_directory:
            background_audio_file = f"{temporary_directory}/test_background.mp3"
            SILENCE_DURATION = 10

            self._generate_silence(
                output_file=background_audio_file, silence_duration=SILENCE_DURATION
            )

            data_dir = os.path.dirname(os.path.realpath(__file__))
            audio_chunk_path = os.path.join(data_dir, "data/this_is_a_test.mp3")

            utterance_metadata = [
                {
                    "start": 3.0,
                    "end": 5.0,
                    "for_dubbing": for_dubbing,
                    "dubbed_path": audio_chunk_path,
                }
            ]
            output_path = audio_processing.insert_audio_at_timestamps(
                utterance_metadata=utterance_metadata,
                background_audio_file=background_audio_file,
                output_directory=temporary_directory,
            )
            file_size = os.path.getsize(output_path)
            assert os.path.exists(output_path)

            tolerance = 1  # Allow for a 1-byte diffence across platforms
            assert abs(expected_file_size - file_size) <= tolerance

    def test_mix_music_and_vocals(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            background_audio_path = f"{temporary_directory}/test_background.mp3"
            vocals_audio_path = f"{temporary_directory}/test_vocals.mp3"

            self._generate_silence(
                output_file=background_audio_path, silence_duration=10
            )
            self._generate_silence(output_file=vocals_audio_path, silence_duration=5)

            output_audio_path = audio_processing.merge_background_and_vocals(
                background_audio_file=background_audio_path,
                dubbed_vocals_audio_file=vocals_audio_path,
                output_directory=temporary_directory,
                target_language="en-US",
            )
            assert os.path.exists(output_audio_path)
