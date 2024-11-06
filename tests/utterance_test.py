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

import json
import os
import tempfile

from open_dubbing.utterance import Utterance


class TestUterrance:

    def testrun_save_utterance(self):

        with tempfile.TemporaryDirectory() as temp_dir:
            directory = temp_dir

            utterance = Utterance(
                target_language="cat",
                output_directory=directory,
            )
            utterance_metadata = [
                {"start": 1.26, "end": 3.94},
                {"start": 5.24, "end": 6.629},
            ]

            utterance.save_utterances(
                source_language="spa",
                utterance_metadata=utterance_metadata,
                preprocesing_output=None,
            )
            metadata_file = os.path.join(directory, "utterance_metadata_cat.json")
            with open(metadata_file, encoding="utf-8") as json_data:
                data = json.load(json_data)
                assert data == {
                    "utterances": [
                        {
                            "id": 1,
                            "start": 1.26,
                            "end": 3.94,
                            "hash": "26d514d9ce21021f51bd010d9946db0f31555ef7145067d4fe5a3b1bdcd84ce7",
                        },
                        {
                            "id": 2,
                            "start": 5.24,
                            "end": 6.629,
                            "hash": "157dc7fb355c7dc13a0ea687e9fd4a6f6c5c03526a959a64dfe1fa7562fedff4",
                        },
                    ],
                    "source_language": "spa",
                }

    def test_hash_utterances(self):

        utterances = [
            {
                "start": 1.26,
                "end": 3.94,
            },
            {
                "start": 5.24,
                "end": 6.629,
            },
        ]
        utterance = Utterance(
            target_language="cat",
            output_directory=None,
        )

        hashed = utterance._hash_utterances(utterances)
        assert hashed == [
            {
                "start": 1.26,
                "end": 3.94,
                "hash": "2fa6f80e0c81fb8e142f2dbbad0bceff7c21a031833b5752bc1cfd799f6b3bc6",
            },
            {
                "start": 5.24,
                "end": 6.629,
                "hash": "34cd5da78cb163ad18996aefffcfeae864727257defc7ae68818a245ca269951",
            },
        ]

    def test_get_modified_utterances(self):
        utterances = [
            {
                "id": 1,
                "start": 1.26,
                "end": 3.94,
                "hash": "26d514d9ce21021f51bd010d9946db0f31555ef7145067d4fe5a3b1bdcd84ce7",
            },
            {
                "id": 2,
                "start": 5.25,
                "end": 6.629,
                "hash": "157dc7fb355c7dc13a0ea687e9fd4a6f6c5c03526a959a64dfe1fa7562fedff4",
            },
        ]
        dubbing = Utterance(
            target_language="cat",
            output_directory=None,
        )

        modified = dubbing.get_modified_utterances(utterances)
        assert 1 == len(modified)

    def test_get_without_empty_blocks(self):
        utterances = [
            {
                "start": 1.26,
                "end": 3.94,
                "text": "Hola",
            },
            {
                "start": 5.24,
                "end": 6.600,
                "text": "",
            },
        ]

        dubbing = Utterance(
            target_language="cat",
            output_directory=None,
        )

        modified = dubbing.get_without_empty_blocks(utterances)
        assert 1 == len(modified)

    def test_add_unique_ids(self):

        utterances = [
            {
                "start": 1.26,
                "end": 3.94,
            },
            {
                "start": 5.24,
                "end": 6.629,
            },
        ]
        utterance = Utterance(
            target_language="cat",
            output_directory=None,
        )

        unique_ids = utterance._add_unique_ids(utterances)
        assert unique_ids == [
            {"id": 1, "start": 1.26, "end": 3.94},
            {"id": 2, "start": 5.24, "end": 6.629},
        ]
