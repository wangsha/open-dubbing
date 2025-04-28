.PHONY: dev run-tests run-e2e-tests publish-release

PATHS = open_dubbing/ tests/ e2e-tests/

dev:
	python -m black $(PATHS)
	python -m flake8 $(PATHS)
	python -m isort $(PATHS)

run-tests:
	python -m pytest tests/

run-e2e-tests:
	CT2_USE_MKL="False" CT2_FORCE_CPU_ISA='GENERIC' KMP_DUPLICATE_LIB_OK="TRUE" python -m pytest e2e-tests/

publish-release:
	rm dist/ -r -f
	python setup.py sdist bdist_wheel
	python -m  twine upload -u "__token__" -p "${PYPI_API_TOKEN}" --repository-url https://upload.pypi.org/legacy/ dist/*

run:
	export PYTHONPATH="${PYTHONPATH}:."; pipenv run python open_dubbing/main.py --tts openai --stt openai-whisper --translator openai --dubbed_subtitles --original_subtitles   --input_file e2e-tests/french_audio/frenchaudio.mp3 --target_language=eng --output_directory e2e-tests/french_audio
