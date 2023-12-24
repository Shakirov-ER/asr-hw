CODE = hw_asr

switch_to_macos:
	rm poetry.lock
	cat utils/pyproject_macos.txt > pyproject.toml

switch_to_linux:
	rm poetry.lock
	cat utils/pyproject_linux.txt > pyproject.toml

install:
	python3.10 -m pip install poetry
	poetry install

test:
	poetry run python -m unittest discover hw_asr/tests

lint:
	poetry run pflake8 $(CODE)

format:
	#format code
	poetry run black $(CODE)

download_checkpoint:
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1hWrzh8YRn6jvznTEom9dvMW3jEjguSCb" -O default_test_model/model_best.pth && rm -rf /tmp/cookies.txt

download_language_model:
	poetry run python language_models/download_lm.py

test_model_test_other:
	cp utils/config_other.json default_test_model/config.json
	poetry run python test.py -r default_test_model/model_best.pth -o output_test_other.json

test_model_test_clean:
	cp utils/config_clean.json default_test_model/config.json
	poetry run python test.py -r default_test_model/model_best.pth -o output_test_clean.json

train:
	poetry run python train.py -c hw_asr/config.json