# Transcribe and Diarize

A script for trancribing and diarizing wav formatted audio files.

## Installation

``` bash
pip install git+https://github.com/AaltoRSE/Diarize.git
```

## Usage

List all parameters using 

``` bash
transcribe_and_diarize --help
```

For example, to run for all files in a folder:

``` bash
transcribe_and_diarize --input_folder=INPUT_FOLDER_NAME --output_folder=OUTPUT_FOLDER_NAME --hugging_face_token YOUR_TOKEN
```

