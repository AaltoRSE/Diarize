# Transcribe and Diarize

A script for trancribing and diarizing wav formatted audio files.

## Installation

``` bash
pip install git+https://github.com/AaltoRSE/Diarize.git
```

## Usage

### Transcription

List all parameters using 

``` bash
transcribe_and_diarize --help
```

For example, to run for all files in a folder:

``` bash
transcribe_and_diarize --input_folder=INPUT_FOLDER_NAME --output_folder=OUTPUT_FOLDER_NAME --hugging_face_token YOUR_TOKEN
```

### Summarizing transcripts

First install the GPT4All client and use it to download a model. Make a note of the folder where the model files are stored. You will need a path to the model file to run summarization.

To summarize all diarized transcripts in a folder:

``` bash
summarize_transcript --input_folder=INPUT_FOLDER_NAME --output_folder=OUTPUT_FOLDER_NAME --model PATH_TO_MODEL
```
