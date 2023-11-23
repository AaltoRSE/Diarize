from setuptools import setup, find_packages
from os.path import join, dirname

with open("README.md", "r") as fh:
    long_description = fh.read()

requirementstxt = join(dirname(__file__), "requirements.txt")
with open(requirementstxt, "r") as file:
    requirements = [line.strip() for line in file if line.strip()]

setup(
    name='transcribe_and_diarize',
    version="1.0.0",
    description='A script for trancribing and diarizing wav formatted audio files.',
    long_description_content_type='text/markdown',
    author='Jarno Rantaharju',
    author_email='jarno.rantaharju@aalto.fi',
    url='https://github.com/AaltoRSE/Diarize',
    packages=find_packages(where='.'),
    py_modules=[
        "transcribe_and_diarize",
        "summarize_transcript"
    ],
    entry_points={
       'console_scripts': [
            'transcribe_and_diarize=transcribe_and_diarize:transcribe_and_diarize_audio',
            'summarize_transcript=summarize_transcript:summarize',
        ],
    },
    python_requires="<3.12",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "Operating System :: OS Independent",
    ],
)
