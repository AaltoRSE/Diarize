import os
import click
import glob
import torch
import whisper
from pyannote.audio import Pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"


def align(transcription, diarization):
    """
    Align diarization with transcription.

    Transcription and diarization segments is measured using overlap in time.

    If no diarization segment overlaps with a given transcription segment, the speaker
    for that transcription segment is None.

    The output object is a dict of lists:

    {
    "start" : [0.0, 4.5, 7.0],
    "end"   : [3.3, 6.0, 10.0],
    "transcription" : ["This is first first speaker segment", "This is the second", "This is from an unknown speaker"],
    "speaker": ["SPEAKER_00", "SPEAKER_01", None]
    }

    Parameters
    ----------
    transcription : list
        Output of Whisper transcribe()
    diarization : list
        Output of Pyannote diarization()

    Returns
    -------
    dict
    """
    transcription_segments = [
        (segment["start"], segment["end"], segment["text"])
        for segment in transcription["segments"]
    ]
    diarization_segments = [
        (segment.start, segment.end, speaker)
        for segment, _, speaker in diarization.itertracks(yield_label=True)
    ]
    alignment = defaultdict(list)
    for transcription_start, transcription_end, text in transcription_segments:
        max_overlap, max_speaker = None, None
        for diarization_start, diarization_end, speaker in diarization_segments:
            overlap = compute_overlap(
                transcription_start,
                transcription_end,
                diarization_start,
                diarization_end,
            )
            if overlap == 0:
                continue
            if max_overlap is None or overlap > max_overlap:
                max_overlap, max_speaker = overlap, speaker

        transcription_start = seconds_to_human_readable(transcription_start)
        transcription_end = seconds_to_human_readable(transcription_end)

        alignment["start"].append(transcription_start)
        alignment["end"].append(transcription_end)
        alignment["speaker"].append(max_speaker)
        alignment["transcription"].append(text.strip())

    return alignment



@click.command(context_settings={'show_default': True})
@click.option('--min_speakers', default=2, help='Mininum number of speakers')
@click.option('--max_speakers', default=5, help='Maximum number of speakers')
@click.option('--input_file', default=None, help='Input file name')
@click.option('--input_folder', default=None, help='Input file name')
@click.option('--output_folder', default=None, help='Output file name')
@click.option('--model', help='Whisper model', default="base",
    type=click.Choice(whisper.available_models(), case_sensitive=False)
)
@click.option('--skip_existing', default=True, help='Skip audio file if output file exists')
@click.option('--hugging_face_token', default=None, help='Hugging face access token (required on first run)')
def transcribe_and_diarize_audio(
    min_speakers, max_speakers, input_file, input_folder, output_folder,
    model, skip_existing, hugging_face_token
):

    if input_file is None and input_folder is None:
        raise click.BadParameter("Provide either input file or input folder")

    if input_file is not None:
        input_files = [input_file]
        if output_folder is None:
            output_folder = "./"

    if input_folder is not None:
        input_files = glob.glob(os.path.join(input_folder, '*.wav'))
        input_files.extend(glob.glob(os.path.join(input_folder, '*.WAV')))
        if output_folder is None:
            output_folder = input_folder
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for infile in input_files:
        outfile = "".join(os.path.basename(infile).split(".")[:-1]) + "_diarized.txt"
        outfile = os.path.join(output_folder, outfile)

        if skip_existing and os.path.exists(outfile):
            continue

        print("Loading models")
        if hugging_face_token is None:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                                use_auth_token=hugging_face_token)
        else:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
        pipeline.to(torch.device(device))

        model = whisper.load_model(model, device=device)

        print(f"Transcribing {infile}")
        asr_result = model.transcribe(infile)
        print(f"Diarizing {infile}")
        diarization_result = pipeline(
            infile, min_speakers=min_speakers, max_speakers=max_speakers
        )
        final_result = align(asr_result, diarization_result)

        with open(outfile, "w") as out_fp:
            for start, end, speaker, text in zip(final_result["start"], final_result["end"], final_result["speaker"], final_result["transcription"]):
                line = f'{seg["start"]:.2f} {seg["end"]:.2f} {speaker} {text}\n'
                print(line)
                out_fp.write(line)
        

if __name__ == '__main__':
    transcribe_and_diarize_audio()

