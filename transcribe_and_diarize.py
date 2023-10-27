import os
import click
import glob
import torch
import whisper
from pyannote.audio import Pipeline
from pyannote_whisper.utils import diarize_text

device = "cuda" if torch.cuda.is_available() else "cpu"

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
        final_result = diarize_text(asr_result, diarization_result)

        with open(outfile, "w") as out_fp:
            for seg, spk, sent in final_result:
                line = f'{seg.start:.2f} {seg.end:.2f} {spk} {sent}\n'
                print(line)
                out_fp.write(line)
        

if __name__ == '__main__':
    transcribe_and_diarize_audio()

