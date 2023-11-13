import click
import glob
import os
import torch

from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


device = "cuda" if torch.cuda.is_available() else "cpu"

@click.command(context_settings={'show_default': True})
@click.option('--input_file', default=None, help='Input file name')
@click.option('--input_folder', default=None, help='Input file name')
@click.option('--output_folder', default=None, help='Output file name')
@click.option('--model', required=True, help='Summarization model file name')
@click.option('--skip_existing', default=True, help='Skip audio file if output file exists')
@click.option('--use_cuda', default=False, help='Use cuda to run the model')
def summarize(input_file, input_folder, output_folder, model, skip_existing, use_cuda):
    #from gpt4all import GPT4All
    
    if input_file is None and input_folder is None:
        raise click.BadParameter("Provide either input file or input folder")

    if input_file is not None:
        input_files = [input_file]
        if output_folder is None:
            output_folder = "./"

    if input_folder is not None:
        input_files = glob.glob(os.path.join(input_folder, '*.txt'))
        if output_folder is None:
            output_folder = input_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for infile in input_files:
        if "_summary.txt" in infile:
            continue
        if "_diarized.txt" in infile:
            outfile = infile.replace('_diarized.txt', '_summary.txt')
        else:
            outfile = infile.replace('.txt', '_summary.txt')

        if skip_existing and os.path.exists(outfile):
            continue

        with open(infile) as f:
            text = f.read()

        text_splitter = RecursiveCharacterTextSplitter()
        texts = text_splitter.split_text(text)
        docs = [Document(page_content=t) for t in texts]

        callbacks = [StreamingStdOutCallbackHandler()]
        llm = GPT4All(model=model, callbacks=callbacks, verbose=True)
        if use_cuda:
            chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True, device=device)
        else:
            chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)

        output_summary = chain.run(docs)

        with open(outfile, "w") as out_fp:
            out_fp.write(output_summary)


if __name__ == '__main__':
    summarize()

