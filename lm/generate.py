from pathlib import Path

import re
import time

import fire

from .fire_utils import only_allow_defined_args
from .inference import ModelWrapper
from .common import END_OF_LINE, END_OF_TEXT

import sequence_common


def find_similar(text, folder):
    seqfind = sequence_common.SequenceFinder(folder)
    seqfind.use_cosine()
    # linies = re.split("[\n\.]", text_net[len(context):])
    linies = re.split("[\.\n]", text)
    for linia in linies:
        if linia.strip() and len(linia.strip()) > 1:
            closest, closest_dist, closest_file = seqfind.closest_cosine(linia)
            # print(closest)
            if closest_dist < 0.3:
                print("\n* {:<64} | {:2f} {:<64} ({})".format(linia[:64], round(closest_dist, 2), closest[:64], closest_file[:64]))
                # print("--\n" + linia + "\n" + closest + "\n" + closest_file[:48] + " - " + str(closest_dist))


def gen_main(model_path, prefix, tokens_to_generate=200, top_k=32, temperature=0.8, num=1, context="", save=False, originals=""):
    print(f'loading model from {model_path}')
    mw = ModelWrapper.load(Path(model_path))

    print(f'generating text for prefix {prefix}')
    if len(str(context)) > 0:
        context_tokens = mw.tokenize(str(context)) + [END_OF_LINE]
    else:
        context_tokens = []
    tokens = [END_OF_TEXT] + context_tokens + mw.tokenize(str(prefix))

    filename_base = "gpt2-{}-{}".format(model_path.replace("/", "_"), time.strftime("%Y%m%d-%H%M"))

    for i in range(0,num):
        tokens_gen = mw.generate_tokens(tokens, tokens_to_generate, top_k, temperature)
        # print(mw.sp_model.DecodePieces(tokens_gen))  # No mostra salts de línia
        text_gen = "".join(tokens_gen[len(context_tokens):]).replace("▁", " ").replace(END_OF_LINE, "\n").replace(END_OF_TEXT, "")
        print("----")
        print(text_gen)
        if save:
            filename = "{}-{}.txt".format(filename_base, str(i).zfill(2))
            with open(filename, "w") as output_file:
                output_file.write(text_gen)

        if originals:
            find_similar(text_gen[len(context_tokens):], originals)


def fire_gen_main():
    fire.Fire(only_allow_defined_args(gen_main))
