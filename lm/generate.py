from pathlib import Path

import fire

from .fire_utils import only_allow_defined_args
from .inference import ModelWrapper
from .common import END_OF_LINE, END_OF_TEXT


def gen_main(model_path, prefix, tokens_to_generate=42, top_k=32, temperature=0.8, context=""):
    print(f'loading model from {model_path}')
    mw = ModelWrapper.load(Path(model_path))

    print(f'generating text for prefix {prefix}')
    if len(str(context)) > 0:
        context_tokens = mw.tokenize(str(context)) + [END_OF_LINE]
    else:
        context_tokens = []
    tokens = [END_OF_TEXT] + context_tokens + mw.tokenize(str(prefix))

    tokens_gen = mw.generate_tokens(tokens, tokens_to_generate, top_k, temperature)
    # print(mw.sp_model.DecodePieces(tokens_gen))  # No mostra salts de línia
    text_gen = "".join(tokens_gen[len(context_tokens):]).replace("▁", " ").replace(END_OF_LINE, "\n").replace(END_OF_TEXT, "")
    print(text_gen)


def fire_gen_main():
    fire.Fire(only_allow_defined_args(gen_main))
