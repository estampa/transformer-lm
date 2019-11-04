import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import sentencepiece as spm
import torch
import numpy as np

from .model import Model, HParams
from .common import END_OF_LINE, END_OF_TEXT


class ModelWrapper:
    END_OF_LINE = END_OF_LINE
    END_OF_TEXT = END_OF_TEXT

    def __init__(self, model: Model, sp_model: spm.SentencePieceProcessor, device,
                 params: Optional[Dict]):
        self.model = model
        self.sp_model = sp_model
        self.params = params or {}
        self.device = device

    @classmethod
    def load(cls, root: Path):
        sp_model = spm.SentencePieceProcessor()
        sp_model.load(str(root / 'sp.model'))
        params = json.loads((root / 'params.json').read_text())
        hparams = params['hparams']
        hparams.setdefault('n_hidden', hparams['n_embed'])
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        model = Model(HParams(**hparams)).to(device)
        state = torch.load(root / 'model.pt', map_location='cpu')
        state_dict = fixed_state_dict(state['state_dict'])
        model.load_state_dict(state_dict)
        if 'seen_tokens' in state:
            params['seen_tokens'] = state['seen_tokens']
        return cls(model, sp_model, device, params=params)

    def tokenize(self, s: str) -> List[str]:
        return self.sp_model.EncodeAsPieces(s)

    def token_to_id(self, token: str) -> int:
        return self.sp_model.PieceToId(token)

    def id_to_token(self, token_id: int) -> str:
        return self.sp_model.IdToPiece(int(token_id))

    def get_log_probs(self, tokens: List[str]) -> torch.Tensor:
        """ Return a tensor with shape (len(tokens), len(self.sp_model)),
        with log-probabilities for tokens after each token in tokens.
        If this is a start of the text, you may want to prepend END_OF_TEXT:
        model.get_log_probs([model.END_OF_TEXT] + tokens).
        Use model.tokenize to obtain tokens.
        """
        assert len(tokens) <= self.model.hparams.n_ctx  # TODO
        ids = [self.token_to_id(t) for t in tokens]
        ctx = torch.LongTensor(ids).unsqueeze(0)
        with torch.no_grad():
            ctx = ctx.to(device=self.device)
            logits = self.model(ctx)['logits'].squeeze(0)
            return torch.log_softmax(logits, dim=1)

    def get_occurred_log_probs(
            self, tokens: List[str]) -> List[Tuple[float, str]]:
        """ Return a list of log probs of actually occurred tokens,
        starting from the second.
        """
        log_probs = self.get_log_probs(tokens)
        out = []
        for idx, token in enumerate(tokens[1:]):
            out.append((float(log_probs[idx, self.token_to_id(token)]), token))
        return out

    def get_next_top_k(
            self, tokens: List[str], top_k: int) -> List[Tuple[float, str]]:
        """ Return a list of top k tuples of log prob and token,
        for what would come after the last token.
        """
        next_log_probs = self.get_log_probs(tokens)[-1]
        next_k_log_probs = torch.topk(next_log_probs, k=top_k)
        return next_k_log_probs

    def generate_tokens(
            self,
            tokens_prefix: List[str],
            tokens_to_generate: int,
            top_k: int,
            temperature: 0.8
            ) -> List[str]:
        tokens = list(tokens_prefix)

        for i in range(tokens_to_generate):

            # generate TOP_K potential next tokens
            ntk_values, ntk_indices = self.get_next_top_k(tokens[- self.model.hparams.n_ctx:], top_k)
            if temperature > 0:
                # Dividim per la temperatura
                ntk_values_temp = torch.div(ntk_values, temperature)

                # Convertim de logaritme a exponencial
                ntk_values_exp = torch.exp(ntk_values_temp)

                # Normalitzem
                ntk_values_exp_sum = torch.sum(ntk_values_exp)
                ntk_values_mult = torch.div(ntk_values_exp, ntk_values_exp_sum)

                # Escollim un token en base a la distribució de probabilitats
                ntk_selected = torch.multinomial(ntk_values_mult, 1)
                next_token_n = ntk_indices[ntk_selected]
                next_token = self.id_to_token(next_token_n)
            else:
                # Escollim el token més probable (el primer, estan ordenats)
                next_token_n = ntk_indices[0]
                next_token = self.id_to_token(next_token_n)

            tokens.append(next_token)

            if next_token == END_OF_TEXT:
                break

        return tokens


def fixed_state_dict(state_dict):
    if all(k.startswith('module.') for k in state_dict):
        # legacy multi-GPU format
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    return state_dict
