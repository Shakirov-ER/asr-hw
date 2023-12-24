from typing import List, NamedTuple
from collections import defaultdict

import torch

from .char_text_encoder import CharTextEncoder
from pyctcdecode import build_ctcdecoder
from string import ascii_lowercase, ascii_uppercase


class Hypothesis(NamedTuple):
    text: str
    prob: float


VOCAB = [""] + list(ascii_lowercase) + [" "]

DECODER = build_ctcdecoder(
    VOCAB,
    kenlm_model_path="language_models/lm.bin",
    alpha=0.5,
    beta=1.5,
)


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        result = ""
        for ind in inds:
            if self.ind2char[ind] == self.EMPTY_TOK:
                continue
            if len(result) == 0:
                result += self.ind2char[ind]
                continue
            if self.ind2char[ind] == result[-1]:
                continue
            result += self.ind2char[ind]
        return result

    def ctc_beam_search(
        self, probs: torch.tensor, probs_length, beam_size: int = 100
    ) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        probs = probs[:probs_length]
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = [Hypothesis("", 1)]
        for frame in probs:
            hypos = self.extend_and_merge(frame, hypos)
            hypos = self.truncate(hypos, beam_size)
        return sorted(hypos, key=lambda x: x.prob, reverse=True)

    def extend_and_merge(self, frame, hypos):
        new_hypos = defaultdict(float)
        for next_char_index, next_char_proba in enumerate(frame):
            next_char = self.ind2char[next_char_index]
            for hypo in hypos:
                if len(hypo.text) != 0:
                    last_char = hypo.text[-1]
                else:
                    last_char = ""

                if last_char == next_char or next_char == self.EMPTY_TOK:
                    new_text = hypo.text
                else:
                    new_text = hypo.text + next_char
                new_prob = hypo.prob * next_char_proba
                new_hypos[new_text] += new_prob
        hypos = [Hypothesis(text, prob) for text, prob in new_hypos.items()]
        return hypos

    def truncate(self, hypos, beam_size):
        return sorted(hypos, key=lambda x: x.prob, reverse=True)[:beam_size]

    def ctc_beam_search_lm(
        self, probs: torch.tensor, probs_length, beam_size: int = 512, n_best=1
    ) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        probs = probs[:probs_length]
        final_hypos: List[Hypothesis] = []

        text = DECODER.decode(probs.detach().numpy(), beam_width=beam_size)
        final_hypos = [Hypothesis(text=text, prob=1)]
        return final_hypos
