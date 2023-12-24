import unittest

from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
import torch


class TestTextEncoder(unittest.TestCase):
    def test_ctc_decode(self):
        text_encoder = CTCCharTextEncoder()
        text = (
            "i^^ ^w^i^sss^hhh^   i ^^^s^t^aaaar^teee^d "
            "dddddd^oooo^in^g tttttttth^iiiis h^^^^^^^^w^ e^a^r^li^er"
        )
        true_text = "i wish i started doing this hw earlier"
        inds = [text_encoder.char2ind[c] for c in text]
        decoded_text = text_encoder.ctc_decode(inds)
        self.assertIn(decoded_text, true_text)

    def test_beam_search(self):
        text_encoder = CTCCharTextEncoder(alphabet=["a", "b", "c"])
        probs = torch.tensor(
            [
                [0.1, 0.2, 0.1, 0.6],
                [0.2, 0.2, 0.5, 0.1],
                [0.1, 0.5, 0.1, 0.3],
            ]
        )
        hypos = text_encoder.ctc_beam_search(probs, 3, beam_size=1)  # simple argmax
        self.assertTrue(hypos[0].text == "cba")
        self.assertEqual(hypos[0].prob, 0.6 * 0.5 * 0.5)

        hypos = text_encoder.ctc_beam_search(probs, 3, beam_size=4)
        self.assertTrue(hypos[0].text == "ca")
        self.assertTrue(
            hypos[0].prob
            - (
                0.6 * 0.2 * 0.1  # ca^
                + 0.6 * 0.2 * 0.5  # c^a
                + 0.1 * 0.1 * 0.5  # ^ca
                + 0.6 * 0.1 * 0.5  # cca
                + 0.6 * 0.2 * 0.5  # caa
            )
            < 1e-5
        )
