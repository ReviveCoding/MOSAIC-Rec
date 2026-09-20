"""Known-answer Cloze masking contract for the local BERT4Rec pilot."""

import unittest

import torch

from mosaic_rec.pilot_bert4rec import BERT4Rec, cloze_batch


class BERT4RecClozeContract(unittest.TestCase):
    def test_final_target_is_masked_and_padding_is_not(self) -> None:
        torch.manual_seed(4)
        sequence = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
        length = torch.tensor([2, 3])
        masked, selected, labels = cloze_batch(sequence, length, mask_id=7)
        self.assertEqual(masked[0, 1].item(), 7)
        self.assertEqual(masked[1, 2].item(), 7)
        self.assertFalse(selected[0, 2:].any().item())
        self.assertFalse(selected[1, 3].item())
        self.assertTrue(torch.equal(labels, sequence[selected]))

    def test_bidirectional_model_shape(self) -> None:
        model = BERT4Rec(catalog_size=6, width=8, length=4).eval()
        with torch.inference_mode():
            states = model(torch.tensor([[1, 2, model.mask_id, 0]]))
        self.assertEqual(tuple(states.shape), (1, 4, 8))
        self.assertTrue(torch.isfinite(states).all().item())

    def test_fit_hidden_item_has_no_cloze_gradient(self) -> None:
        torch.manual_seed(3)
        model = BERT4Rec(catalog_size=6, width=8, length=4)
        sequence = torch.tensor([[1, 2, 3, 0]])
        masked, selected, labels = cloze_batch(
            sequence, torch.tensor([3]), model.mask_id
        )
        train_ids = torch.tensor([1, 2, 3])
        lookup = torch.full((8,), -1, dtype=torch.long)
        lookup[train_ids] = torch.arange(3)
        logits = model(masked)[selected] @ model.item(train_ids).T
        loss = torch.nn.functional.cross_entropy(logits, lookup[labels])
        loss.backward()
        self.assertEqual(model.item.weight.grad[6].abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
