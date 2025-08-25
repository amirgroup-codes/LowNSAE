from typing import Optional

import numpy as np
import polars as pl
import torch
from scipy.sparse import csr_matrix
from transformers import PreTrainedModel, PreTrainedTokenizer


def get_layer_activations(
    tokenizer: PreTrainedTokenizer,
    plm: PreTrainedModel,
    seqs: list[str],
    layer: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Get the activations of a specific layer in a pLM model. Let:

    ```
    N = len(seqs)
    L = max(len(seq) for seq in seqs) + 2 # +2 for BOS and EOS tokens
    D_MODEL = the layer dimension of the pLM, i.e. "Embedding Dim" column here
        https://github.com/facebookresearch/esm/tree/main?tab=readme-ov-file#available-models
    ```

    The output tensor is of shape (N, L, D_MODEL)

    Args:
        tokenizer: The tokenizer to use.
        plm: The pLM model to get the activations from.
        seqs: The sequences to get the activations for.
        layer: The layer to get the activations from.
        device: The device to use.

    Returns:
        The (N, L, D_MODEL) activations of the specified layer.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = tokenizer(seqs, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = plm(**inputs, output_hidden_states=True)
    layer_acts = outputs.hidden_states[layer]
    del outputs
    return layer_acts


def tensor_to_sparse_matrix(T):
    return csr_matrix(T.cpu().numpy().astype(np.float32))


def sparse_matrix_to_tensor(sparse_matrix):
    coo = sparse_matrix.tocoo()
    values = torch.tensor(coo.data, dtype=torch.float32)
    indices = torch.tensor([coo.row, coo.col], dtype=torch.int64)
    return torch.sparse_coo_tensor(indices, values, coo.shape)


from torch.utils.data import random_split
def train_val_test_split(dataset, val_frac=0.1, test_frac=0.1):
    total_len = len(dataset)
    val_len = int(total_len * val_frac)
    test_len = int(total_len * test_frac)
    train_len = total_len - val_len - test_len

    return random_split(dataset, [train_len, val_len, test_len])


def parse_swissprot_annotation(annotation_str: str, header: str) -> list[dict]:
    """
    Parse a SwissProt annotation string like this:

    ```
    MOTIF 119..132; /note="JAMM motif"; /evidence="ECO:0000255|PROSITE-ProRule:PRU01182"
    ```
    where MOTIF is the header argument.

    Returns:
        [{
            "start": 119,
            "end": 132,
            "note": "JAMM motif",
            "evidence": "ECO:0000255|PROSITE-ProRule:PRU01182",
        }]
    """
    res = []
    occurrences = [o for o in annotation_str.split(header + " ") if len(o) > 0]
    for o in occurrences:
        parts = [p for p in o.split("; /") if len(p) > 0]

        pos_part = parts[0]
        coords = pos_part.split("..")

        annotations_dict = {}
        for part in parts[1:]:
            key, value = part.split("=", 1)
            annotations_dict[key] = value.replace('"', "").replace(";", "").strip()

        try:
            list(map(int, coords))
        except ValueError:
            continue
        if len(annotations_dict) == 0:
            continue

        res.append(
            {
                "start": int(coords[0]),
                "end": int(coords[1]) if len(coords) > 1 else int(coords[0]),
                **annotations_dict,
            }
        )
    return res
