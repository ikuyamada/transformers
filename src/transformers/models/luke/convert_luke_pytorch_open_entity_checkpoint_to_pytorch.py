# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert LUKE checkpoint."""

import argparse
import json
import os

import torch

from transformers import LukeConfig, LukeForEntityClassification, LukeTokenizer, RobertaTokenizer
from transformers.tokenization_utils_base import AddedToken


@torch.no_grad()
def convert_luke_checkpoint(checkpoint_path, metadata_path, entity_vocab_path, pytorch_dump_folder_path, model_size):
    # Load configuration defined in the metadata file
    with open(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)
    config = LukeConfig(use_entity_aware_attention=True, **metadata["model_config"])

    # Load in the weights from the checkpoint_path
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Load the entity vocab file
    entity_vocab = load_entity_vocab(entity_vocab_path)

    tokenizer = RobertaTokenizer.from_pretrained(metadata["model_config"]["bert_model_name"])

    # Add special tokens to the token vocabulary for downstream tasks
    entity_token_1 = AddedToken("<ent>", lstrip=False, rstrip=False)
    entity_token_2 = AddedToken("<ent2>", lstrip=False, rstrip=False)
    tokenizer.add_special_tokens(dict(additional_special_tokens=[entity_token_1, entity_token_2]))
    config.vocab_size += 2

    print(f"Saving tokenizer to {pytorch_dump_folder_path}")
    tokenizer.save_pretrained(pytorch_dump_folder_path)
    with open(os.path.join(pytorch_dump_folder_path, LukeTokenizer.vocab_files_names["entity_vocab_file"]), "w") as f:
        json.dump(entity_vocab, f)

    tokenizer = LukeTokenizer.from_pretrained(pytorch_dump_folder_path)

    config.num_labels = 9

    orig_word_emb = state_dict["embeddings.word_embeddings.weight"]
    state_dict["embeddings.word_embeddings.weight"] = torch.cat([orig_word_emb, torch.zeros(1, orig_word_emb.size(1))])

    orig_entity_emb = state_dict["entity_embeddings.entity_embeddings.weight"]
    new_entity_emb = torch.zeros(config.entity_vocab_size, config.entity_emb_size)
    new_entity_emb[entity_vocab["[PAD]"]] = orig_entity_emb[0]
    new_entity_emb[entity_vocab["[MASK]"]] = orig_entity_emb[1]
    state_dict["entity_embeddings.entity_embeddings.weight"] = new_entity_emb

    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("typing."):
            key = "classifier." + key[7:]
        else:
            key = "luke." + key
        new_state_dict[key] = value

    model = LukeForEntityClassification(config=config).eval()

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    assert len(missing_keys) == 1 and missing_keys[0] == "luke.embeddings.position_ids"
    assert not unexpected_keys

    # Check outputs
    tokenizer = LukeTokenizer.from_pretrained(pytorch_dump_folder_path, task="entity_classification")

    text = "Top seed Ana Ivanovic said on Thursday she could hardly believe her luck as a fortuitous netcord helped the new world number one avoid a humiliating second- round exit at Wimbledon ."
    entity_spans = [(39, 42)]
    encoding = tokenizer(text, entity_spans=entity_spans, add_prefix_space=True, return_tensors="pt")

    outputs = model(**encoding)

    # The following values were obtained from the following URL:
    # https://colab.research.google.com/drive/1RcKoLtFjXjZQY0-lCYKE0U2-FwYenFs6?usp=sharing
    assert outputs.logits.shape == torch.Size((1, 9))
    assert torch.allclose(outputs.logits[0, :3], torch.tensor([-5.6505, -5.9512, -5.5152]), atol=1e-4)

    print("Saving PyTorch model to {}".format(pytorch_dump_folder_path))
    model.save_pretrained(pytorch_dump_folder_path)


def load_entity_vocab(entity_vocab_path):
    entity_vocab = {}
    with open(entity_vocab_path, "r", encoding="utf-8") as f:
        for (index, line) in enumerate(f):
            title, _ = line.rstrip().split("\t")
            entity_vocab[title] = index

    return entity_vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--checkpoint_path", type=str, help="Path to a pytorch_model.bin file.")
    parser.add_argument(
        "--metadata_path", default=None, type=str, help="Path to a metadata.json file, defining the configuration."
    )
    parser.add_argument(
        "--entity_vocab_path",
        default=None,
        type=str,
        help="Path to an entity_vocab.tsv file, containing the entity vocabulary.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to where to dump the output PyTorch model."
    )
    parser.add_argument(
        "--model_size", default="base", type=str, choices=["base", "large"], help="Size of the model to be converted."
    )
    args = parser.parse_args()
    convert_luke_checkpoint(
        args.checkpoint_path,
        args.metadata_path,
        args.entity_vocab_path,
        args.pytorch_dump_folder_path,
        args.model_size,
    )
