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

from transformers import LukeConfig, LukeForEntityPairClassification, LukeTokenizer, RobertaTokenizer
from transformers.tokenization_utils_base import AddedToken


@torch.no_grad()
def convert_luke_checkpoint(checkpoint_path, metadata_path, entity_vocab_path, pytorch_dump_folder_path, model_size):
    # Load configuration defined in the metadata file
    with open(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)
    bert_model_name = metadata['model_config'].pop('bert_model_name')
    config = LukeConfig(use_entity_aware_attention=True, **metadata["model_config"])

    # Load in the weights from the checkpoint_path
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Load the entity vocab file
    entity_vocab = load_entity_vocab(entity_vocab_path)

    tokenizer = RobertaTokenizer.from_pretrained(bert_model_name)

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

    config.num_labels = 42
    config.id2label = {
        0: "no_relation",
        1: "org:alternate_names",
        2: "org:city_of_headquarters",
        3: "org:country_of_headquarters",
        4: "org:dissolved",
        5: "org:founded",
        6: "org:founded_by",
        7: "org:member_of",
        8: "org:members",
        9: "org:number_of_employees/members",
        10: "org:parents",
        11: "org:political/religious_affiliation",
        12: "org:shareholders",
        13: "org:stateorprovince_of_headquarters",
        14: "org:subsidiaries",
        15: "org:top_members/employees",
        16: "org:website",
        17: "per:age",
        18: "per:alternate_names",
        19: "per:cause_of_death",
        20: "per:charges",
        21: "per:children",
        22: "per:cities_of_residence",
        23: "per:city_of_birth",
        24: "per:city_of_death",
        25: "per:countries_of_residence",
        26: "per:country_of_birth",
        27: "per:country_of_death",
        28: "per:date_of_birth",
        29: "per:date_of_death",
        30: "per:employee_of",
        31: "per:origin",
        32: "per:other_family",
        33: "per:parents",
        34: "per:religion",
        35: "per:schools_attended",
        36: "per:siblings",
        37: "per:spouse",
        38: "per:stateorprovince_of_birth",
        39: "per:stateorprovince_of_death",
        40: "per:stateorprovinces_of_residence",
        41: "per:title",
    }
    config.label2id = {v: k for k, v in config.id2label.items()}

    orig_entity_emb = state_dict["entity_embeddings.entity_embeddings.weight"]
    new_entity_emb = torch.zeros(config.entity_vocab_size, config.entity_emb_size)
    new_entity_emb[entity_vocab["[PAD]"]] = orig_entity_emb[0]
    new_entity_emb[entity_vocab["[MASK]"]] = orig_entity_emb[1]
    new_entity_emb[entity_vocab["[MASK2]"]] = orig_entity_emb[2]
    state_dict["entity_embeddings.entity_embeddings.weight"] = new_entity_emb

    new_state_dict = {}
    for key, value in state_dict.items():
        if not key.startswith("classifier."):
            key = "luke." + key
        new_state_dict[key] = value

    model = LukeForEntityPairClassification(config=config).eval()

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    assert len(missing_keys) == 1 and missing_keys[0] == "luke.embeddings.position_ids"
    assert not unexpected_keys

    # Check outputs
    tokenizer = LukeTokenizer.from_pretrained(pytorch_dump_folder_path, task="entity_pair_classification")

    text = "She is an American actress and singer ."
    entity_spans = [(0, 3), (31, 37)]
    encoding = tokenizer(text, entity_spans=entity_spans, add_prefix_space=True, return_tensors="pt")

    outputs = model(**encoding)

    # The following values were obtained from the following URL::
    # https://colab.research.google.com/drive/1tNngKNfZV6lGNWp2JhhSgpPKdLCfvlQk?usp=sharing
    assert outputs.logits.shape == torch.Size((1, 42))
    assert torch.allclose(outputs.logits[0, :3], torch.tensor([7.1670, -1.9315, -3.4485]), atol=1e-4)

    print("Saving PyTorch model to {}".format(pytorch_dump_folder_path))
    model.save_pretrained(pytorch_dump_folder_path)
    tokenizer.save_pretrained(pytorch_dump_folder_path)


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
