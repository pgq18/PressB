#!/usr/bin/env python3
"""Build the PressB pi0.5 tokenizer from OpenPI's official public SPM artifact.

Run in the isolated training conda environment (Transformers 5.3.0). No model
weights or tokenizer files need to be transferred from the local workstation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile
import time
from urllib.request import urlopen


SOURCE = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
SOURCE_SHA256 = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
CONFIG = {
    "tokenizer_class": "GemmaTokenizer", "pad_token": "<pad>",
    "bos_token": "<bos>", "eos_token": "<eos>", "unk_token": "<unk>",
    "mask_token": "<mask>", "add_bos_token": True, "add_eos_token": False,
    "padding_side": "right", "model_max_length": 2048,
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompts():
    rng = random.Random(717)
    tasks = ["Stack the red cube on the blue cube.", "Put all objects in the black box."]
    result = []
    for index in range(100):
        state = " ".join(str(rng.randrange(256)) for _ in range(32))
        result.append(f"Task: {tasks[index % 2]}, State: {state};\nAction: ")
    result.extend([
        "Task: Pick_up the red cube.\nAction: ",
        "Task: 把红方块放到蓝方块上, State: -1 0 255;\nAction: ",
        "a\tb", "  Hello   world.\n",
    ])
    return result


def build(destination, *, source_model=None, reference=None):
    import sentencepiece
    from tokenizers.processors import TemplateProcessing
    import transformers
    from transformers import AutoTokenizer

    destination = destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing tokenizer directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pi05-tokenizer-", dir=destination.parent) as temporary:
        work = Path(temporary)
        source = work / "tokenizer.model"
        if source_model is not None:
            shutil.copyfile(source_model.expanduser().resolve(strict=True), source)
        else:
            for attempt in range(4):
                try:
                    with urlopen(SOURCE, timeout=90) as stream, source.open("wb") as output:
                        shutil.copyfileobj(stream, output)
                    break
                except OSError:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
        if digest(source) != SOURCE_SHA256:
            raise ValueError("Official SPM checksum differs from the verified PressB pi0.5 tokenizer")
        (work / "tokenizer_config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
        tokenizer = AutoTokenizer.from_pretrained(work, local_files_only=True)
        # Match the existing PressB tokenizer's exact single/pair BOS template.
        tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
            single="<bos> $A", pair="<bos> $A $B", special_tokens=[("<bos>", 2)],
        )
        tokenizer.backend_tokenizer.save(str(work / "tokenizer.json"))
        # Reopen the actual files to be used by the trainer, rather than only
        # testing the transient converter instance.
        tokenizer = AutoTokenizer.from_pretrained(work, local_files_only=True)
        spm = sentencepiece.SentencePieceProcessor(model_file=str(source))
        if len(tokenizer) != 257152 or spm.vocab_size() != 257152:
            raise ValueError("Unexpected PaliGemma vocabulary size")
        if [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id,
                tokenizer.unk_token_id] != [0, 1, 2, 3]:
            raise ValueError("Unexpected special token IDs")
        corpus = prompts()
        reference_tokenizer = None if reference is None else AutoTokenizer.from_pretrained(
            reference.expanduser().resolve(strict=True), local_files_only=True,
        )
        for index, prompt in enumerate(corpus):
            actual = tokenizer.encode(prompt)
            if actual != spm.encode(prompt, add_bos=True):
                raise ValueError(f"SPM/Hugging Face token IDs differ for validation prompt {index}")
            if reference_tokenizer is not None and actual != reference_tokenizer.encode(prompt):
                raise ValueError(f"Existing/rebuilt token IDs differ for validation prompt {index}")
        padded = tokenizer(corpus[:2], padding="max_length", max_length=200,
                           truncation=True, return_tensors="pt")
        if tuple(padded["input_ids"].shape) != (2, 200):
            raise ValueError("Unexpected padded tokenizer shape")
        evidence = {
            "artifact_origin": SOURCE,
            "source_documentation": "https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/tokenizer.py",
            "transformers_version": transformers.__version__,
            "matching_prompt_count": len(corpus), "reference_checked": reference is not None,
            "vocab_size": len(tokenizer), "pad": 0, "eos": 1, "bos": 2, "unk": 3,
            "files": {name: digest(work / name) for name in
                      ("tokenizer.model", "tokenizer.json", "tokenizer_config.json")},
        }
        (work / "provenance.json").write_text(json.dumps(evidence, indent=2) + "\n")
        work.rename(destination)
    print(json.dumps({"destination": str(destination), **evidence}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, help="Reuse an existing official SPM for offline self-testing")
    parser.add_argument("--reference", type=Path, help="Optional existing HF tokenizer to compare token IDs")
    args = parser.parse_args()
    build(args.destination, source_model=args.source_model, reference=args.reference)


if __name__ == "__main__":
    main()
