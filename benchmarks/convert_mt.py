"""Convert the translation candidates to CTranslate2 once, into models/mt/.

Run from the project folder:
    python benchmarks/convert_mt.py
    python benchmarks/convert_mt.py opus-mt-hy-en
"""

import argparse
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

# Run as a script, so the program's own package isn't on the path yet.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asr_benchmark import load_hf_token  # noqa: E402
from live_hy_en.translator import MT_MODELS, MT_MODELS_DIR, MtModel  # noqa: E402

# Weights are stored as float16 to halve their size; CTranslate2 converts them
# to the compute type chosen at load time (int8 on the CPU).
STORED_TYPE = "float16"


def tokenizer_files(model: MtModel) -> list[str]:
    """Return the sentencepiece files the translator needs beside the weights."""
    prompt_format = model.prompt_format
    return sorted({prompt_format.source_spm, prompt_format.target_spm})


def convert_hub_model(
    hub_id: str,
    output_dir: Path,
    copy_files: list[str],
    add_files: Callable[[Path], None] | None = None,
) -> Path:
    """Convert a Hugging Face model to CTranslate2 once; return its folder.

    `add_files` can write extra files the runtime needs into the new folder.
    """
    if output_dir.is_dir():
        print(f"{output_dir.name}: already converted in {output_dir}")
        return output_dir
    from ctranslate2.converters import TransformersConverter

    # Convert into a side folder first, so a failed run never looks finished.
    partial_dir = output_dir.with_name(f"{output_dir.name}.part")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"{output_dir.name}: converting {hub_id} ...", flush=True)
    try:
        TransformersConverter(
            hub_id,
            copy_files=copy_files,
            load_as_float16=True,
            low_cpu_mem_usage=True,
        ).convert(str(partial_dir), quantization=STORED_TYPE, force=True)
        if add_files is not None:
            add_files(partial_dir)
    except Exception:
        shutil.rmtree(partial_dir, ignore_errors=True)
        raise
    partial_dir.rename(output_dir)
    print(f"{output_dir.name}: saved to {output_dir}")
    return output_dir


def convert(name: str, models_dir: Path = MT_MODELS_DIR) -> Path:
    """Convert one translation candidate, unless it is already there."""
    model = MT_MODELS[name]
    return convert_hub_model(model.hub_id, models_dir / name, tokenizer_files(model))


def parse_model_names(
    argv: Sequence[str] | None, candidates: Sequence[str], description: str
) -> list[str]:
    """Return the candidates named on the command line, or all of them.

    Shared with convert_asr.py; an unknown name stops with a usage error.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "models", nargs="*", help=f"default: all of {', '.join(candidates)}"
    )
    names = parser.parse_args(argv).models
    unknown = [name for name in names if name not in candidates]
    if unknown:
        parser.error(f"unknown model(s): {', '.join(unknown)}")
    return names or list(candidates)


def convert_each(names: Sequence[str], convert_one: Callable[[str], Path]) -> int:
    """Convert each model, going on after a failure; return 1 if any failed.

    Shared with convert_asr.py.
    """
    # Puts HF_TOKEN from .env into the environment, where the Hub client reads it.
    load_hf_token()
    failed = []
    for name in names:
        try:
            convert_one(name)
        except Exception as error:
            # One model failing (e.g. a removed upload) shouldn't stop the others.
            print(f"{name}: conversion failed: {error}")
            failed.append(name)
    if failed:
        print(f"Failed: {', '.join(failed)}")
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Convert each chosen candidate; return 1 if any of them failed."""
    names = parse_model_names(argv, list(MT_MODELS), __doc__.splitlines()[0])
    return convert_each(names, lambda name: convert(name, MT_MODELS_DIR))


if __name__ == "__main__":
    sys.exit(main())
