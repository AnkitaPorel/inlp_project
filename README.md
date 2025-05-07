# Bengali Poem Generator

## How to use

You need the UV package/project manager to install the dependencies.  
You can get it from [here](https://docs.astral.sh/uv/getting-started/installation/).

Set up the environment. (Only once)

```bash
uv venv
# .venv/Scripts/activate # for Windows
.venv/bin/activate # for Linux
# uv pip install torch --index-url https://download.pytorch.org/whl/cu118 --link-mode=symlink
uv sync --link-mode=symlink --extra=cpu   # for CPU only
uv sync --link-mode=symlink --extra=cu124 # for CUDA support
```

To run any script, append `uv run` before the `python` command.

## The Notebooks

The `notebooks` folder contains the Jupyter notebooks (self-explanatory) for training and inference.

Most notebooks were run on Kaggle, with a few which were run on Google Colab. _(The differences are evident in the notebooks.)_

## Models

The models are available on HuggingFace. You can use them directly from the `transformers` library using the following names:

```properties
GEMMA_3_4B_CHAT_POEM = "Ankita-Porel/gemma3-4b-bn-chat-poem-ft"
GEMMA_3_4B_WIKI_POEM = "Venkat423/Gemma-3-4b-finetuned-final"
SARVAM_1_CHAT_POEM = "Ankita-Porel/sarvam1-bn-chat-poem-ft"
SARVAM_1_WIKI_POEM = "Ankita-Porel/sarvam1-wiki-poem-bn"
```

## Gated Repos on HuggingFace

Some files like [bengali-poem-generator/utils/compare_tokenizers.py](./bengali-poem-generator/utils/compare_tokenizers.py),
require access Gated Public Repositories on HuggingFace.

You must log in to HuggingFace using `huggingface-cli`.

1. [Generate a token with the `Read access to contents of all public gated repos you can access` permission](https://huggingface.co/settings/tokens/new?tokenType=fineGrained) on HuggingFace. (Account Required)
2. Run `huggingface-cli login` and paste the token generated in the previous step.
