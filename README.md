# polyBERT: A chemical language model to enable fully machine-driven ultrafast polymer informatics

This repository holds the code to train polyBERT. polyBERT is a chemical language machine learning model that has learned the PSMILES chemical language of polymers. It can be used for generating polymer fingerprints (based on PSMILES strings) at unprecedented speed. polyBERT enables ultrafast polymer informatics and fully machine-driven property predictions. polyBERT runs on GPUs and CPUs and is seamlessly scalable on cloud infrastructure. The polyBERT paper is available at [ArXiv](https://arxiv.org/abs/). 

**Abstract:** Polymers are a vital part of everyday life. Their chemical universe is so large that it presents unprecedented opportunities as well as significant challenges to identify suitable application-specific candidates. We present a complete end-to-end machine-driven polymer informatics pipeline that can search this space for suitable candidates at unprecedented speed and accuracy. This pipeline includes a polymer chemical fingerprinting capability called polyBERT (inspired by Natural Language Processing concepts), and a multitask learning approach that maps the polyBERT fingerprints to a host of properties. polyBERT is a chemical linguist that treats the chemical structure of polymers as a chemical language. The present approach outstrips the best presently available concepts for polymer property prediction based on handcrafted fingerprint schemes in speed by two orders of magnitude while preserving accuracy, thus making it a strong candidate for deployment in scalable architectures including cloud infrastructures.


The trained polyBERT model is available at the [Hugginface hub](https://huggingface.co/kuelumbus/polyBERT) for download.

## polyBERT polymer fingerprints

I strongly recommend using the `psmiles` Python package for generating polymer fingerprints with polyBERT. It takes care of downloading polyBERT, canonicalization of PSMILES strings, and computing polyBERT fingerprints. Please see the [PSMILES](https://github.com/Ramprasad-Group/psmiles) on GitHub.

## Prepare Python environment

1. Make sure [Poetry](https://python-poetry.org/docs/) is installed

2. Clone the code 
```bash
git clone git@github.com:Ramprasad-Group/polyBERT.git
cd polyBERT
```

3. Install environment

```bash
poetry config virtualenvs.in-project true
poetry install
```

## Train polyBERT

1. Download the 100 million PSMILES strings from [zenodo](https://zenodo.org/record/7124188) and place it in the `polyBERT` directory. You need the `generated_polymer_smiles_train.txt` and `generated_polymer_smiles_dev.txt` files.

2. Train the tokenizer `poetry run python train_tokenizer.py`. Make sure the path to the data set of 100 million PSMILES is correct. 

3. Tokenize the data set `poetry run python do_tokenize.py`.

4. Train polyBERT `poetry run python train_transformer.py`. This can be very slow if you do not have access to GPUs.
