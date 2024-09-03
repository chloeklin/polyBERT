# polyBERT: A chemical language model to enable fully machine-driven ultrafast polymer informatics

# The code and data shared here is available for academic non-commercial use only

This repository contains the code and data set of the polyBERT paper, which can be accessed on [Nature Communications](https://doi.org/10.1038/s41467-023-39868-6) and [arXiv](https://arxiv.org/abs/2209.14803). polyBERT is a large chemical language machine learning model that has learned the PSMILES chemical language of polymers. It generates polymer fingerprints (PSMILES-based) quickly, facilitating rapid polymer informatics and machine-driven property predictions. polyBERT is compatible with GPUs and CPUs, and can easily scale on cloud systems.

**Abstract:** Polymers are a vital part of everyday life. Their chemical universe is so large that it presents unprecedented opportunities as well as significant challenges to identify suitable application-specific candidates. We present a complete end-to-end machine-driven polymer informatics pipeline that can search this space for suitable candidates at unprecedented speed and accuracy. This pipeline includes a polymer chemical fingerprinting capability called polyBERT (inspired by Natural Language Processing concepts), and a multitask learning approach that maps the polyBERT fingerprints to a host of properties. polyBERT is a chemical linguist that treats the chemical structure of polymers as a chemical language. The present approach outstrips the best presently available concepts for polymer property prediction based on handcrafted fingerprint schemes in speed by two orders of magnitude while preserving accuracy, thus making it a strong candidate for deployment in scalable architectures including cloud infrastructures.

# Use polyBERT

The trained polyBERT model is available at the [Hugginface hub](https://huggingface.co/kuelumbus/polyBERT) for download.

If want to generate polymer fingerprints from PSMILES strings with polyBERT, I strongly recommend using the `psmiles` Python package. It takes care of downloading polyBERT, canonicalization of PSMILES strings, and computing polyBERT fingerprints. Please see the [PSMILES](https://github.com/Ramprasad-Group/psmiles) on GitHub.

# polyOne data set

The data set contains 100 million hypothetical polymers each with 29 predicted properties using polyBERT. It is openly available at [Zenodo](https://zenodo.org/record/7766806).

The data set is sharded into smaller chunks (each has about 115MB) for better and faster processing. If you want to download all files, I recommend using `zenodo-get`.

```py
pip install zenodo-get
mkdir polyOne && cd polyOne
zenodo_get 7124188
```

This will download all files to the current directory. I recommend using dask (`pip install dask`) to load and process the data set. For example, compute the data set description:

```py
import dask.dataframe as dd
ddf = dd.read_parquet("*.parquet", engine="pyarrow")

df_describe = ddf.describe().compute()
print(df_describe)
```

# Train polyBERT from scratch

In case you want to train polyBERT from scratch.

1. Make sure [Poetry](https://python-poetry.org/docs/) is installed

2. Clone the code and install the Python environment
```bash
git clone git@github.com:Ramprasad-Group/polyBERT.git
cd polyBERT

poetry config virtualenvs.in-project true
poetry install
```

3. Download the 100 million PSMILES strings from [Zenodo](https://zenodo.org/record/7766806) and place it in the `polyBERT` directory. You need the `generated_polymer_smiles_train.txt` and `generated_polymer_smiles_dev.txt` files.

4. Train the tokenizer `poetry run python train_tokenizer.py`. Make sure the path to the downloaded data files is correct. 

5. Tokenize the data set `poetry run python do_tokenize.py`.

6. Train polyBERT `poetry run python train_transformer.py`. This can be very slow if you do not have access to GPUs.
