# Low‐resource Approaches for Automatic Speech Recognition for Maltese

This repository contains the complete implementation, experimentation and evaluation pipeline for adapting wav2vec-BERT to the Maltese language and decoding using a KenLM language model.

## Repository Structure

* `training.py` – The finalised Python script used to train the final acoustic model.
* `Fine_Tune_W2V2_Bert_on_Maltese.ipynb` – Core development notebook covering the experimental training runs and the integration/evaluation of the KenLM language model.
* `KenLM_Maltese.ipynb` – Includes the data pipeline for preprocessing the `KorpusMalti` dataset and trains the KenLM n-gram language model.
* `qualitative_analysis.ipynb` – Notebook utilised to perform error analysis.
* `Inference_Test.ipynb` – An interactive notebook designed to manually test audio samples for demonstration purposes.

---
#### 1. Acoustic Model Training (`training.py`)
The environment required to run the final training script can be installed using the provided requirements file:

```bash
pip install -r requirements.txt

#### 2. Language Model Pipeline (`KenLM_Maltese.ipynb`)
The text processing, text normalisation and n-gram training pipeline utilities are handled via cell-level installations directly inside the notebook.

Please note that compiling and building the **KenLM** binaries requires a Linux-based architecture. If you are operating on a Windows machine, this entire notebook must be executed inside **Windows Subsystem for Linux (WSL)** to ensure successful compilation.