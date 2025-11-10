# 🏆[SIGSPATIAL 25 Best Short Paper Award] TyphoFormer
Language-Augmented Transformer for Accurate Typhoon (Hurricane) Track Forecasting

## 🧭 1.Project Overview
> TyphoFormer is a hybrid multi-modal Transformer designed for tropical cyclone (other names: Hurricane, Typhoon) track prediction. It integrates `numerical meteorological features` and `LLM-augmented language embeddings` through a Prompt-aware Gating Fusion (PGF) module, followed by a spatio-temporal Transformer backbone and autoregressive decoding for track forecasting.


## 🧱 2.Repository Structure
```
TyphoFormer/
├── model/
│   ├── STTransformer.py       # Spatio-Temporal backbone
│   ├── PGF_module.py          # Prompt-aware Gating Fusion module
│   ├── TyphoFormer.py         # TyphoFormer model architecture
│
│
├── data/                      # Processed Typhoon datasets in '.npy' files
│   ├── train/
│   ├── val/
│   └── test/
│
├── embedding_chunks/          # LLM generated semantic descriptions are embeded by sentence-transformer
│   ├── emb_chunk_000.npy
│   ├── ......
│   ├── emb_chunk_006.npy ...
│
├── HURDAT_2new_3000.csv       # Raw typhoon dataset, includes 4 years' typhoon data here as an example
├── generate_text_description_new.py   # GPT-based language generation
├── generate_text_embeddings.py        # Embedding generation via MiniLM
├── prepare_typhoformer_data.py        # Dataset preparation script
├── train_typhoformer.py               # Training entry point
├── eval_typhoformer.py                # Evaluation script
└── README.md
```


## 🫶 How to Cite:
> If you find our work useful, please kindly cite our paper, thank you for your appreciation!

```
@inproceedings{lityphoformer2025,
author = {Li, Lincan and Ozguven, Eren Erman and Zhao, Yue and Wang, Guang and Xie, Yiqun and Dong, Yushun},
title = {TyphoFormer: Language-Augmented Transformer for Accurate Typhoon Track Forecasting},
location = {Minnesota, MN, USA},
series = {SIGSPATIAL '25},
year = {2025}
}
```
