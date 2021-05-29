## Hypergraph Aggregation Neural Network(HANN)

This is the code of the Hypergraph Aggregation Neural Network(HANN) in [our paper: A Novel Joint Biomedical Event Extraction Framework via Two-level Modeling of Documents](https://www.sciencedirect.com/science/article/pii/S0020025520310409).

### Requirement
- python 3
- pytorch == 1.5.1
- torchtext== 0.6.0
- torch-geometric==1.6.1

To install the requirements, run `pip -r requirements.txt`.

### How to run the code?
After preprocessing the [MLEE](http://nactem.ac.uk/MLEE/) dataset and put it under `mlee`, the main entrance is in `train.py`.

### Cite
Please cite our paper:
```bibtex
@article{ZHAO2021,
    title = {A novel joint biomedical event extraction framework via two-level modeling of documents},
    journal = {Information Sciences},
    volume = {550},
    pages = {27-40},
    year = {2021},
    issn = {0020-0255},
    doi = {https://doi.org/10.1016/j.ins.2020.10.047},
    url = {https://www.sciencedirect.com/science/article/pii/S0020025520310409}
}
```