# Project Plan

- We're training our own statistical model (Ndyuka -> German & Ndyuka -> Chinese)
    - use data cleaning (https://github.com/GILT-Forum/TM-Mgmt-Best-Practices)
- Will not be training our own neural model (to start)

### Backtranslation

- Use _German --> Ndyuka_ & _Chinese -> Ndyuka_ direction to generate additional synthetic training data
    - Which German / Chinese data do we translate to Ndyuka?
- Enhance SMT & pretrained mBART model with the synthetic data (https://aclanthology.org/2024.naacl-long.170.pdf)

### Transfer Learning

- initialize model with parameters based on English or Portuguese, since Nduyka is supposedly most similar to these 2 languages
