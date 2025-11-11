import datasets


def get_creole_dataset(language_pair: str) -> datasets.DatasetDict:
    return datasets.load_dataset("jhu-clsp/kreyol-mt", language_pair)

def get_german_backtranslations() -> datasets.Dataset:
    return datasets.load_from_disk("/vol/tensusers/dbrugmans/projects/uu-machine-translation-project/data/backtranslations/djk-eng-deu")
