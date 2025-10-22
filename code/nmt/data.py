import datasets


def get_creole_dataset(language_pair: str) -> datasets.DatasetDict:
    return datasets.load_dataset("jhu-clsp/kreyol-mt", language_pair)
