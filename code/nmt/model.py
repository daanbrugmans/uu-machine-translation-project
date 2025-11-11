from abc import ABC

import transformers
import datasets
import evaluate
import peft
from tqdm import tqdm


class NMTModel(ABC):
    def __init__(self, use_custom: bool):
        super().__init__()
        
        self.use_custom = use_custom

        self.tokenizer = transformers.MBartTokenizer.from_pretrained(
            # "jhu-clsp/kreyol-mt", use_fase=False
            "jhu-clsp/kreyol-mt-pubtrain", use_fase=False
        )
        
        base_model = self.mbart = transformers.MBartForConditionalGeneration.from_pretrained(
            # "jhu-clsp/kreyol-mt"
            "jhu-clsp/kreyol-mt-pubtrain"
        )
        
        if use_custom:
            self.mbart = peft.PeftModel.from_pretrained(base_model, "/vol/tensusers/dbrugmans/projects/uu-machine-translation-project/models/finetuned_creole_mbart_20")
        else:
            self.mbart = base_model

        self.pad_id = self.tokenizer(
            "<pad>", add_special_tokens=False, return_tensors="pt", padding=True
        ).input_ids.item()

        self.bos_id = self.tokenizer(
            "<s>", add_special_tokens=False, return_tensors="pt", padding=True
        ).input_ids.item()

        self.eos_id = self.tokenizer(
            "</s>", add_special_tokens=False, return_tensors="pt", padding=True
        ).input_ids.item()

    def translate(self, source_sentence: str, source_lang: str, target_lang: str) -> str:
        self.mbart.eval()
        
        source_lang_code = self._get_lang_code(source_lang)
        target_lang_code = self._get_lang_code(target_lang)

        tokenizer_input = source_sentence + f" </s> {source_lang_code}"
        model_input = self.tokenizer(
            tokenizer_input, add_special_tokens=False, return_tensors="pt", padding=True
        ).input_ids

        model_output = self.mbart.generate(
            inputs=model_input,
            use_cache=True,
            num_beams=4,
            max_length=60,
            min_length=1,
            early_stopping=True,
            pad_token_id=self.pad_id,
            bos_token_id=self.bos_id,
            eos_token_id=self.eos_id,
            decoder_start_token_id=self.tokenizer.lang_code_to_id[target_lang_code],
            # decoder_start_token_id=250025,
        )

        translated_sentence = self.tokenizer.decode(
            model_output[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        return translated_sentence
    
    def _get_lang_code(self, language: str) -> str:
        if language == "deu":
            return "de_DE"
        elif language == "djk":
            return "gu_IN"
        elif language == "eng":
            return "en_XX"
        else:
            raise ValueError(
                f"Specified language '{language}' could not be recognized."
            )

    def translate_batch(
        self, dataset: datasets.Dataset, source_lang: str, target_lang: str
    ) -> tuple[list[str], list[str], list[str]]:
        sources = []
        targets = []
        predictions = []
        
        for row in tqdm(dataset["translation"]):
            sources.append(row["src_text"])
            targets.append(row["tgt_text"])
            predictions.append(self.translate(row["src_text"], source_lang, target_lang))

        return sources, targets, predictions

    def get_bleu(self, dataset: datasets.Dataset, source_lang: str, target_lang: str) -> tuple[float, dict, tuple[list[str], list[str], list[str]]]:
        sources, targets, predictions = self.translate_batch(dataset, source_lang, target_lang)
        
        preprocessed_targets = remove_removable_chars(targets)
        preprocessed_predictions = remove_removable_chars(predictions)
        
        preprocessed_targets = list(map(lambda x: x.lower(), preprocessed_targets))
        preprocessed_targets = list(map(lambda x: [x], preprocessed_targets))
        
        preprocessed_predictions = list(map(lambda x: x.lower(), preprocessed_predictions))

        bleu = evaluate.load("bleu")
        bleu_results = bleu.compute(references=preprocessed_targets, predictions=preprocessed_predictions)

        return bleu_results["bleu"], bleu_results, (sources, targets, predictions)

    def get_chrf(self, dataset: datasets.Dataset, source_lang: str, target_lang: str) -> tuple[float, dict, tuple[list[str], list[str], list[str]]]:
        sources, targets, predictions = self.translate_batch(dataset, source_lang, target_lang)
        
        preprocessed_targets = remove_removable_chars(targets)
        preprocessed_predictions = remove_removable_chars(predictions)
        
        preprocessed_targets = list(map(lambda x: x.lower(), preprocessed_targets))
        preprocessed_targets = list(map(lambda x: [x], preprocessed_targets))
        
        preprocessed_predictions = list(map(lambda x: x.lower(), preprocessed_predictions))

        chrf = evaluate.load("chrf")
        chrf_results = chrf.compute(references=preprocessed_targets, predictions=preprocessed_predictions)

        return chrf_results["score"], chrf_results, (sources, targets, predictions)
    
def remove_removable_chars(corpus: list[str]) -> list[str]:
        preprocessed_corpus: list[str] = []
        removable_chars = '….,?!:;‘“”()[]{}+=-*_–\|/<>#€%^&*@1234567890"\n│•·†„‘·`‚¬½'

        for text in corpus:
            text = "".join([char for char in text if char not in removable_chars])
            text = text.replace("’", "'")
            text = text.replace("''", "'")
            text = text.strip()

            preprocessed_corpus.append(text)

        return preprocessed_corpus

