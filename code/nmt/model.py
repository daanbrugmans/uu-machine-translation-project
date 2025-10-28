from abc import ABC

import transformers
import datasets
import evaluate


class NMTModel(ABC):
    def __init__(self):
        super().__init__()

        self.tokenizer = transformers.MBartTokenizer.from_pretrained(
            "jhu-clsp/kreyol-mt-pubtrain", use_fase=False
        )
        self.mbart = transformers.MBartForConditionalGeneration.from_pretrained(
            "jhu-clsp/kreyol-mt-pubtrain"
        )

        self.pad_id = self.tokenizer(
            "<pad>", add_special_tokens=False, return_tensors="pt", padding=True
        ).input_ids.item()

        self.bos_id = self.tokenizer(
            "<s>", add_special_tokens=False, return_tensors="pt", padding=True
        ).input_ids.item()

        self.eos_id = self.tokenizer(
            "</s>", add_special_tokens=False, return_tensors="pt", padding=True
        ).input_ids.item()

    def translate(self, source_sentence: str, target_lang: str) -> str:
        self.mbart.eval()

        tokenizer_input = source_sentence + " </s> gu_IN"
        model_input = self.tokenizer(
            tokenizer_input, add_special_tokens=False, return_tensors="pt", padding=True
        ).input_ids

        if target_lang == "deu":
            target_lang_code = "de_DE"
        elif target_lang == "zho":
            target_lang_code = "zh_CN"
        else:
            raise ValueError(
                "Specified target language could not be recognized. Please use 'deu' for German or 'zho' for Chinese."
            )

        model_output = self.mbart.generate(
            model_input,
            use_cache=True,
            num_beams=4,
            max_length=60,
            min_length=1,
            early_stopping=True,
            pad_token_id=self.pad_id,
            bos_token_id=self.bos_id,
            eos_token_id=self.eos_id,
            # decoder_start_token_id=self.tokenizer.lang_code_to_id[target_lang_code],
            decoder_start_token_id=250025,
        )

        translated_sentence = self.tokenizer.decode(
            model_output[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        print(translated_sentence)

        return translated_sentence

    def translate_batch(
        self, dataset: datasets.Dataset, target_lang: str
    ) -> tuple[list[str], list[str], list[str]]:
        sources = []
        targets = []
        predictions = []

        for row in dataset["translation"]:
            sources.append(row["src_text"])
            targets.append(row["tgt_text"])
            predictions.append(self.translate(row["src_text"], target_lang))

        return sources, targets, predictions

    def get_bleu(self, dataset: datasets.Dataset, target_lang: str) -> tuple[float, dict, tuple[list[str], list[str], list[str]]]:
        sources, targets, predictions = self.translate_batch(dataset, target_lang)
        
        preprocessed_targets = remove_removable_chars(targets)
        preprocessed_predictions = remove_removable_chars(predictions)
        
        preprocessed_targets = list(map(lambda x: x.lower(), preprocessed_targets))
        preprocessed_targets = list(map(lambda x: [x], preprocessed_targets))
        
        preprocessed_predictions = list(map(lambda x: x.lower(), preprocessed_predictions))

        bleu = evaluate.load("bleu")
        bleu_results = bleu.compute(references=preprocessed_targets, predictions=preprocessed_predictions)

        return bleu_results["bleu"], bleu_results, (sources, targets, predictions)

    def get_chrf(self, dataset: datasets.Dataset, target_lang: str) -> tuple[float, dict, tuple[list[str], list[str], list[str]]]:
        sources, targets, predictions = self.translate_batch(dataset, target_lang)
        
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

