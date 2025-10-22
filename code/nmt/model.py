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
            target_lang_code = "hr_HR"
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
            decoder_start_token_id=self.tokenizer._convert_token_to_id_with_added_voc(
                target_lang_code
            ),
        )

        translated_sentence = self.tokenizer.decode(
            model_output[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return translated_sentence

    def translate_batch(
        self, dataset: datasets.Dataset
    ) -> tuple[list[str], list[str], list[str]]:
        sources = []
        targets = []
        predictions = []

        for row in dataset["translation"]:
            sources.append(row["src_text"])
            targets.append(row["tgt_text"])
            predictions.append(self.translate(row["src_text"], row["tgt_text"]))

        return sources, targets, predictions

    # TODO: Implement preprocessing
    def get_bleu(self, dataset: datasets.Dataset) -> tuple[float, dict]:
        _, targets, predictions = self.translate_batch(dataset)
        nested_targets = map(lambda x: [x], targets)

        bleu = evaluate.load("bleu")
        bleu_results = bleu.compute(references=nested_targets, predictions=predictions)

        return bleu_results["bleu"], bleu_results

    # TODO: Implement preprocessing
    def get_chrf(self, dataset: datasets.Dataset) -> tuple[float, dict]:
        _, targets, predictions = self.translate_batch(dataset)
        nested_targets = map(lambda x: [x], targets)

        chrf = evaluate.load("chrf")
        chrf_results = chrf.compute(references=nested_targets, predictions=predictions)

        return chrf_results["score"], chrf_results
