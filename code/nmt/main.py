import os
os.environ["HF_HOME"] = "/vol/tensusers/dbrugmans/projects/ru-master-thesis-24-25/models"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import datasets
import transformers
import peft

import model
import data
import runtime

def create_custom_dataset(dataset_for_backtranslations: datasets.DatasetDict):
    sources = []
    translations = []
    backtranslations = []
    for split in ["train", "test", "validation"]:
        split_sources, split_translations, split_backtranslations = nmt_model.translate_batch(dataset_for_backtranslations[split], "eng", "deu")
        
        sources.extend(split_sources)
        translations.extend(split_translations)
        backtranslations.extend(split_backtranslations)
        
    data_points = []
    for source, translation, backtranslation in zip(sources, translations, backtranslations):
        data_points.append({
            "src_lang": "djk",
            "src_text": source,
            "tgt_lang": "deu",
            "tgt_text": backtranslation,
            "int_lang": "eng",
            "int_text": translation
        })

    backtranslation_dataset = datasets.Dataset.from_dict({
        "translation": data_points,
        "source": sources,
        "intermediate": translations,
        "backtranslation": backtranslations
    })
    backtranslation_dataset.save_to_disk("/vol/tensusers/dbrugmans/projects/uu-machine-translation-project/data/backtranslations/djk-eng-deu")
    
    
def train(creole_model: model.NMTModel, train: datasets.Dataset, validation: datasets.Dataset) -> None:
    def prepare_dataset_for_training(dataset: datasets.Dataset) -> datasets.Dataset:        
        source_sentences = [row["src_text"] for row in dataset["translation"]]
        target_sentences = [row["tgt_text"] for row in dataset["translation"]]
        
        tokenized_sources = creole_model.tokenizer(
            source_sentences,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        tokenized_targets = creole_model.tokenizer(
            target_sentences,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).input_ids

        tokenized_sources["labels"] = tokenized_targets
        
        return tokenized_sources

    lora_config = peft.LoraConfig(
        r=8,
        lora_alpha=16, # Usually twice the rank `r`
        lora_dropout=0.1,
        bias="none",
        task_type=peft.TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"] 
    )
    
    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=None,
        eval_strategy="epoch",
        num_train_epochs=20,
        learning_rate=1e-3,
        weight_decay=1e-5,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        save_total_limit=1,
        fp16=True,
        seed=31313131,
    )
    
    lora_mbart = peft.get_peft_model(creole_model.mbart, lora_config)
    tokenized_train = train.map(prepare_dataset_for_training, batched=True)
    tokenized_validation = validation.map(prepare_dataset_for_training, batched=True)
    
    trainer = transformers.Seq2SeqTrainer(
        lora_mbart,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        tokenizer=creole_model.tokenizer,
    )
    
    lora_mbart.train()
    trainer.train()
    lora_mbart.save_pretrained("/vol/tensusers/dbrugmans/projects/uu-machine-translation-project/models/finetuned_creole_mbart_20/")
    
    
if __name__ == "__main__":
    runtime.set_universal_seed(31313131)
    device = runtime.get_cuda_device()
        
    # nmt_model = model.NMTModel(False)
    # print(nmt_model.mbart.__class__)
    sentence_pairs = data.get_creole_dataset("djk-deu")
    # sentence_pairs = data.get_german_backtranslations()
    # sentence_pairs.to_parquet("/vol/tensusers/dbrugmans/projects/uu-machine-translation-project/data/backtranslations/djk-eng-deu/kreyol-mt-backtranslations.parquet")
    print(sentence_pairs)
    # [print(row) for row in sentence_pairs]
    # print(nmt_model.tokenizer.lang_code_to_id)
    
    # create_custom_dataset(sentence_pairs)
    
    # bleu_score, bleu_details, bleu_data = nmt_model.get_chrf(sentence_pairs["test"], "djk", "deu")
    
    # train(nmt_model, sentence_pairs, data.get_creole_dataset("djk-deu")["validation"])
    # custom_nmt_model = model.NMTModel(True)
    # print(custom_nmt_model.mbart.__class__)
    # bleu_score, bleu_details, bleu_data = custom_nmt_model.get_bleu(sentence_pairs["test"], "djk", "deu")
    
    # sources, targets, predictions = bleu_data
    # for source, target, prediction in zip(sources, targets, predictions):
    #     print(source)
    #     print(target)
    #     print(prediction)
    #     print()
    # print(f"BLEU: {bleu_score}")
    # print(custom_nmt_model.use_custom)
