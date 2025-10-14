import transformers

if __name__ == "__main__":
    nmt_tokenizer = transformers.MBartTokenizer.from_pretrained("jhu-clsp/kreyol-mt-pubtrain", use_fase=False)
    
    source_text = "Ne wanten wanten a sikiman kon bun baka. Ne a teke en kama, ne a go. Ma namo, da ala a sani be e pasa a wan kina dei fu den Dyu."
    tokenizer_input = source_text + " </s> gu_IN"
    model_input = nmt_tokenizer(tokenizer_input, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
    
    pad_id = nmt_tokenizer("<pad>", add_special_tokens=False, return_tensors="pt", padding=True).input_ids.item()
    bos_id = nmt_tokenizer("<s>", add_special_tokens=False, return_tensors="pt", padding=True).input_ids.item()
    eos_id = nmt_tokenizer("</s>", add_special_tokens=False, return_tensors="pt", padding=True).input_ids.item()
    
    nmt_model = transformers.MBartForConditionalGeneration.from_pretrained("jhu-clsp/kreyol-mt-pubtrain")
    nmt_model.eval()
    
    model_output = nmt_model.generate(
        model_input,
        use_cache=True,
        num_beams=4,
        max_length=60,
        min_length=1,
        early_stopping=True,
        pad_token_id=pad_id, 
        bos_token_id=bos_id, 
        eos_token_id=eos_id, 
        decoder_start_token_id=nmt_tokenizer._convert_token_to_id_with_added_voc("de_DE")
    )
    predicted_text = nmt_tokenizer.decode(model_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    print(predicted_text)
    