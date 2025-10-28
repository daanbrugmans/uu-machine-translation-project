import model
import data
import runtime

if __name__ == "__main__":
    runtime.set_universal_seed(31313131)
    device = runtime.get_cuda_device()
    
    nmt_model = model.NMTModel()
    target_lang = "zho"
    sentence_pairs = data.get_creole_dataset(f"djk-{target_lang}")
    print(nmt_model.tokenizer.lang_code_to_id)

    bleu_score, bleu_details, bleu_data = nmt_model.get_bleu(sentence_pairs["test"], target_lang)
    print(f"BLEU: {bleu_score}")
    
    chrf_score, chrf_details, chrf_data = nmt_model.get_chrf(sentence_pairs["test"], target_lang)
    print(f"ChrF: {chrf_score}")
    
    sources, targets, predictions = chrf_data
    for source, target, prediction in zip(sources, targets, predictions):
        print(source)
        print(target)
        print(prediction)
        print()
