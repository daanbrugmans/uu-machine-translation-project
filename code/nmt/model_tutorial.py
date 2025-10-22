from transformers import MBartForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import MbartTokenizer, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/kreyol-mt-pubtrain", do_lower_case=False, use_fast=False, keep_accents=True)

# Or use tokenizer = MbartTokenizer.from_pretrained("jhu-clsp/kreyol-mt-pubtrain", use_fast=False)

model = AutoModelForSeq2SeqLM.from_pretrained("jhu-clsp/kreyol-mt-pubtrain")

# Or use model = MBartForConditionalGeneration.from_pretrained("jhu-clsp/kreyol-mt-pubtrain")

# First tokenize the input and outputs. The format below is how the model was trained so the input should be "Sentence </s> SRCCODE". Similarly, the output should be "TGTCODE Sentence </s>". 
# Example: For Saint Lucian Patois to English translation, we need to use language indicator tags: <2acf> and <2eng> where acf represents Saint Lucian Patois and eng represents English.
# For a mapping of the original language and language code (3 character) to mBART-50 compatible language tokens consider the following dictionary:
# dictmap = {'acf': 'ar_AR', 'ara': 'cs_CZ', 'aze': 'it_IT', 'bzj': 'hi_IN', 'cab': 'az_AZ', 'ceb': 'et_EE', 'crs': 'fi_FI', 'deu': 'de_DE', 'djk': 'gu_IN', 'eng': 'en_XX', 'fra': 'fr_XX', 'gcf': 'ja_XX', 'gul': 'kk_KZ', 'hat': 'ko_KR', 'icr': 'lt_LT', 'jam': 'lv_LV', 'kea': 'my_MM', 'kri': 'ne_NP', 'ktu': 'nl_XX', 'mart1259': 'ro_RO', 'mfe': 'ru_RU', 'nep': 'si_LK', 'pap': 'tr_TR', 'pcm': 'vi_VN', 'por': 'pt_XX', 'sag': 'af_ZA', 'spa': 'es_XX', 'srm': 'bn_IN', 'srn': 'fa_IR', 'tpi': 'he_IL', 'zho': 'hr_HR', 'wes': 'zh_CN', 'trf': 'id_ID', 'svc': 'ka_GE', 'rcf': 'km_KH', 'pre': 'mk_MK', 'pov': 'ml_IN', 'mue': 'mn_MN', 'lou': 'mr_IN', 'gyn': 'pl_PL', 'gpe': 'ps_AF', 'gcr': 'sv_SE', 'fpe': 'sw_KE', 'fng': 'ta_IN', 'fab': 'te_IN', 'dcr': 'th_TH', 'cri': 'tl_XX', 'bzk': 'uk_UA', 'brc': 'ur_PK', 'bah': 'xh_ZA', 'aoa': 'gl_ES'}
# Note: We mapped languages to their language tokens manually. For example, we used en_XX, fr_XX, es_XX for English, French and Spanish as in the original mBART-50 model. But then we repurposed other tokens for Creoles.

# As for what the language codes and their corresponding languages are, please refer to: https://github.com/JHU-CLSP/Kreyol-MT?tab=readme-ov-file#building-machine-translation-for-latin-american-caribbean-and-colonial-african-creole-languages

inp = tokenizer('Mi tingk se yu de tel mi lai. </s> lv_LV', add_special_tokens=False, return_tensors="pt", padding=True).input_ids

model.eval() # Set dropouts to zero

model_output=model.generate(inp, use_cache=True, num_beams=4, max_length=60, min_length=1, early_stopping=True, pad_token_id=pad_id, bos_token_id=bos_id, eos_token_id=eos_id, decoder_start_token_id=tokenizer._convert_token_to_id_with_added_voc("en_XX"))

decoded_output=tokenizer.decode(model_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

print(decoded_output)
