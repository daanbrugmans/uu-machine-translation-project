import model
import data
import runtime

if __name__ == "__main__":
    runtime.set_universal_seed(3131)
    device = runtime.get_cuda_device()

    nmt_model = model.NMTModel()
    djk_deu_dataset = data.get_creole_dataset("djk-deu")

    print(nmt_model.get_bleu(djk_deu_dataset["test"]))
    print(nmt_model.get_chrf(djk_deu_dataset["test"]))
