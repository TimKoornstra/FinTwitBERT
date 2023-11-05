from transformers import BertForMaskedLM, AutoTokenizer


class Evaluate:
    def __init__(self):
        self.model = BertForMaskedLM.from_pretrained("StephanAkkerman/FinTwitBERT")
        self.tokenizer = AutoTokenizer.from_pretrained("StephanAkkerman/FinTwitBERT")
