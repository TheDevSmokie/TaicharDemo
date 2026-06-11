import Beseda
import json

class Tajnica():
    vocab_file_path = "vocab/vocabulary.json"
    vse_besede = []
    def __init__(self, vocab_file_path = "vocab/vocabulary.json"):
        self.vocab_file_path = vocab_file_path
        pass

    def preberi_besede(self):
        with open(self.vocab_file_path, 'r', encoding='utf-8') as file:
            besede = json.load(file)
            for vnos in besede:
                self.vse_besede.append(Beseda.Beseda(**vnos))

