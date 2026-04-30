import Beseda
import json

class Tajnica():
    vocabFilePath = "vocab/vocabulary.json"
    vseBesede = []
    def __init__(self, vocabFilePath = "vocab/vocabulary.json"):
        self.vocabFilePath = vocabFilePath
        pass

    def preberiBesede(self):
        with open(self.vocabFilePath, 'r') as file:
            besede = json.load(file)
            for beseda in besede:
                self.vseBesede.append(Beseda.Beseda(**beseda))

