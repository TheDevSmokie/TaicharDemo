class Beseda():
    def __init__(self, osnOblika, mnozina=None, besednaVrsta = None, tipVeznika = None, spol = None, jeModalni = None, praeteritum = None,
                 partizipPerfekt = None, prevod = None):
        self.osnOblika = osnOblika
        self.mnozina = mnozina
        self.besednaVrsta = besednaVrsta
        self.tipVeznika = tipVeznika
        self.spol = spol
        self.jeModalni = jeModalni
        self.praeteritum =  praeteritum
        self.partizipPerfekt = partizipPerfekt
        self.prevod = prevod
    
    def izpis(self):
        print(f"{self.osnOblika}\n\t{self.mnozina}\n\t{self.praeteritum}")

##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