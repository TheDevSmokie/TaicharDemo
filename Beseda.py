class Beseda():
    def __init__(self, osnovna_oblika, mnozina = None, besedna_vrsta = None, spol = None, tip_veznika = None, 
                 je_modalni = None, praeteritum = None, partizip_perfekt = None, prevod = None, imel_tezave = None):
        self.osnovna_oblika = osnovna_oblika 
        self.mnozina = mnozina 
        self.besedna_vrsta = besedna_vrsta 
        self.spol = spol 
        self.tip_veznika = tip_veznika 
        self.je_modalni = je_modalni 
        self.praeteritum = praeteritum 
        self.partizip_perfekt = partizip_perfekt 
        self.prevod = prevod 
        self.imel_tezave = imel_tezave 
        self.practiced = None