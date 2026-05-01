import faiss
#import numpy as np
import random
from sentence_transformers import SentenceTransformer

# --- 1. The Updated Data Structure ---
class Beseda():
    def __init__(self, osnOblika, mnozina=None, besednaVrsta=None, tipVeznika=None, 
                 spol=None, jeModalni=None, praeteritum=None, partizipPerfekt=None, 
                 prevod=None, practiced=False):
        self.osnOblika = osnOblika
        self.mnozina = mnozina
        self.besednaVrsta = besednaVrsta  # Expected: 'glagol', 'samostalnik', 'pridevnik'
        self.tipVeznika = tipVeznika
        self.spol = spol
        self.jeModalni = jeModalni
        self.praeteritum = praeteritum
        self.partizipPerfekt = partizipPerfekt
        self.prevod = prevod
        self.practiced = practiced

# --- 2. The Application Logic ---
class GrammarAwarePracticeApp:
    #sentence-transformers/all-MiniLM-L12-v2
    #google/embeddinggemma-300m
    def __init__(self, model_name="google/embeddinggemma-300m"):
        print("Loading Model...")
        self.model = SentenceTransformer(model_name)
        
        # We store words in separate lists to facilitate besedna_vrsta-specific searching
        self.pools = {
            "glagol": [],       # Verbs
            "modalni": [],
            "samostalnik": [],  # Nouns
            "pridevnik": [],    # Adjectives
            "veznik" : []
        }
        self.indices = {}

    def build_database(self, beseda_list):
        # Categorize words
        for b in beseda_list:
            if b.besednaVrsta in self.pools:
                self.pools[b.besednaVrsta].append(b)
                if b.jeModalni == True:
                    self.pools["modalni"].append(b)
        
        skupaj = []
        # Build a FAISS index for each category
        for besedna_vrsta, words in self.pools.items():
            if not words: continue
            
            texts = []
            for w in words:
                if w.besednaVrsta == "samostalnik" and w.spol in ["m", "f", "n"]:
                    if(w.spol == "m"):
                        prefix = "der"
                    elif (w.spol == "f"):
                        prefix = "die"
                    elif (w.spol == "n"):
                        prefix = "das"

                    texts.append(prefix + w.osnOblika)
                else:
                    texts.append(w.osnOblika)

            skupaj.append(texts)
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings.astype('float32'))
            self.indices[besedna_vrsta] = index
            
        print(f"Naloženih je {len(skupaj)} besed:\n{', \n}'.join([w for w in skupaj])}")

    def _get_closest_unpracticed(self, target_vector, besedna_vrsta, count=1):
        """Finds the semantically closest unpracticed words of a specific besedna_vrsta."""
        if besedna_vrsta not in self.indices or not self.pools[besedna_vrsta]:
            return []

        # Search for more than needed because many might be 'practiced'
        _, indices = self.indices[besedna_vrsta].search(target_vector.astype('float32'), k=len(self.pools[besedna_vrsta]))
        
        found = []
        for idx in indices[0]:
            word = self.pools[besedna_vrsta][idx]
            if not word.practiced:
                found.append(word)
            if len(found) >= count:
                break
        return found

    def run_session(self):
        # Reset session
        for category in self.pools.values():
            for b in category: b.practiced = False

        # We drive the session based on available verbs
        verbs = [v for v in self.pools['glagol']]
        random.shuffle(verbs)

        print("\n=== Advanced Sentence Practice ===")
        print("Rule: Use the Verb, Noun(s), and Adjective provided.\n")

        for verb in verbs:
            if verb.practiced: continue
            
            # Get the embedding for the verb to find related nouns/adjectives
            verb_vec = self.model.encode([verb.osnOblika], normalize_embeddings=True)
            
            # Find up to 2 related unpracticed nouns
            nouns = self._get_closest_unpracticed(verb_vec, "samostalnik", count=2)
            
            # Find up to 1 related unpracticed adjective
            adjs = self._get_closest_unpracticed(verb_vec, "pridevnik", count=1)

            # Mark them all as practiced
            verb.practiced = True
            for n in nouns: n.practiced = True
            for a in adjs: a.practiced = True

            # Format the prompt
            current_group = [verb] + nouns + adjs
            prompt_str = ", ".join([b.osnOblika for b in current_group])
            
            print(f"BESEDE: {prompt_str}")
            # Show hints (gender/plural) to help the user form the sentence correctly
            input("(pritisni karkoli, da se pokažejo namigi)")
            hints = [f"{b.osnOblika} ({b.spol if b.spol else b.besednaVrsta})" for b in current_group]
            print(f"NAMIGI:  {', '.join(hints)}")
            
            input("(naprej)")

        print("\n=== Vaja zaključena! Glagolov je zmanjkalo... ===")

# --- 3. Execution with Sample Data ---
if __name__ == "__main__":

    # Sample Dictionary
    vocab_data = [
        # Verbs (glagol)
        Beseda(osnOblika = "laufen", besednaVrsta="glagol", prevod="to run"),
        Beseda(osnOblika = "essen", besednaVrsta="glagol", prevod="to eat"),
        Beseda(osnOblika = "schlafen", besednaVrsta="glagol", prevod="to sleep"),
        Beseda(osnOblika = "können", besednaVrsta="glagol", jeModalni = True, prevod="to sleep"),
        
        # Nouns (samostalnik)
        Beseda(osnOblika = "Hund", besednaVrsta="samostalnik", spol="m", mnozina="Hunde", prevod="dog"),
        Beseda(osnOblika = "Apfel", besednaVrsta="samostalnik", spol="m", mnozina="Äpfel", prevod="apple"),
        Beseda(osnOblika = "Bett", besednaVrsta="samostalnik", spol="n", mnozina="Betten", prevod="bed"),
        Beseda(osnOblika = "Park", besednaVrsta="samostalnik", spol="m", mnozina="Parks", prevod="park"),
        Beseda(osnOblika = "Pizza", besednaVrsta="samostalnik", spol="f", mnozina="Pizzen", prevod="pizza"),
        
        # Adjectives (pridevnik)
        Beseda(osnOblika = "schnell", besednaVrsta="pridevnik", prevod="fast"),
        Beseda(osnOblika = "lecker", besednaVrsta="pridevnik", prevod="delicious"),
        Beseda(osnOblika = "müde", besednaVrsta="pridevnik", prevod="tired"),
    ]

    app = GrammarAwarePracticeApp()
    app.build_database(vocab_data)
    app.run_session()