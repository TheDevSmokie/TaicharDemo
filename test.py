import faiss
import numpy as np
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
    def __init__(self, model_name="google/embeddinggemma-300m"):
        print("Loading Model...")
        self.model = SentenceTransformer(model_name)
        
        # We store words in separate lists to facilitate PoS-specific searching
        self.pools = {
            "glagol": [],       # Verbs
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
        
        # Build a FAISS index for each category
        for pos, words in self.pools.items():
            if not words: continue
            texts = [w.osnOblika for w in words]
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings.astype('float32'))
            self.indices[pos] = index
            
        print(f"Database built. Verbs: {len(self.pools['glagol'])}, "
              f"Nouns: {len(self.pools['samostalnik'])}, "
              f"Adjectives: {len(self.pools['pridevnik'])}")

    def _get_closest_unpracticed(self, target_vector, pos, count=1):
        """Finds the semantically closest unpracticed words of a specific PoS."""
        if pos not in self.indices or not self.pools[pos]:
            return []

        # Search for more than needed because many might be 'practiced'
        _, indices = self.indices[pos].search(target_vector.astype('float32'), k=len(self.pools[pos]))
        
        found = []
        for idx in indices[0]:
            word = self.pools[pos][idx]
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
            
            print(f"PROMPT: {prompt_str}")
            # Show hints (gender/plural) to help the user form the sentence correctly
            input("(Press any key to show hints)")
            hints = [f"{b.osnOblika} ({b.spol if b.spol else b.besednaVrsta})" for b in current_group]
            print(f"HINTS:  {', '.join(hints)}")
            
            input("USER (Type sentence): ")
            print("-" * 30)

        print("\n=== Session Complete! All verbs exhausted. ===")

# --- 3. Execution with Sample Data ---
if __name__ == "__main__":

    # Sample Dictionary
    vocab_data = [
        # Verbs (glagol)
        Beseda("laufen", besednaVrsta="glagol", prevod="to run"),
        Beseda("essen", besednaVrsta="glagol", prevod="to eat"),
        Beseda("schlafen", besednaVrsta="glagol", prevod="to sleep"),
        
        # Nouns (samostalnik)
        Beseda("der Hund", besednaVrsta="samostalnik", spol="m", mnozina="Hunde", prevod="dog"),
        Beseda("der Apfel", besednaVrsta="samostalnik", spol="m", mnozina="Äpfel", prevod="apple"),
        Beseda("das Bett", besednaVrsta="samostalnik", spol="n", mnozina="Betten", prevod="bed"),
        Beseda("der Park", besednaVrsta="samostalnik", spol="m", mnozina="Parks", prevod="park"),
        Beseda("die Pizza", besednaVrsta="samostalnik", spol="f", mnozina="Pizzen", prevod="pizza"),
        
        # Adjectives (pridevnik)
        Beseda("schnell", besednaVrsta="pridevnik", prevod="fast"),
        Beseda("lecker", besednaVrsta="pridevnik", prevod="delicious"),
        Beseda("müde", besednaVrsta="pridevnik", prevod="tired"),
    ]

    app = GrammarAwarePracticeApp()
    app.build_database(vocab_data)
    app.run_session()