import faiss
import random
from sentence_transformers import SentenceTransformer
from Tajnica import Tajnica

class GrammarAwarePracticeApp:
    def __init__(self, model_name="google/embeddinggemma-300m"):
        print("Loading Model...")
        self.model = SentenceTransformer(model_name)
        
        self.pools = {
            "glagol": [],
            "modalni": [],
            "samostalnik": [],
            "pridevnik": [],
            "veznik": [],
            "prislov": [],
            "besedna_zveza": []
        }
        self.indices = {}
        self.threshold = 0.6

    def build_database(self, beseda_list):
        for b in beseda_list:
            if b.besedna_vrsta in self.pools:
                self.pools[b.besedna_vrsta].append(b)
                if getattr(b, 'je_modalni', False) in [True, "True", "true"]:
                    self.pools["modalni"].append(b)
        
        for besedna_vrsta, words in self.pools.items():
            if not words: continue
            
            texts = []
            for w in words:
                if w.besedna_vrsta == "samostalnik" and w.spol in ["moski", "zenski", "srednji"]:
                    prefix = {"moski": "der ", "zenski": "die ", "srednji": "das "}.get(w.spol, "")
                    texts.append(prefix + w.osnovna_oblika)
                else:
                    texts.append(w.osnovna_oblika)

            embeddings = self.model.encode(texts, normalize_embeddings=True)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings.astype('float32'))
            self.indices[besedna_vrsta] = index

    def _get_relevant_unpracticed(self, target_vector, besedna_vrsta, threshold=0, max_count=1):
        """Finds unpracticed words meeting a similarity threshold."""
        if besedna_vrsta not in self.indices or not self.pools[besedna_vrsta]:
            return []

        # Search the entire pool for this category
        scores, indices = self.indices[besedna_vrsta].search(
            target_vector.astype('float32'), 
            k=len(self.pools[besedna_vrsta])
        )
        
        found = []
        for score, idx in zip(scores[0], indices[0]):
            word = self.pools[besedna_vrsta][idx]
            # FAISS IndexFlatIP with normalized vectors returns cosine similarity
            if score >= threshold and not word.practiced:
                found.append(word)
            
            if len(found) >= max_count:
                break
        return found

    def run_session(self, mode="sl_to_de"):
        for category in self.pools.values():
            for b in category: b.practiced = False

        anchors = self.pools['glagol'] + self.pools['besedna_zveza']
        random.shuffle(anchors)

        print(f"\n=== Advanced Sentence Practice (Način: {mode}) ===")
        print(f"Tvori smiselno poved z uporabo podanih besed (Prag podobnosti: {self.threshold}).\n")

        for anchor in anchors:
            if anchor.practiced: continue
            
            anchor_vec = self.model.encode([anchor.osnovna_oblika], normalize_embeddings=True)
            
            # Smart grouping based on threshold and limits
            # Limits: 1 anchor (verb/phrase), 2 nouns, 2 adjectives, 1 adverb
            current_group = [anchor]
            
            requirements = [
                ("samostalnik", 2),
                ("pridevnik", 2),
                ("prislov", 1)
            ]

            for kind, limit in requirements:
                matches = self._get_relevant_unpracticed(anchor_vec, kind, threshold=self.threshold, max_count=limit)
                current_group.extend(matches)

            # Mark all as practiced
            for b in current_group:
                b.practiced = True

            # Display logic
            if mode == "sl_to_de":
                prompt_str = ", ".join([b.prevod if b.prevod else b.osnovna_oblika for b in current_group])
                print(f"SLOVENSKO: {prompt_str}")
                input("(namigi...)")
                hints = [f"{b.osnovna_oblika}{f' ({b.spol})' if b.spol else ''}" for b in current_group]
                print(f"NEMŠKO:    {', '.join(hints)}")
            else:
                prompt_str = ", ".join([b.osnovna_oblika for b in current_group])
                print(f"BESEDE: {prompt_str}")
                input("(namigi...)")
                hints = [f"{b.osnovna_oblika} ({b.spol if b.spol else b.besedna_vrsta})" for b in current_group]
                print(f"NAMIGI:  {', '.join(hints)}")
            
            print("-" * 30)

        print("\n=== Vaja zaključena! ===")

if __name__ == "__main__":
    try:
        tajnica = Tajnica("vocab/vocabulary.json")
        tajnica.preberi_besede()
        besede_za_vajo = tajnica.vse_besede
    except FileNotFoundError:
        besede_za_vajo = [] 

    app = GrammarAwarePracticeApp()
    app.build_database(besede_za_vajo)
    app.run_session(mode="sl_to_de")