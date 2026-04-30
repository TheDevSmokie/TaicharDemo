import Tajnica

def main():
    print("Hello from taichar!")
    tajnica = Tajnica.Tajnica()
    tajnica.preberiBesede()    
    for beseda in tajnica.vseBesede:
        beseda.izpis()

if __name__ == "__main__":
    main()
