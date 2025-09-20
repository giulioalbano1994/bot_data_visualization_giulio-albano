#!/usr/bin/env python3
"""
Script di test per verificare i moduli senza Telegram.
Utile per debug e sviluppo.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Aggiungi la directory parent al path
sys.path.insert(0, str(Path(__file__).parent))

from modules.data_query import DataFrameManager
from modules.llm_processor import LLMProcessor, QueryType, ChartType, QueryParameters
from modules.chart_generator import ChartGenerator

# Carica variabili d'ambiente
load_dotenv()

def test_data_loading():
    """Test caricamento dati"""
    print("=" * 50)
    print("TEST: Caricamento Dati")
    print("=" * 50)
    
    df_manager = DataFrameManager(data_path="data/")
    
    # Cerca file dati
    data_files = list(Path("data/").glob("*.csv")) + list(Path("data/").glob("*.xlsx"))
    
    if not data_files:
        print("❌ Nessun file dati trovato in data/")
        print("Crea la cartella 'data/' e inserisci il tuo CSV o XLSX")
        return None
    
    # Carica il primo file trovato
    print(f"📂 Caricamento {data_files[0].name}...")
    df_manager.load_data(data_files[0].name)
    
    print(f"✅ Caricati {len(df_manager.df)} record")
    print(f"📊 Colonne: {len(df_manager.df.columns)}")
    print(f"📅 Anni disponibili: {df_manager.get_available_years()}")
    
    return df_manager

def test_queries(df_manager):
    """Test query sui dati"""
    print("\n" + "=" * 50)
    print("TEST: Query sui Dati")
    print("=" * 50)
    
    # Test 1: Dati singolo comune
    print("\n1. Dati di Roma:")
    roma_data = df_manager.get_comune_data("ROMA", columns=['comune', 'popolazione', 'reddito_imponibile_ammontare_in_euro'])
    if not roma_data.empty:
        print(roma_data.head())
    else:
        print("Nessun dato trovato per Roma")
    
    # Test 2: Top comuni
    print("\n2. Top 5 comuni per popolazione:")
    top_comuni = df_manager.get_top_comuni('popolazione', n=5)
    if not top_comuni.empty:
        print(top_comuni)
    
    # Test 3: Confronto comuni
    print("\n3. Confronto Roma vs Milano:")
    confronto = df_manager.compare_comuni(
        ['ROMA', 'MILANO'],
        ['popolazione', 'reddito_imponibile_ammontare_in_euro']
    )
    if not confronto.empty:
        print(confronto)
    
    return True

def test_llm_processor():
    """Test processore LLM"""
    print("\n" + "=" * 50)
    print("TEST: Processore LLM")
    print("=" * 50)
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️ OPENAI_API_KEY non configurata - skip test LLM")
        return None
    
    processor = LLMProcessor(api_key)
    
    # Test alcune richieste
    test_requests = [
        "mostra il reddito medio di Milano",
        "confronta Roma e Napoli per popolazione",
        "top 10 comuni per indice gini",
        "andamento popolazione Torino negli ultimi 5 anni"
    ]
    
    for request in test_requests:
        print(f"\n📝 Richiesta: '{request}'")
        try:
            params = processor.process_request(request)
            print(f"✅ Parametri estratti:")
            print(f"   - Query Type: {params.query_type.value}")
            print(f"   - Chart Type: {params.chart_type.value}")
            print(f"   - Comuni: {params.comuni}")
            print(f"   - Metriche: {params.metrics}")
        except Exception as e:
            print(f"❌ Errore: {e}")
    
    return processor

def test_chart_generation(df_manager):
    """Test generazione grafici"""
    print("\n" + "=" * 50)
    print("TEST: Generazione Grafici")
    print("=" * 50)
    
    generator = ChartGenerator()
    
    # Prepara dati di test
    top_comuni = df_manager.get_top_comuni('popolazione', n=5)
    
    if top_comuni.empty:
        print("❌ Nessun dato per generare grafici")
        return
    
    try:
        # Genera un grafico
        print("📊 Generazione grafico top 5 comuni...")
        chart_bytes = generator.generate_chart(
            data=top_comuni,
            chart_type='bar',
            title='Top 5 Comuni per Popolazione',
            xlabel='Comune',
            ylabel='Popolazione'
        )
        
        # Salva il grafico
        output_path = Path("test_chart.png")
        with open(output_path, 'wb') as f:
            f.write(chart_bytes)
        
        print(f"✅ Grafico salvato in: {output_path}")
        
    except Exception as e:
        print(f"❌ Errore nella generazione: {e}")

def main():
    """Funzione principale di test"""
    print("\n🚀 AVVIO TEST MODULI BOT")
    print("=" * 70)
    
    # Test 1: Caricamento dati
    df_manager = test_data_loading()
    if not df_manager:
        return
    
    # Test 2: Query
    test_queries(df_manager)
    
    # Test 3: LLM
    test_llm_processor()
    
    # Test 4: Grafici
    test_chart_generation(df_manager)
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETATI")
    print("\nSe tutti i test sono passati, il bot è pronto!")
    print("Avvia con: python main.py")

if __name__ == "__main__":
    main()