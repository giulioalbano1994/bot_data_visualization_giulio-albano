#!/usr/bin/env python3
"""
Script per generare dati di esempio per test.
Crea un CSV con dati fittizi di comuni italiani.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import random

def generate_sample_data(n_comuni: int = 50, years: list = None):
    """
    Genera un dataset di esempio con dati comunali.
    
    Args:
        n_comuni: Numero di comuni da generare
        years: Lista degli anni (default: ultimi 5 anni)
    """
    
    if years is None:
        years = list(range(2019, 2024))
    
    # Liste di comuni reali per esempio
    comuni_esempi = [
        ('ROMA', 'RM', 'LAZIO', 2761477, True),
        ('MILANO', 'MI', 'LOMBARDIA', 1406242, True),
        ('NAPOLI', 'NA', 'CAMPANIA', 909048, True),
        ('TORINO', 'TO', 'PIEMONTE', 848885, True),
        ('PALERMO', 'PA', 'SICILIA', 630167, True),
        ('GENOVA', 'GE', 'LIGURIA', 558745, True),
        ('BOLOGNA', 'BO', 'EMILIA-ROMAGNA', 394463, True),
        ('FIRENZE', 'FI', 'TOSCANA', 366927, True),
        ('BARI', 'BA', 'PUGLIA', 316015, True),
        ('CATANIA', 'CT', 'SICILIA', 298762, True),
        ('VENEZIA', 'VE', 'VENETO', 254661, True),
        ('VERONA', 'VR', 'VENETO', 257274, True),
        ('MESSINA', 'ME', 'SICILIA', 218786, True),
        ('PADOVA', 'PD', 'VENETO', 205727, True),
        ('TRIESTE', 'TS', 'FRIULI-VENEZIA GIULIA', 198417, True),
        ('BRESCIA', 'BS', 'LOMBARDIA', 196058, True),
        ('PARMA', 'PR', 'EMILIA-ROMAGNA', 195687, True),
        ('TARANTO', 'TA', 'PUGLIA', 189366, True),
        ('PRATO', 'PO', 'TOSCANA', 195089, True),
        ('MODENA', 'MO', 'EMILIA-ROMAGNA', 188011, True),
        ('REGGIO CALABRIA', 'RC', 'CALABRIA', 171885, True),
        ('REGGIO EMILIA', 'RE', 'EMILIA-ROMAGNA', 169087, True),
        ('PERUGIA', 'PG', 'UMBRIA', 164417, True),
        ('LIVORNO', 'LI', 'TOSCANA', 153773, True),
        ('RAVENNA', 'RA', 'EMILIA-ROMAGNA', 153740, True),
        ('CAGLIARI', 'CA', 'SARDEGNA', 148881, True),
        ('FOGGIA', 'FG', 'PUGLIA', 146416, True),
        ('RIMINI', 'RN', 'EMILIA-ROMAGNA', 148908, True),
        ('SALERNO', 'SA', 'CAMPANIA', 128068, True),
        ('FERRARA', 'FE', 'EMILIA-ROMAGNA', 128874, True),
        # Aggiungi comuni minori
        ('SIRMIONE', 'BS', 'LOMBARDIA', 8200, False),
        ('BELLAGIO', 'CO', 'LOMBARDIA', 3700, False),
        ('PORTOFINO', 'GE', 'LIGURIA', 400, False),
        ('POSITANO', 'SA', 'CAMPANIA', 3900, False),
        ('TAORMINA', 'ME', 'SICILIA', 10900, False),
        ('ALBEROBELLO', 'BA', 'PUGLIA', 10700, False),
        ('MATERA', 'MT', 'BASILICATA', 60400, False),
        ('AOSTA', 'AO', 'VALLE D\'AOSTA', 33900, False),
        ('TRENTO', 'TN', 'TRENTINO-ALTO ADIGE', 118000, False),
        ('BOLZANO', 'BZ', 'TRENTINO-ALTO ADIGE', 107000, False),
        ('URBINO', 'PU', 'MARCHE', 14500, False),
        ('ASCOLI PICENO', 'AP', 'MARCHE', 46600, False),
        ('LECCE', 'LE', 'PUGLIA', 95400, False),
        ('COMO', 'CO', 'LOMBARDIA', 84000, False),
        ('VARESE', 'VA', 'LOMBARDIA', 80500, False),
        ('MONZA', 'MB', 'LOMBARDIA', 123000, False),
        ('BERGAMO', 'BG', 'LOMBARDIA', 120000, False),
        ('PESCARA', 'PE', 'ABRUZZO', 119000, False),
        ('LATINA', 'LT', 'LAZIO', 126000, False),
        ('TERNI', 'TR', 'UMBRIA', 109000, False)
    ]
    
    # Limita al numero richiesto
    comuni_esempi = comuni_esempi[:min(n_comuni, len(comuni_esempi))]
    
    data = []
    
    for anno in years:
        for idx, (comune, prov, regione, pop_base, is_capoluogo) in enumerate(comuni_esempi):
            # Genera codice ISTAT fittizio
            istat_code = f"{idx+1:06d}"
            codice_catastale = f"{chr(65 + idx % 26)}{idx:03d}"
            
            # Variazione popolazione per anno
            popolazione = int(pop_base * (1 + random.uniform(-0.02, 0.02)))
            
            # Numero contribuenti (circa 60-70% della popolazione)
            numero_contribuenti = int(popolazione * random.uniform(0.6, 0.7))
            
            # Reddito basato sulla regione e dimensione città
            reddito_base = 25000
            if regione in ['LOMBARDIA', 'VENETO', 'EMILIA-ROMAGNA']:
                reddito_base *= 1.3
            elif regione in ['LAZIO', 'PIEMONTE']:
                reddito_base *= 1.2
            elif regione in ['CALABRIA', 'SICILIA', 'CAMPANIA']:
                reddito_base *= 0.8
            
            if is_capoluogo:
                reddito_base *= 1.15
            
            # Aggiungi variazione annuale
            reddito_imponibile = reddito_base * (1 + (anno - 2019) * 0.02) * random.uniform(0.9, 1.1)
            
            # Altri tipi di reddito
            reddito_lavoro_dip = reddito_imponibile * 0.65
            reddito_pensione = reddito_imponibile * 0.25
            reddito_autonomo = reddito_imponibile * 0.1
            
            # Indice Gini (disuguaglianza)
            gini_index = random.uniform(0.28, 0.45)
            if is_capoluogo:
                gini_index += 0.03
            
            # Dati imprese (per provincia)
            imprese_attive = int(popolazione / 100 * random.uniform(8, 12))
            imprese_registrate = int(imprese_attive * 1.1)
            
            # Dati demografici
            laureati_tot = int(popolazione * random.uniform(0.15, 0.35))
            if is_capoluogo:
                laureati_tot = int(popolazione * random.uniform(0.25, 0.45))
            
            saldo_migratorio = random.randint(-500, 1000)
            if is_capoluogo:
                saldo_migratorio = random.randint(0, 2000)
            
            # Distribuzione reddito per fasce
            pop_fasce = numero_contribuenti
            f1 = int(pop_fasce * 0.15)  # 0-10k
            f2 = int(pop_fasce * 0.12)  # 10-15k
            f3 = int(pop_fasce * 0.35)  # 15-26k
            f4 = int(pop_fasce * 0.25)  # 26-55k
            f5 = int(pop_fasce * 0.08)  # 55-75k
            f6 = int(pop_fasce * 0.04)  # 75-120k
            f7 = int(pop_fasce * 0.01)  # >120k
            
            record = {
                'istat_comune': istat_code,
                'anno': anno,
                'codice_catastale': codice_catastale,
                'comune': comune,
                'denominazione_comune_norm': comune,
                'sigla_provincia': prov,
                'sigla_provincia_norm': prov,
                'regione_mef': regione,
                'regione_norm': regione,
                'codice_istat_regione': idx % 20 + 1,
                'anno_di_imposta': anno,
                'is_capoluogo': 1 if is_capoluogo else 0,
                'numero_contribuenti': numero_contribuenti,
                
                # Redditi
                'reddito_imponibile_frequenza': numero_contribuenti,
                'reddito_imponibile_ammontare_in_euro': round(reddito_imponibile, 2),
                'reddito_da_lavoro_dipendente_frequenza': int(numero_contribuenti * 0.7),
                'reddito_da_lavoro_dipendente_ammontare_in_euro': round(reddito_lavoro_dip, 2),
                'reddito_da_pensione_frequenza': int(numero_contribuenti * 0.3),
                'reddito_da_pensione_ammontare_in_euro': round(reddito_pensione, 2),
                'reddito_da_lavoro_autonomo_comprensivo_dei_valori_nulli_frequenza': int(numero_contribuenti * 0.1),
                'reddito_da_lavoro_autonomo_comprensivo_dei_valori_nulli_ammontare_in_euro': round(reddito_autonomo, 2),
                
                # Fasce di reddito
                'reddito_complessivo_da_0_a_10000_euro_frequenza': f1,
                'reddito_complessivo_da_0_a_10000_euro_ammontare_in_euro': f1 * 7000,
                'reddito_complessivo_da_10000_a_15000_euro_frequenza': f2,
                'reddito_complessivo_da_10000_a_15000_euro_ammontare_in_euro': f2 * 12500,
                'reddito_complessivo_da_15000_a_26000_euro_frequenza': f3,
                'reddito_complessivo_da_15000_a_26000_euro_ammontare_in_euro': f3 * 20000,
                'reddito_complessivo_da_26000_a_55000_euro_frequenza': f4,
                'reddito_complessivo_da_26000_a_55000_euro_ammontare_in_euro': f4 * 40000,
                'reddito_complessivo_da_55000_a_75000_euro_frequenza': f5,
                'reddito_complessivo_da_55000_a_75000_euro_ammontare_in_euro': f5 * 65000,
                'reddito_complessivo_da_75000_a_120000_euro_frequenza': f6,
                'reddito_complessivo_da_75000_a_120000_euro_ammontare_in_euro': f6 * 95000,
                'reddito_complessivo_oltre_120000_euro_frequenza': f7,
                'reddito_complessivo_oltre_120000_euro_ammontare_in_euro': f7 * 150000,
                
                # Altri indicatori
                'popolazione': popolazione,
                'gini_index': round(gini_index, 3),
                'imprese_attive_prov': imprese_attive,
                'imprese_registrate_prov': imprese_registrate,
                'laureati_tot': laureati_tot,
                'saldo_migratorio_tot_com': saldo_migratorio,
            }
            
            data.append(record)
    
    # Crea DataFrame
    df = pd.DataFrame(data)
    
    # Aggiungi altre colonne vuote per compatibilità
    missing_cols = [
        'reddito_da_fabbricati_frequenza', 'reddito_da_fabbricati_ammontare_in_euro',
        'reddito_da_lavoro_dipendente_e_pensione_frequenza', 
        'reddito_da_lavoro_dipendente_e_pensione_ammontare_in_euro',
        'reddito_di_spettanza_dell_imprenditore_in_contabilita_ordinaria_comprensivo_dei_valori_nulli_frequenza',
        'reddito_di_spettanza_dell_imprenditore_in_contabilita_ordinaria_comprensivo_dei_valori_nulli_ammontare_in_euro',
        'reddito_di_spettanza_dell_imprenditore_in_contabilita_semplificata_comprensivo_dei_valori_nulli_frequenza',
        'reddito_di_spettanza_dell_imprenditore_in_contabilita_semplificata_comprensivo_dei_valori_nulli_ammontare_in_euro',
        'reddito_da_partecipazione_comprensivo_dei_valori_nulli_frequenza',
        'reddito_da_partecipazione_comprensivo_dei_valori_nulli_ammontare_in_euro',
        'imposta_netta_frequenza', 'imposta_netta_ammontare_in_euro',
        'reddito_imponibile_addizionale_irpef_frequenza', 'reddito_imponibile_addizionale_irpef_ammontare_in_euro',
        'addizionale_regionale_dovuta_frequenza', 'addizionale_regionale_dovuta_ammontare_in_euro',
        'addizionale_comunale_dovuta_frequenza', 'addizionale_comunale_dovuta_ammontare_in_euro',
        'reddito_complessivo_minore_o_uguale_a_zero_euro_frequenza',
        'reddito_complessivo_minore_o_uguale_a_zero_euro_ammontare_in_euro',
        'is_media', 'provincia_norm_mef', 'dato_provinc_registrate', 'dato_provinc_attive',
        'dato_provinc_iscrizioni', 'dato_provinc_cessazioni_non_d_ufficio', 'dato_provinc_saldo',
        'comune_norm', 'comune_label', 'comune_upper', 'capoluogo_flag', 'provincia_norm_istat',
        'provincia_label', 'regione_istat', 'pop_totale', 'iscritti_altrocom_com',
        'iscritti_estero_com', 'iscritti_altri_com', 'cancellati_altrocom_com',
        'cancellati_estero_com', 'cancellati_altri_com', 'saldo_migratorio_estero_com',
        'laureati_femmine', 'laureati_maschi', 'laureati_res_femmine',
        'laureati_res_maschi', 'laureati_res_tot', 'brevetti_num_prov', 'brevetti_pct_prov',
        'imprese_iscrizioni_prov', 'imprese_cessazioni_prov', 'imprese_saldo_prov',
        '_merge', 'provincia_norm_u', 'merci_scaricate_tonnellate', 'nuts3_code'
    ]
    
    for col in missing_cols:
        if col not in df.columns:
            df[col] = 0
    
    return df

def main():
    """Funzione principale"""
    print("🎲 Generazione dati di esempio...")
    
    # Crea cartella data se non esiste
    data_path = Path("data")
    data_path.mkdir(exist_ok=True)
    
    # Genera dati
    df = generate_sample_data(n_comuni=50, years=[2019, 2020, 2021, 2022, 2023])
    
    # Salva in CSV
    output_file = data_path / "sample_comuni_data.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✅ Generati {len(df)} record")
    print(f"📊 {df['comune'].nunique()} comuni unici")
    print(f"📅 Anni: {sorted(df['anno'].unique())}")
    print(f"💾 Salvato in: {output_file}")
    
    # Mostra statistiche
    print("\n📈 Statistiche Dataset:")
    print(f"- Popolazione media: {df['popolazione'].mean():,.0f}")
    print(f"- Reddito medio: €{df['reddito_imponibile_ammontare_in_euro'].mean():,.2f}")
    print(f"- Gini medio: {df['gini_index'].mean():.3f}")
    print(f"- Comuni capoluogo: {df[df['is_capoluogo']==1]['comune'].nunique()}")
    
    print("\n✨ Dati di esempio pronti per il test del bot!")
    print("Puoi ora eseguire: python test_modules.py")

if __name__ == "__main__":
    main()