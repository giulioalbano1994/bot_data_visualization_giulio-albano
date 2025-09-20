# Socio-Economic Bot

This repository contains a **Telegram bot for socio-economic data analysis**, developed in Python.  
It allows users to query Italian municipal data in **natural language** and automatically receive  
statistics, time series, and visualisations.

## ✨ Features
- **Natural Language Queries**: users can ask questions like  
  - “Popolazione di Bari e Napoli nel tempo”  
  - “Reddito medio a Milano 2010-2020”  
  - “Indice di Gini a Palermo e Bari”  
- **Data Processing**: automated cleaning, aggregation, and normalisation of socio-economic indicators (ISTAT, MEF, MIUR, Infocamere, Eurostat).  
- **Visualization**: automatic chart generation (line, bar, distribution) with `matplotlib`.  
- **Automation**: includes a **data-analysis bot** for reproducible workflows and a **Telegram interface** for interactive exploration.  
- **Integration with LLMs**: query classification, variable mapping, and automatic commentary generation.  
- **Robustness**: request throttling, nonsense filtering, and error handling.  

## 🔧 Technologies
- Python, pandas, matplotlib, aiohttp, re  
- Telegram Bot API (`python-telegram-bot`)  
- Custom modules: `DataFrameManager`, `LLMProcessor`, `ChartGenerator`, `Classifier`  
- Optional integration with OpenAI LLM API  
