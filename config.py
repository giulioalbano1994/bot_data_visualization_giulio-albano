"""
Modulo di configurazione centralizzata per il bot.
Gestisce tutte le impostazioni e parametri del sistema.
"""

import os
from enum import Enum
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Carica variabili d'ambiente
load_dotenv()

class Environment(Enum):
    """Tipi di ambiente di esecuzione"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class QuestionType(Enum):
    """Tipi di domande che il bot può gestire"""
    POPULATION = "population"          # Domande sulla popolazione
    ECONOMY = "economy"                # Dati economici
    EMPLOYMENT = "employment"          # Occupazione/disoccupazione
    DEMOGRAPHICS = "demographics"      # Dati demografici
    COMPARISON = "comparison"          # Confronto tra comuni
    STATISTICS = "statistics"          # Statistiche generali
    UNKNOWN = "unknown"               # Non classificabile

@dataclass
class BotConfig:
    """Configurazione principale del bot"""
    # Telegram
    telegram_token: str
    telegram_max_message_length: int = 4096
    
    # OpenAI
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    openai_max_tokens: int = 500
    openai_temperature: float = 0.7
    
    # Database
    database_url: Optional[str] = None
    database_pool_size: int = 5
    database_max_overflow: int = 10
    
    # Redis Cache
    redis_url: Optional[str] = None
    cache_ttl: int = 3600  # 1 ora in secondi
    
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    
    # Limiti e timeout
    request_timeout: int = 30  # secondi
    max_retries: int = 3
    rate_limit_per_user: int = 60  # richieste per minuto
    
    # Grafici
    chart_width: int = 800
    chart_height: int = 600
    chart_dpi: int = 100
    chart_style: str = "seaborn"
    
    @classmethod
    def from_env(cls) -> 'BotConfig':
        """Crea configurazione dalle variabili d'ambiente"""
        env_str = os.getenv('ENVIRONMENT', 'development').lower()
        
        # Mappa stringa a enum
        env_map = {
            'development': Environment.DEVELOPMENT,
            'staging': Environment.STAGING,
            'production': Environment.PRODUCTION
        }
        environment = env_map.get(env_str, Environment.DEVELOPMENT)
        
        return cls(
            telegram_token=os.getenv('TELEGRAM_BOT_TOKEN', ''),
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            database_url=os.getenv('DATABASE_URL'),
            redis_url=os.getenv('REDIS_URL'),
            environment=environment,
            debug=environment == Environment.DEVELOPMENT
        )
    
    def validate(self) -> bool:
        """Valida che la configurazione sia completa"""
        if not self.telegram_token:
            raise ValueError("TELEGRAM_BOT_TOKEN è richiesto!")
        
        if self.environment == Environment.PRODUCTION:
            if not self.database_url:
                raise ValueError("DATABASE_URL è richiesto in produzione!")
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY è richiesto in produzione!")
        
        return True

class Messages:
    """Messaggi predefiniti del bot"""
    
    # Messaggi di benvenuto
    WELCOME = (
        "Benvenuto! 👋\n\n"
        "Sono il bot per informazioni socio-economiche comunali.\n"
        "Posso fornirti dati, statistiche e grafici sui comuni italiani."
    )
    
    # Messaggi di errore
    ERROR_GENERIC = "❌ Si è verificato un errore. Riprova più tardi."
    ERROR_NOT_FOUND = "❌ Non ho trovato dati per questa richiesta."
    ERROR_INVALID_COMUNE = "❌ Il comune specificato non è valido."
    ERROR_RATE_LIMIT = "⚠️ Hai fatto troppe richieste. Attendi un minuto."
    
    # Messaggi di elaborazione
    PROCESSING = "🔄 Sto elaborando la tua richiesta..."
    GENERATING_CHART = "📊 Sto generando il grafico..."
    FETCHING_DATA = "📂 Sto recuperando i dati..."
    
    # Messaggi informativi
    NO_DATA_AVAILABLE = "📭 Non ci sono dati disponibili per questo parametro."
    DATA_SOURCE = "📌 Fonte dati: ISTAT - Ultimo aggiornamento: {date}"

class Patterns:
    """Pattern regex per il riconoscimento delle domande"""
    
    # Pattern per identificare comuni
    COMUNE_PATTERN = r'\b(?:comune di |città di )?([A-Za-zÀ-ÿ\s\'-]+)\b'
    
    # Pattern per tipi di dati
    POPULATION_KEYWORDS = ['popolazione', 'abitanti', 'residenti']
    ECONOMY_KEYWORDS = ['economia', 'pil', 'reddito', 'economico']
    EMPLOYMENT_KEYWORDS = ['lavoro', 'occupazione', 'disoccupazione', 'occupati']
    
    # Pattern per confronti
    COMPARISON_PATTERN = r'(?:confronta|paragona|differenza tra|versus)'

# Istanza globale della configurazione
config = BotConfig.from_env()