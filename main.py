import os
import logging
from io import BytesIO
import pandas as pd
from pathlib import Path
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
from datetime import datetime, timedelta
from collections import defaultdict, deque
import re
from modules import (
    LLMProcessor, QueryType,
    ChartGenerator
)
from modules.llm_processor import ChartType, QueryParameters
from modules.data_query import DataFrameManager
from modules.classifier import Classifier

# ---------- Config ----------
load_dotenv()
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

LOG_FILE = Path(os.getenv("BOT_LOG_FILE", "conversazioni_bot.csv"))

HELP_TEXT = (
    "📊 **Come usare il bot**\n\n"
    "Scrivimi una domanda sui dati socio-economici comunali.\n\n"
    "✨ **Esempi utili**:\n"
    "- 👥 Popolazione di *Milano* e *Roma*\n"
    "- 👩‍🦳 Pensionati a *Napoli* e *Milano* rispetto alla popolazione (2015-2023)\n"
    "- 💼 Reddito medio a *Torino* nel tempo\n"
    "- 📈 Gini index a *Bari* e *Palermo*\n"
    "- 🎓 Laureati residenti a *Firenze* e *Bologna* (ultimo anno disponibile)\n\n"
    "⚙️ **Comandi rapidi**:\n"
    "• /start → messaggio di benvenuto\n"
    "• /help → questa guida\n"
    "• /info → informazioni sul bot\n"
    "• /plot → genera un grafico a partire da una query\n"
)

INFO_TEXT = (
    "ℹ️ **Informazioni sul Bot**\n\n"
    "👨‍💻 Autore: *Giulio Albano* (Univ. Bari, tirocinio Banca d'Italia)\n"
    "📚 Fonti: *ISTAT, MEF, MIUR, Infocamere, Eurostat*\n\n"
    "🔎 Il bot normalizza i dati, calcola metriche derivate "
    "(pro capite, percentuali, quote, crescite) e genera grafici leggibili. "
    "Perfetto per confronti tra comuni, serie storiche e analisi distributive."
)

# Limiti e blocchi
MAX_REQUESTS_PER_MINUTE = 5
MAX_NONSENSE = 5
BLOCK_DURATION = timedelta(minutes=30)
user_requests = defaultdict(deque)
user_nonsense = defaultdict(int)
user_blocked = {}

# ---------- Funzioni utility ----------

def log_message(user, message_text, message_type, direction="IN", query_type=None, comuni=None, metrics=None):
    clean_text = message_text.encode("utf-8","ignore").decode("utf-8") if message_text else ""
    clean_text = clean_text.replace("\n"," ").replace("\r"," ").replace('"',"'")
    rec = {
        "timestamp": pd.Timestamp.now(),
        "user_id": user.id,
        "username": user.username or user.first_name,
        "first_name": user.first_name,
        "last_name": user.last_name or "",
        "direction": direction,
        "message_type": message_type,
        "message_text": clean_text,
        "query_type": query_type or "",
        "comuni": ",".join(comuni) if comuni else "",
        "metrics": ",".join(metrics) if metrics else "",
        "character_count": len(clean_text),
        "word_count": len(clean_text.split()) if clean_text else 0
    }
    df = pd.DataFrame([rec])
    try:
        if LOG_FILE.exists():
            df.to_csv(LOG_FILE, mode="a", header=False, index=False, encoding="utf-8-sig")
        else:
            df.to_csv(LOG_FILE, mode="w", header=True, index=False, encoding="utf-8-sig")
    except Exception as e:
        logger.error(f"log save error: {e}")

def is_user_blocked(user_id: int) -> bool:
    if user_id in user_blocked:
        if datetime.now() < user_blocked[user_id]:
            return True
        else:
            del user_blocked[user_id]
            user_nonsense[user_id] = 0
    return False

def register_request(user_id: int) -> bool:
    now = datetime.now()
    dq = user_requests[user_id]
    while dq and (now - dq[0]).seconds > 60:
        dq.popleft()
    dq.append(now)
    if len(dq) > MAX_REQUESTS_PER_MINUTE:
        user_blocked[user_id] = now + BLOCK_DURATION
        return False
    return True

def is_nonsense_message(message: str) -> bool:
    s = (message or "").strip().lower()
    if len(s) < 2: return True
    if re.match(r"^[^\w\s]+$", s) or re.match(r"^\d+$", s): return True
    if re.match(r"^(.)\1{4,}$", s): return True
    if re.match(r"^[a-z]{10,}$", s) and not any(v in s for v in "aeiou"): return True
    return False

# ---------- Classe principale ----------

class SocioEconomicBot:
    def __init__(self, token: str):
        self.token = token
        self.application = None
        self.df_manager = DataFrameManager(data_dir=os.getenv("DATA_DIR", r"resources"))
        self.chart_generator = ChartGenerator()
        openai_key = os.getenv("OPENAI_API_KEY")
        self.classifier = Classifier(openai_key) if openai_key else None
        self.llm_processor = LLMProcessor(openai_key) if openai_key else None
        self.main_keyboard = ReplyKeyboardMarkup(
            [[KeyboardButton("/start"), KeyboardButton("/help")],
             [KeyboardButton("/info"), KeyboardButton("/plot")]],
            resize_keyboard=True, is_persistent=True
        )

    # ---------- Setup ----------
    def setup(self):
        self.application = Application.builder().token(self.token).build()
        try:
            self.df_manager.load_data()
        except Exception as e:
            logger.error(f"Errore caricamento dati: {e}")
        self._register_handlers()

    def _register_handlers(self):
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("info", self.info_command))
        self.application.add_handler(CommandHandler("plot", self.plot_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_question))

    # ---------- Comandi ----------
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        msg = (
            f"Ciao {user.first_name} 👋\n\n"
            "Sono il bot per esplorare i dati socio-economici comunali 🇮🇹\n\n"
            "Scrivimi una richiesta, ad esempio:\n"
            "• `Popolazione Bari e Napoli nel tempo`\n"
            "• `Reddito medio a Milano 2010-2020`\n"
            "• `Quota pensionati Roma e Firenze`\n\n"
            "Per altri esempi digita /help 😉"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN, reply_markup=self.main_keyboard)
        log_message(user, msg, "RESPONSE", "OUT", query_type="START")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        msg = HELP_TEXT
        try:
            if self.llm_processor:
                vars = self.llm_processor.available_variables(limit=20)
                if vars:
                    msg += "\n\n🔑 **Variabili disponibili (esempi)**:\n" + ", ".join(f"`{v}`" for v in vars)
        except Exception:
            pass
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN, reply_markup=self.main_keyboard)
        log_message(user, msg, "RESPONSE", "OUT", query_type="HELP")

    async def info_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        msg = INFO_TEXT
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN, reply_markup=self.main_keyboard)
        log_message(user, msg, "RESPONSE", "OUT", query_type="INFO")

    # ---------- Gestione domande ----------
    async def handle_question(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        text = update.message.text.strip()

        try:
            if self.classifier:
                res = self.classifier.classify(text)
                cat = res.get('category')
                if cat == 'help_request':
                    await self.help_command(update, context)
                    return
                if cat == 'info_request':
                    await self.info_command(update, context)
                    return
                if cat == 'offensive':
                    await update.message.reply_text("🚫 Per favore usa un linguaggio rispettoso.")
                    return
                if cat == 'nonsense':
                    await update.message.reply_text("⚠️ Messaggio non riconosciuto. Usa /help per esempi.")
                    return
        except Exception:
            pass

        if is_user_blocked(user.id):
            await update.message.reply_text("🚫 Sei temporaneamente bloccato. Riprova tra 30 minuti.")
            return
        if not register_request(user.id):
            await update.message.reply_text("🚫 Troppe richieste. Riprova tra 30 minuti.")
            return
        if is_nonsense_message(text):
            user_nonsense[user.id] += 1
            if user_nonsense[user.id] >= MAX_NONSENSE:
                user_blocked[user.id] = datetime.now() + BLOCK_DURATION
                await update.message.reply_text("🚫 Messaggi non validi. Riprova tra 30 minuti.")
            else:
                await update.message.reply_text("⚠️ Messaggio non riconosciuto. Usa /help per esempi.")
            return

        await self.plot_command(update, context, user_input=text)

    # ---------- Comando /plot ----------
    async def plot_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_input: str = None):
        user = update.effective_user
        text = user_input or " ".join(context.args)
        if not text:
            await update.message.reply_text("❓ Usa `/plot popolazione Milano` o simili.", parse_mode=ParseMode.MARKDOWN)
            return

        processing_msg = await update.message.reply_text("🔄 Sto elaborando la tua richiesta...")

        try:
            if self.llm_processor is None:
                await processing_msg.edit_text("❌ LLM non configurato. Aggiungi OPENAI_API_KEY nel .env")
                return

            params = self.llm_processor.process_request(text)
            logger.info(
                f"Richiesta parsata: query={getattr(params.query_type,'value',params.query_type)}, "
                f"comuni={params.comuni}, metrics={params.metrics}, anno={params.anno}, "
                f"range={params.start_year}-{params.end_year}"
            )

            df, xlabel, ylabel = self.df_manager.query_data(params)

            if not getattr(params, 'comuni', None):
                inferred = self._infer_comuni_from_text(text)
                if inferred:
                    params.comuni = inferred
                    logger.info(f"Comuni inferiti dal testo: {inferred}")
                    df, xlabel, ylabel = self.df_manager.query_data(params)

            tl = text.lower()
            if ('line' in tl or 'linea' in tl or 'andamento' in tl or 'nel tempo' in tl or 'storico' in tl or 'evoluzione' in tl):
                params.chart_type = ChartType.LINE

            if df is None or df.empty:
                await processing_msg.edit_text("❌ Nessun dato trovato.")
                return

            metrics_label = ' / '.join(params.metrics or [])
            comuni_label = ", ".join(params.comuni) if params.comuni else "(tutti i comuni)"

            if params.anno:
                period_label = str(params.anno)
            elif params.start_year and params.end_year:
                period_label = f"{params.start_year}-{params.end_year}"
            elif "anno" in df.columns and df["anno"].notna().any():
                ymin = int(pd.to_numeric(df["anno"], errors="coerce").dropna().min())
                ymax = int(pd.to_numeric(df["anno"], errors="coerce").dropna().max())
                period_label = f"{ymin}-{ymax}" if ymin != ymax else str(ymin)
            else:
                period_label = ""

            title_parts = [metrics_label, comuni_label, period_label]
            title = " - ".join([p for p in title_parts if p])

            img = self.chart_generator.generate_chart(
                df, chart_type=(getattr(params.chart_type, 'value', params.chart_type)),
                title=title, xlabel=xlabel, ylabel=ylabel
            )

            comment = ""
            try:
                if self.llm_processor:
                    comment = self.llm_processor.generate_commentary(df, params)
            except Exception:
                comment = ""

            await update.message.reply_photo(BytesIO(img), caption=title, reply_markup=self.main_keyboard)
            if comment:
                await update.message.reply_text(comment)

        except Exception as e:
            logger.exception("Errore nella generazione del grafico")
            await processing_msg.edit_text("❌ Errore durante l'elaborazione. Riprova.")
            log_message(user, str(e), "ERROR", "OUT")

    # ---------- Run ----------
    def run(self):
        if not self.application:
            self.setup()
        logger.info("Bot in esecuzione...")
        self.application.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)

# ---------- Avvio ----------
def main():
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    if not TOKEN:
        logger.error("Token mancante. Imposta TELEGRAM_BOT_TOKEN nel .env")
        return
    bot = SocioEconomicBot(TOKEN)
    try:
        bot.run()
    except KeyboardInterrupt:
        logger.info("Bot arrestato dall'utente")
    except Exception as e:
        logger.error(f"Errore critico: {e}")

if __name__ == "__main__":
    main()
