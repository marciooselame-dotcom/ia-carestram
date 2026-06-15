import difflib
import logging
import sqlite3
import re
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class UserProfile:
    """Manages the radiologist's writing style, custom templates, and vocabulary preferences.
    
    Uses SQLite database to persist preferences, and difflib to compare LLM outputs
    with final physician modifications, updating style guidelines dynamically.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        """Initializes the UserProfile database tables and loads default templates.

        Args:
            db_path (Optional[str]): Path to the SQLite database.
        """
        self.db_path = db_path or "copilot_data.db"
        self.conn = sqlite3.connect(self.db_path)
        self._init_tables()
        self._load_default_templates()

    def _init_tables(self) -> None:
        """Creates the necessary tables for preferences and style learning if they don't exist."""
        try:
            cursor = self.conn.cursor()
            # Table to store default and custom templates per joint
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS templates (
                    joint_type TEXT PRIMARY KEY,
                    template_text TEXT
                )
            """)
            # Table to store learned word vocabulary adjustments (e.g. from -> to)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learned_vocabulary (
                    original_word TEXT PRIMARY KEY,
                    preferred_word TEXT,
                    frequency INTEGER DEFAULT 1
                )
            """)
            # Table to store custom style rules (e.g., "Sempre colocar conclusões em lista")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS style_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_text TEXT UNIQUE,
                    active INTEGER DEFAULT 1
                )
            """)
            self.conn.commit()
            logger.info("UserProfile database initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize UserProfile tables: %s", str(e), exc_info=True)

    def _load_default_templates(self) -> None:
        """Populates default templates if they are missing."""
        default_templates: Dict[str, str] = {
            "joelho": (
                "RM DE JOELHO\n\n"
                "MENISCOS:\n"
                "- Menisco medial: espessura, morfologia e sinal normais.\n"
                "- Menisco lateral: espessura, morfologia e sinal normais.\n\n"
                "LIGAMENTOS:\n"
                "- Ligamento cruzado anterior (LCA): integridade e sinal preservados.\n"
                "- Ligamento cruzado posterior (LCP): integridade e sinal preservados.\n"
                "- Colaterais (LCL e LCM): sem alterações significativas.\n\n"
                "CARTILAGENS:\n"
                "- Superfícies articulares femorotibial e patelofemoral com espessura e sinal normais.\n\n"
                "MECANISMO EXTENSOR:\n"
                "- Tendão quadricipital e patelar sem alterações.\n\n"
                "ESTRUTURAS ÓSSEAS:\n"
                "- Alinhamento patelofemoral preservado.\n"
                "- Sinal da medula óssea normal.\n\n"
                "PARTES MOLES:\n"
                "- Ausência de derrame articular ou cistos significativos.\n\n"
                "CONCLUSÃO:\n"
                "1. Exame dentro dos limites normais."
            ),
            "ombro": (
                "RM DE OMBRO\n\n"
                "TENDÕES (MANGUITO ROTADOR):\n"
                "- Supraespinhal: espessura, morfologia e sinal normais.\n"
                "- Infraespinhal: espessura, morfologia e sinal normais.\n"
                "- Subescapular: espessura, morfologia e sinal normais.\n"
                "- Redondo menor: espessura, morfologia e sinal normais.\n\n"
                "INTERVALO DOS ROTADORES E CABO LONGO DO BÍCEPS (CLB):\n"
                "- CLB posicionado no sulco intertubercular com espessura e sinal normais.\n\n"
                "ARTICULAÇÃO ACROMIOCLAVICULAR:\n"
                "- Espaço articular preservado, sem osteófitos.\n\n"
                "ESPAÇO SUBACROMIAL E ACROMIO:\n"
                "- Acúmulo adiposo subacromial preservado, acrômio tipo I de Bigliani.\n\n"
                "ESTRUTURAS ÓSSEAS E CARTILAGENS:\n"
                "- Cabeça umeral e cavidade glenoide normais.\n\n"
                "PARTES MOLES:\n"
                "- Ausência de líquido na bursa subacromiosubdeltoidea.\n\n"
                "CONCLUSÃO:\n"
                "1. Exame dentro dos limites normais."
            ),
            "quadril": (
                "RM DE QUADRIL\n\n"
                "ARTICULAÇÃO COXOFEMORAL:\n"
                "- Cabeça femoral com esfericidade preservada.\n"
                "- Labrum acetabular íntegro.\n"
                "- Cartilagem de revestimento com espessura normal.\n\n"
                "ESTRUTURAS ÓSSEAS:\n"
                "- Sinal da medula óssea normal.\n\n"
                "TENDÕES E MÚSCULOS:\n"
                "- Tendões glúteos (médio e mínimo) e iliopsoas normais.\n\n"
                "PARTES MOLES:\n"
                "- Ausência de derrame articular significativo ou bursites.\n\n"
                "CONCLUSÃO:\n"
                "1. Exame dentro dos limites normais."
            ),
            "tornozelo": (
                "RM DE TORNOZELO\n\n"
                "LIGAMENTOS:\n"
                "- Complexo ligamentar lateral (talofibular anterior/posterior, calcaneofibular): íntegros.\n"
                "- Complexo ligamentar medial (deltoide): íntegro.\n\n"
                "TENDÕES:\n"
                "- Aquiles e fáscia plantar normais.\n"
                "- Tendões fibulares e tibiais (anterior/posterior) íntegros.\n\n"
                "ESTRUTURAS ÓSSEAS E ARTICULAÇÃO:\n"
                "- Alinhamento preservado, sinal da medula óssea normal.\n"
                "- Cartilagem de revestimento preservada.\n\n"
                "PARTES MOLES:\n"
                "- Ausência de líquido intra-articular livre.\n\n"
                "CONCLUSÃO:\n"
                "1. Exame dentro dos limites normais."
            ),
            "punho": (
                "RM DE PUNHO\n\n"
                "COMPARTIMENTOS E LIGAMENTOS:\n"
                "- Complexo da fibrocartilagem triangular (FCT) íntegro.\n"
                "- Ligamento escafolunar e semilunotriquetral íntegros.\n\n"
                "TENDÕES E NERVOS:\n"
                "- Canal do carpo de dimensões normais, nervo mediano com sinal preservado.\n"
                "- Tendões flexores e extensores íntegros.\n\n"
                "ESTRUTURAS ÓSSEAS:\n"
                "- Relações articulares preservadas, sem edema ósseo.\n\n"
                "CONCLUSÃO:\n"
                "1. Exame dentro dos limites normais."
            ),
            "cotovelo": (
                "RM DE COTOVELO\n\n"
                "ARTICULAÇÃO E CARTILAGENS:\n"
                "- Relações umeroulnar, umerorradial e radioulnar proximal preservadas.\n\n"
                "TENDÕES E LIGAMENTOS:\n"
                "- Tendão comum dos extensores e flexores íntegros.\n"
                "- Complexo ligamentar colateral lateral e medial íntegros.\n"
                "- Tendão do bíceps distal e braquial íntegros.\n\n"
                "ESTRUTURAS ÓSSEAS E NERVOS:\n"
                "- Nervo ulnar no sulco epitrocleolecraniano com sinal normal.\n"
                "- Sinal da medula óssea preservado.\n\n"
                "CONCLUSÃO:\n"
                "1. Exame dentro dos limites normais."
            ),
            "sacroilíacas": (
                "RESSONÂNCIA MAGNÉTICA DAS ARTICULAÇÕES SACROILÍACAS\n\n"
                "Superfícies articulares sacroilíacas regulares, sem evidência de edema ósseo subcondral "
                "ou focos de erosão óssea detectáveis pelo método.\n"
                "Não há focos de captação anômalos pelo meio de contraste no presente exame.\n"
                "Restante das estruturas ósseas avaliadas de aspecto habitual.\n"
                "Forames sacrais livres.\n"
                "Planos musculares sem alterações significativas.\n"
                "Feixes neurovasculares livres.\n"
                "Subcutâneo preservado.\n\n"
                "IMPRESSÃO:\n"
                "- Estudo por ressonância magnética sem alterações significativas."
            ),
            "coluna_lombar": (
                "RM DE COLUNA LOMBAR\n\n"
                "ALINHAMENTO:\n"
                "- Alinhamento vertebral preservado, sem listeses ou escoliose significativa.\n\n"
                "CORPOS VERTEBRAIS:\n"
                "- Altura e sinal da medula óssea normais.\n\n"
                "DISCO INTERVERTEBRAIS:\n"
                "- Espaços discais preservados, sem herniações ou protrusões significativas.\n\n"
                "CANAL VERTEBRAL E FORAMES:\n"
                "- Canal vertebral de dimensões normais.\n"
                "- Forames de conjugação amplos, sem estenose.\n\n"
                "CÔNUS MEDULAR E RAIZES:\n"
                "- Cônus medular em posição habitual, sem alterações de sinal.\n"
                "- Raízes da cauda equina sem alterações.\n\n"
                "PARTES MOLES:\n"
                "- Planos musculares paravertebrais sem alterações.\n\n"
                "CONCLUSÃO:\n"
                "1. Exame dentro dos limites normais."
            )
        }

        try:
            cursor = self.conn.cursor()
            for joint, text in default_templates.items():
                cursor.execute("INSERT OR IGNORE INTO templates (joint_type, template_text) VALUES (?, ?)", (joint, text))
            self.conn.commit()
        except Exception as e:
            logger.error("Failed to populate default templates: %s", str(e), exc_info=True)

    def get_template(self, joint_type: str) -> str:
        """Retrieves the template text for a specific joint type.

        Args:
            joint_type (str): Joint type name (e.g. joelho).

        Returns:
            str: Template text, or empty string if not found.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT template_text FROM templates WHERE joint_type = ?", (joint_type.lower(),))
            row = cursor.fetchone()
            if row:
                return str(row[0])
        except Exception as e:
            logger.error("Failed to fetch template for %s: %s", joint_type, str(e))
        return ""

    def update_template(self, joint_type: str, template_text: str) -> bool:
        """Updates or inserts a customized template for a specific joint.

        Args:
            joint_type (str): Joint type name.
            template_text (str): New template text.

        Returns:
            bool: True if updated successfully.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO templates (joint_type, template_text)
                VALUES (?, ?)
                ON CONFLICT(joint_type) DO UPDATE SET template_text = excluded.template_text
            """, (joint_type.lower(), template_text))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error("Failed to update template for %s: %s", joint_type, str(e), exc_info=True)
            self.conn.rollback()
            return False

    def learn_from_edit(self, ai_text: str, final_text: str) -> List[Tuple[str, str]]:
        """Compares AI output and final text, learning preferences and word choices.

        Args:
            ai_text (str): Output originally generated by the local LLM.
            final_text (str): Modified output edited and approved by the radiologist.

        Returns:
            List[Tuple[str, str]]: Word substitutions learned in this session.

        Complexity:
            Time Complexity: O(A * F) where A is the length of ai_text and F is final_text (difflib matching).
            Space Complexity: O(A + F) for matching lists.
        """
        # Clean words helper
        def clean_word(w: str) -> str:
            return re.sub(r"[^\wáéíóúâêîôûãõç-]", "", w.lower()).strip()

        try:
            ai_words = [clean_word(w) for w in ai_text.split() if clean_word(w)]
            final_words = [clean_word(w) for w in final_text.split() if clean_word(w)]
            
            matcher = difflib.SequenceMatcher(None, ai_words, final_words)
            learned: List[Tuple[str, str]] = []
            
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == "replace":
                    # Check if it looks like a clean word-for-word or small phrase substitution
                    ai_sub = ai_words[i1:i2]
                    final_sub = final_words[j1:j2]
                    
                    if len(ai_sub) == 1 and len(final_sub) == 1:
                        o_word = ai_sub[0]
                        p_word = final_sub[0]
                        # Don't learn single letter edits or typos
                        if len(o_word) > 3 and len(p_word) > 3 and o_word != p_word:
                            learned.append((o_word, p_word))

            # Persist learned vocabulary substitutions in SQLite
            if learned:
                cursor = self.conn.cursor()
                for o_word, p_word in learned:
                    cursor.execute("""
                        INSERT INTO learned_vocabulary (original_word, preferred_word, frequency)
                        VALUES (?, ?, 1)
                        ON CONFLICT(original_word) DO UPDATE SET 
                            preferred_word = excluded.preferred_word,
                            frequency = frequency + 1
                    """, (o_word, p_word))
                self.conn.commit()
                logger.info("Learned new vocabulary mappings: %s", learned)

            return learned

        except Exception as e:
            logger.error("Failed learning vocabulary from edits: %s", str(e), exc_info=True)
            self.conn.rollback()
            return []

    def get_style_preferences_prompt(self) -> str:
        """Generates a text summary of learned preferences to inject into LLM system prompts.

        Returns:
            str: Style guidelines to inject into system prompts.
        """
        preferences: List[str] = []
        try:
            cursor = self.conn.cursor()
            # Fetch vocabularies with frequency >= 2 (higher confidence)
            cursor.execute("SELECT original_word, preferred_word FROM learned_vocabulary WHERE frequency >= 2")
            vocab_rows = cursor.fetchall()
            if vocab_rows:
                vocab_rules = [f"Substituir '{orig}' por '{pref}'" for orig, pref in vocab_rows]
                preferences.append("- Vocabulário Preferencial:\n  " + "\n  ".join(vocab_rules))
                
            cursor.execute("SELECT rule_text FROM style_rules WHERE active = 1")
            style_rows = cursor.fetchall()
            if style_rows:
                rules = [f"- {row[0]}" for row in style_rows]
                preferences.append("- Regras de Estilo Adicionais:\n  " + "\n  ".join(rules))

        except Exception as e:
            logger.error("Failed to load style preferences prompt: %s", str(e))

        return "\n\n".join(preferences) if preferences else "Nenhuma preferência personalizada registrada ainda."

    def add_style_rule(self, rule_text: str) -> bool:
        """Manually registers a custom writing style rule.

        Args:
            rule_text (str): The rule description (e.g. 'Sempre numerar itens na Conclusão').

        Returns:
            bool: True if successful.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("INSERT OR IGNORE INTO style_rules (rule_text) VALUES (?)", (rule_text,))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error("Failed to add style rule: %s", str(e), exc_info=True)
            return False

    def close(self) -> None:
        """Closes SQLite database connection safely."""
        try:
            self.conn.close()
        except Exception:
            pass
