import os
import sys
import logging
from typing import Dict

# Ensure project root is in path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.rag_engine import RAGEngine
from src.core.user_profile import UserProfile
from src.core.anonymizer import Anonymizer

# Setup logging to console
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("template_importer")

def import_user_templates(template_dir: str) -> None:
    """Reads normal report templates from directory, indexes them in RAG, and updates joint templates.

    Args:
        template_dir (str): Directory containing physician's template files (PDFs).

    Complexity:
        Time Complexity: O(F * P) where F is the number of files and P is average page size.
        Space Complexity: O(T) to hold texts in memory.
    """
    logger.info("Starting template import from: %s", template_dir)

    if not os.path.exists(template_dir):
        logger.error("Directory not found: %s", template_dir)
        return

    # Instantiate engines
    rag = RAGEngine(db_path="copilot_data.db", faiss_path="copilot_index.index")
    profile = UserProfile(db_path="copilot_data.db")
    anonymizer = Anonymizer()

    # Supported joints map
    joints_map = {
        "joelho": "joelho",
        "ombro": "ombro",
        "quadril": "quadril",
        "tornozelo": "tornozelo",
        "punho": "punho",
        "cotovelo": "cotovelo"
    }

    files = [f for f in os.listdir(template_dir) if f.lower().endswith(".pdf")]
    logger.info("Found %d PDF template files to import.", len(files))

    imported_rag_count = 0
    updated_template_count = 0

    for file in files:
        full_path = os.path.join(template_dir, file)
        
        try:
            # 1. Index report in RAG database
            success = rag.index_report(full_path)
            if success:
                imported_rag_count += 1
            
            # 2. Update default templates if filename matches key joint
            name_lower = os.path.splitext(file)[0].lower()
            if name_lower in joints_map:
                joint = joints_map[name_lower]
                text = rag.extract_text(full_path)
                if text:
                    # Clean/anonymize and update user template
                    sanitized_text = anonymizer.anonymize_text(text)
                    profile.update_template(joint, sanitized_text)
                    logger.info("Updated default template for joint: %s", joint)
                    updated_template_count += 1

        except Exception as e:
            logger.error("Failed to import %s. Error: %s", file, str(e), exc_info=True)

    rag.close()
    profile.close()
    
    logger.info("Import finished. RAG indexed: %d, Templates updated: %d", imported_rag_count, updated_template_count)

if __name__ == "__main__":
    target_folder = r"C:\Users\marci\Desktop\IA\laudos normais"
    import_user_templates(target_folder)
