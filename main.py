import sys
import os

# Ensure the root project directory is in the path so python imports 'src' package correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
from PySide6 import QtWidgets, QtGui
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("copilot_debug.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("main")

# Load environment configs
load_dotenv()

def main() -> None:
    """Application entry point. Initializes PySide6 application, creates main window and starts loop.

    Complexity:
        Time Complexity: O(1) for startup setup, then event loop executes.
        Space Complexity: O(1) memory for startup setup.
    """
    logger.info("Starting Radiology Report Copilot...")
    
    try:
        # Create PySide Application
        app = QtWidgets.QApplication(sys.argv)
        app.setStyle("Fusion")
        
        # Set app-wide icon if available, or set standard styling
        app.setApplicationName("Radiology Report Copilot")
        app.setApplicationVersion("1.0.0")

        # Import window locally to ensure all modules are loaded under app context
        from src.gui.main_window import MainWindow
        
        window = MainWindow()
        window.show()
        
        logger.info("Main GUI window displayed successfully. Entering event loop.")
        sys.exit(app.exec())
        
    except Exception as e:
        # Section 2.1: Resilience Pipeline
        # 1. Catching: Generic catch for fatal launch errors
        # 2. Logging: Record fatal details to files
        logger.critical("Fatal crash during application startup: %s", str(e), exc_info=True)
        
        # 3. Graceful degradation: Display a clean error window to user instead of silent crash
        try:
            # Try utilizing PySide message box if window system was initialized
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Erro Crítico de Inicialização")
            msg.setInformativeText(f"Não foi possível iniciar o Radiology Report Copilot:\n{str(e)}")
            msg.setWindowTitle("Erro Fatal")
            msg.exec()
        except Exception:
            # fallback to ctypes MessageBox if Qt initialization failed completely
            try:
                import ctypes
                ctypes.windll.user32.MessageBoxW(
                    0, 
                    f"Erro de Inicialização:\n{str(e)}\n\nConsulte o arquivo 'copilot_debug.log' para detalhes.", 
                    "Erro Crítico - Radiology Copilot", 
                    0x10 | 0x0
                )
            except Exception as ctypes_err:
                logger.error("Could not display native dialog box: %s", str(ctypes_err))
        
        sys.exit(1)

if __name__ == "__main__":
    main()
