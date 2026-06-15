import os
import logging
from typing import Optional
from PySide6 import QtWidgets, QtCore, QtGui
from src.core.user_profile import UserProfile
from src.core.llm_client import LLMClient

logger = logging.getLogger(__name__)

class ConfigDialog(QtWidgets.QDialog):
    """Configuration dialog for the Radiology Report Copilot.
    
    Allows customizing Ollama settings, selecting models, specifying the RAG source folder,
    and modifying report templates for joints.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        """Initializes the ConfigDialog window and loads current settings."""
        super().__init__(parent)
        self.setWindowTitle("Configurações do Copilot")
        self.resize(550, 600)
        self.setModal(True)

        self.user_profile = UserProfile()
        self.llm_client = LLMClient()
        
        self._init_ui()
        self._load_settings()

    def _init_ui(self) -> None:
        """Sets up the visual layout and style sheets for the dialog (sleek dark mode)."""
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e24;
                color: #ffffff;
                font-family: 'Segoe UI', sans-serif;
            }
            QLabel {
                color: #cccccc;
                font-size: 13px;
                font-weight: 500;
            }
            QLineEdit, QComboBox, QTextEdit {
                background-color: #2a2a35;
                border: 1px solid #444455;
                border-radius: 6px;
                color: #ffffff;
                padding: 6px;
                font-size: 13px;
            }
            QLineEdit:focus, QComboBox:focus, QTextEdit:focus {
                border: 1px solid #6c5ce7;
            }
            QPushButton {
                background-color: #6c5ce7;
                border: none;
                border-radius: 6px;
                color: #ffffff;
                font-weight: bold;
                padding: 8px 16px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #8073e8;
            }
            QPushButton:pressed {
                background-color: #5b4dbf;
            }
            QPushButton#btnCancel {
                background-color: #3e3e4a;
            }
            QPushButton#btnCancel:hover {
                background-color: #4f4f60;
            }
            QListWidget {
                background-color: #2a2a35;
                border: 1px solid #444455;
                border-radius: 6px;
                color: #ffffff;
                font-size: 13px;
            }
            QListWidget::item {
                padding: 6px;
            }
            QListWidget::item:selected {
                background-color: #6c5ce7;
                color: #ffffff;
            }
            QTabWidget::pane {
                border: 1px solid #444455;
                border-radius: 6px;
                background-color: #1e1e24;
            }
            QTabBar::tab {
                background-color: #2a2a35;
                color: #cccccc;
                padding: 8px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #6c5ce7;
                color: #ffffff;
            }
        """)

        main_layout = QtWidgets.QVBoxLayout(self)
        
        self.tabs = QtWidgets.QTabWidget()
        
        # Tab 1: Geral Settings
        self.tab_general = QtWidgets.QWidget()
        self._init_general_tab()
        self.tabs.addTab(self.tab_general, "Geral")
        
        # Tab 2: Templates Edit
        self.tab_templates = QtWidgets.QWidget()
        self._init_templates_tab()
        self.tabs.addTab(self.tab_templates, "Templates")

        main_layout.addWidget(self.tabs)
        
        # Bottom Buttons
        button_box = QtWidgets.QHBoxLayout()
        button_box.addStretch()
        
        self.btn_cancel = QtWidgets.QPushButton("Cancelar")
        self.btn_cancel.setObjectName("btnCancel")
        self.btn_cancel.clicked.connect(self.reject)
        button_box.addWidget(self.btn_cancel)
        
        self.btn_save = QtWidgets.QPushButton("Salvar")
        self.btn_save.clicked.connect(self._save_settings)
        button_box.addWidget(self.btn_save)
        
        main_layout.addLayout(button_box)

    def _init_general_tab(self) -> None:
        """Sets up widgets for General tab (Ollama settings, directories)."""
        layout = QtWidgets.QVBoxLayout(self.tab_general)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)

        # Ollama URL
        layout.addWidget(QtWidgets.QLabel("Endereço do Ollama:"))
        self.txt_ollama_url = QtWidgets.QLineEdit()
        layout.addWidget(self.txt_ollama_url)

        # Model Selectors
        layout.addWidget(QtWidgets.QLabel("Modelo Principal:"))
        self.combo_default_model = QtWidgets.QComboBox()
        self.combo_default_model.setEditable(True)
        layout.addWidget(self.combo_default_model)

        layout.addWidget(QtWidgets.QLabel("Modelo de Fallback (Menor/CPU):"))
        self.combo_fallback_model = QtWidgets.QComboBox()
        self.combo_fallback_model.setEditable(True)
        layout.addWidget(self.combo_fallback_model)

        # Scan for models button
        self.btn_fetch_models = QtWidgets.QPushButton("Carregar Modelos do Ollama")
        self.btn_fetch_models.clicked.connect(self._fetch_ollama_models)
        layout.addWidget(self.btn_fetch_models)

        # RAG Reports Directory
        layout.addWidget(QtWidgets.QLabel("Diretório de Laudos (RAG):"))
        dir_layout = QtWidgets.QHBoxLayout()
        self.txt_laudos_dir = QtWidgets.QLineEdit()
        self.btn_browse_dir = QtWidgets.QPushButton("...")
        self.btn_browse_dir.setFixedWidth(40)
        self.btn_browse_dir.clicked.connect(self._browse_laudos_dir)
        dir_layout.addWidget(self.txt_laudos_dir)
        dir_layout.addWidget(self.btn_browse_dir)
        layout.addLayout(dir_layout)
        
        layout.addStretch()

    def _init_templates_tab(self) -> None:
        """Sets up widgets for Templates tab (joint list & template edit)."""
        layout = QtWidgets.QHBoxLayout(self.tab_templates)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)

        # Left list of joints
        self.list_joints = QtWidgets.QListWidget()
        self.list_joints.addItems(["Joelho", "Ombro", "Quadril", "Tornozelo", "Punho", "Cotovelo"])
        self.list_joints.setFixedWidth(120)
        self.list_joints.currentRowChanged.connect(self._on_joint_changed)
        layout.addWidget(self.list_joints)

        # Right editor pane
        editor_layout = QtWidgets.QVBoxLayout()
        editor_layout.addWidget(QtWidgets.QLabel("Editar Estrutura do Template:"))
        self.txt_template_content = QtWidgets.QTextEdit()
        editor_layout.addWidget(self.txt_template_content)
        layout.addLayout(editor_layout)

        # Pre-select first item
        self.list_joints.setCurrentRow(0)

    def _load_settings(self) -> None:
        """Loads and applies parameters from the environment and database."""
        # General configurations
        self.txt_ollama_url.setText(os.getenv("OLLAMA_URL", "http://localhost:11434"))
        self.txt_laudos_dir.setText(os.getenv("LAUDOS_DIR", "D:\\Laudos"))
        
        default_model = os.getenv("DEFAULT_MODEL", "gemma3:4b")
        fallback_model = os.getenv("FALLBACK_MODEL", "qwen2.5:0.5b")

        # Try to pre-populate models
        models = self.llm_client.get_available_models()
        if models:
            self.combo_default_model.addItems(models)
            self.combo_fallback_model.addItems(models)
            
            # Find and set indexes
            def_idx = self.combo_default_model.findText(default_model)
            if def_idx >= 0: self.combo_default_model.setCurrentIndex(def_idx)
            else: self.combo_default_model.setEditText(default_model)
                
            fall_idx = self.combo_fallback_model.findText(fallback_model)
            if fall_idx >= 0: self.combo_fallback_model.setCurrentIndex(fall_idx)
            else: self.combo_fallback_model.setEditText(fallback_model)
        else:
            self.combo_default_model.setEditText(default_model)
            self.combo_fallback_model.setEditText(fallback_model)

    def _fetch_ollama_models(self) -> None:
        """Queries Ollama and refreshes the dropdowns."""
        self.llm_client.base_url = self.txt_ollama_url.text().strip()
        models = self.llm_client.get_available_models()
        
        if models:
            # Store current text inputs
            curr_def = self.combo_default_model.currentText()
            curr_fall = self.combo_fallback_model.currentText()

            self.combo_default_model.clear()
            self.combo_fallback_model.clear()
            
            self.combo_default_model.addItems(models)
            self.combo_fallback_model.addItems(models)

            # Restore or set default
            def_idx = self.combo_default_model.findText(curr_def)
            if def_idx >= 0: self.combo_default_model.setCurrentIndex(def_idx)
            else: self.combo_default_model.setEditText(curr_def)

            fall_idx = self.combo_fallback_model.findText(curr_fall)
            if fall_idx >= 0: self.combo_fallback_model.setCurrentIndex(fall_idx)
            else: self.combo_fallback_model.setEditText(curr_fall)

            QtWidgets.QMessageBox.information(self, "Sucesso", f"Encontrados {len(models)} modelos no Ollama.")
        else:
            QtWidgets.QMessageBox.warning(self, "Aviso", "Não foi possível conectar ao Ollama ou nenhuma model encontrada.")

    def _browse_laudos_dir(self) -> None:
        """Opens a directory browser for the RAG source folder."""
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Selecionar Pasta de Laudos", self.txt_laudos_dir.text())
        if folder:
            self.txt_laudos_dir.setText(os.path.normpath(folder))

    def _on_joint_changed(self, row: int) -> None:
        """Loads the template of the selected joint into the editor text field."""
        if row < 0:
            return
        # Save previous edits to database if needed, but since it's simple let's just save templates when the dialog saves
        joint = self.list_joints.item(row).text().lower()
        template = self.user_profile.get_template(joint)
        self.txt_template_content.setPlainText(template)

    def _save_settings(self) -> None:
        """Saves values to .env config file and template database."""
        try:
            # 1. Save active template in editor before writing all
            curr_row = self.list_joints.currentRow()
            if curr_row >= 0:
                joint = self.list_joints.item(curr_row).text().lower()
                self.user_profile.update_template(joint, self.txt_template_content.toPlainText())

            # 2. Write variables back to .env
            env_lines = [
                f"OLLAMA_URL={self.txt_ollama_url.text().strip()}",
                f"DEFAULT_MODEL={self.combo_default_model.currentText().strip()}",
                f"FALLBACK_MODEL={self.combo_fallback_model.currentText().strip()}",
                f"LAUDOS_DIR={self.txt_laudos_dir.text().strip()}",
                "DB_PATH=copilot_data.db",
                "FAISS_INDEX_PATH=copilot_index.index"
            ]
            
            with open(".env", "w", encoding="utf-8") as f:
                f.write("\n".join(env_lines) + "\n")
            
            # Reload environment configs locally
            os.environ["OLLAMA_URL"] = self.txt_ollama_url.text().strip()
            os.environ["DEFAULT_MODEL"] = self.combo_default_model.currentText().strip()
            os.environ["FALLBACK_MODEL"] = self.combo_fallback_model.currentText().strip()
            os.environ["LAUDOS_DIR"] = self.txt_laudos_dir.text().strip()

            logger.info("Settings saved successfully to .env and DB.")
            self.accept()
        except Exception as e:
            # Resilience pipeline
            logger.error("Failed to save settings. Error: %s", str(e), exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Erro", f"Falha ao salvar as configurações:\n{str(e)}")
