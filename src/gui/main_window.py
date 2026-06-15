import os
import logging
from typing import Optional, List, Tuple
from PySide6 import QtWidgets, QtCore, QtGui

from src.core.pacs_detector import PACSDetector
from src.core.llm_client import LLMClient
from src.core.rag_engine import RAGEngine
from src.core.user_profile import UserProfile

logger = logging.getLogger(__name__)

class LLMWorker(QtCore.QThread):
    """Background worker thread to handle streaming LLM completions without freezing the GUI."""
    chunk_received = QtCore.Signal(str)
    finished = QtCore.Signal(str)
    error = QtCore.Signal(str)

    def __init__(self, client: LLMClient, system_prompt: str, user_prompt: str) -> None:
        """Initializes the worker with prompt data."""
        super().__init__()
        self.client = client
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.full_response = ""

    def run(self) -> None:
        """Runs the streaming chat loop in the background thread."""
        try:
            generator = self.client.stream_chat(self.system_prompt, self.user_prompt)
            for chunk in generator:
                self.full_response += chunk
                self.chunk_received.emit(chunk)
            self.finished.emit(self.full_response)
        except Exception as e:
            logger.error("Error running LLM worker thread: %s", str(e), exc_info=True)
            self.error.emit(str(e))


class RAGScanWorker(QtCore.QThread):
    """Background worker to scan and index reports from the configured folder."""
    finished = QtCore.Signal(int)
    error = QtCore.Signal(str)

    def __init__(self, rag_engine: RAGEngine, folder_path: str) -> None:
        """Initializes the worker with directory information."""
        super().__init__()
        self.rag_engine = rag_engine
        self.folder_path = folder_path

    def run(self) -> None:
        """Runs the file scanning and vector indexing in the background."""
        try:
            count = self.rag_engine.scan_directory(self.folder_path)
            self.finished.emit(count)
        except Exception as e:
            logger.error("Error running RAG scan worker thread: %s", str(e), exc_info=True)
            self.error.emit(str(e))


class MainWindow(QtWidgets.QMainWindow):
    """The main interface for the Radiology Report Copilot.
    
    Provides buttons to trigger text captures, formatting, reviews, templates,
    and a live preview pane to edit/insert content directly back into Carestream PACS.
    """

    def __init__(self) -> None:
        """Initializes components, configures window traits, and triggers initial RAG scan."""
        super().__init__()
        self.setWindowTitle("Radiology Copilot")
        self.setMinimumSize(360, 680)
        self.resize(360, 720)
        
        # Make the window float on top by default
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Window)

        # Initialize engines
        self.pacs_detector = PACSDetector()
        self.llm_client = LLMClient()
        self.rag_engine = RAGEngine()
        self.user_profile = UserProfile()

        # State tracking variables
        self.last_ai_generated_text: str = ""
        self.last_pacs_captured_text: str = ""
        self.detected_joint_type: str = "joelho"
        
        self._init_ui()
        
        # Start background scan of laudos folder on startup
        self._trigger_background_rag_scan()

    def _init_ui(self) -> None:
        """Defines layouts, components, and applies premium dark theme styles."""
        # CSS Stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121216;
                color: #ffffff;
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            QLabel#titleLabel {
                color: #6c5ce7;
                font-size: 18px;
                font-weight: bold;
                padding-bottom: 5px;
            }
            QLabel#subtitleLabel {
                color: #888899;
                font-size: 11px;
                padding-bottom: 15px;
            }
            QPushButton {
                background-color: #1e1e26;
                border: 1px solid #2e2e3a;
                border-radius: 8px;
                color: #e0e0e0;
                font-weight: 600;
                padding: 10px;
                font-size: 12px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #2a2a38;
                border: 1px solid #6c5ce7;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #181822;
            }
            QPushButton#btnFormat {
                background-color: #6c5ce7;
                color: #ffffff;
                border: none;
                font-size: 13px;
                text-align: center;
            }
            QPushButton#btnFormat:hover {
                background-color: #8073e8;
            }
            QPushButton#btnFormat:pressed {
                background-color: #5b4dbf;
            }
            QTextEdit {
                background-color: #181820;
                border: 1px solid #2e2e3a;
                border-radius: 8px;
                color: #e2e2e2;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                padding: 8px;
            }
            QTextEdit:focus {
                border: 1px solid #6c5ce7;
            }
            QProgressBar {
                background-color: #1e1e26;
                border: 1px solid #2e2e3a;
                border-radius: 4px;
                text-align: center;
                color: #ffffff;
                font-size: 10px;
                height: 12px;
            }
            QProgressBar::chunk {
                background-color: #6c5ce7;
                border-radius: 4px;
            }
            QCheckBox {
                color: #cccccc;
                font-size: 12px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #444455;
                border-radius: 4px;
                background-color: #1e1e26;
            }
            QCheckBox::indicator:checked {
                background-color: #6c5ce7;
                image: url(check.png); /* Fallback to standard check box styling */
            }
            QStatusBar {
                background-color: #181820;
                color: #888899;
                font-size: 11px;
                border-top: 1px solid #2e2e3a;
            }
        """)

        # Central Widget & Main Layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        # Header Title
        title_label = QtWidgets.QLabel("RADIOLOGY COPILOT")
        title_label.setObjectName("titleLabel")
        main_layout.addWidget(title_label)
        
        subtitle_label = QtWidgets.QLabel("ASSISTENTE LOCAL DE LAUDOS DE RM MSK")
        subtitle_label.setObjectName("subtitleLabel")
        main_layout.addWidget(subtitle_label)

        # Action Buttons Layout (Grid)
        actions_grid = QtWidgets.QGridLayout()
        actions_grid.setSpacing(8)

        self.btn_format = QtWidgets.QPushButton("⚡ FORMATAR LAUDO")
        self.btn_format.setObjectName("btnFormat")
        self.btn_format.clicked.connect(self._on_format_clicked)
        actions_grid.addWidget(self.btn_format, 0, 0, 1, 2)

        self.btn_review = QtWidgets.QPushButton("📝 Revisar Texto")
        self.btn_review.clicked.connect(self._on_review_clicked)
        actions_grid.addWidget(self.btn_review, 1, 0)

        self.btn_conclusion = QtWidgets.QPushButton("🔎 Gerar Conclusão")
        self.btn_conclusion.clicked.connect(self._on_conclusion_clicked)
        actions_grid.addWidget(self.btn_conclusion, 1, 1)

        self.btn_template = QtWidgets.QPushButton("📋 Inserir Template")
        self.btn_template.clicked.connect(self._on_template_clicked)
        actions_grid.addWidget(self.btn_template, 2, 0)

        self.btn_learn = QtWidgets.QPushButton("🧠 Aprender com este Laudo")
        self.btn_learn.clicked.connect(self._on_learn_clicked)
        actions_grid.addWidget(self.btn_learn, 2, 1)

        main_layout.addLayout(actions_grid)

        # Config & Stay on top controls
        options_layout = QtWidgets.QHBoxLayout()
        
        self.chk_always_on_top = QtWidgets.QCheckBox("Sempre Visível")
        self.chk_always_on_top.setChecked(True)
        self.chk_always_on_top.stateChanged.connect(self._on_always_on_top_changed)
        options_layout.addWidget(self.chk_always_on_top)

        self.chk_auto_apply = QtWidgets.QCheckBox("Auto-Aplicar no PACS")
        self.chk_auto_apply.setChecked(True)
        options_layout.addWidget(self.chk_auto_apply)
        
        main_layout.addLayout(options_layout)

        # Preview area
        preview_header = QtWidgets.QHBoxLayout()
        preview_header.addWidget(QtWidgets.QLabel("Visualização Prévia do Laudo:"))
        
        self.btn_apply_now = QtWidgets.QPushButton("Inserir no PACS")
        self.btn_apply_now.setFixedHeight(26)
        self.btn_apply_now.setFixedWidth(110)
        self.btn_apply_now.setStyleSheet("""
            QPushButton {
                background-color: #3e3e4f;
                padding: 2px 8px;
                font-size: 11px;
                text-align: center;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #6c5ce7;
            }
        """)
        self.btn_apply_now.clicked.connect(self._on_apply_now_clicked)
        preview_header.addWidget(self.btn_apply_now)
        main_layout.addLayout(preview_header)

        self.txt_preview = QtWidgets.QTextEdit()
        self.txt_preview.setPlaceholderText("O laudo processado aparecerá aqui...")
        main_layout.addWidget(self.txt_preview)

        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # Config Button at bottom
        self.btn_config = QtWidgets.QPushButton("⚙️ Abrir Configurações")
        self.btn_config.setStyleSheet("text-align: center;")
        self.btn_config.clicked.connect(self._on_config_clicked)
        main_layout.addWidget(self.btn_config)

        # Status Bar
        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Copilot pronto. Inicializando...")

    def _on_always_on_top_changed(self, state: int) -> None:
        """Adjusts the window's top-level status dynamically based on checkbox."""
        if state == QtCore.Qt.Checked.value:
            self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowStaysOnTopHint)
        self.show()

    def _trigger_background_rag_scan(self) -> None:
        """Starts background scanning thread for medical files."""
        folder_path = os.getenv("LAUDOS_DIR", "D:\\Laudos")
        
        # Check C:\Laudos fallback if D:\ doesn't exist
        if not os.path.exists(folder_path):
            folder_path = "C:\\Laudos"
            if not os.path.exists(folder_path):
                # Fallback to current workspace directory
                folder_path = os.path.join(os.getcwd(), "Laudos_Local")
                os.makedirs(folder_path, exist_ok=True)

        self.status_bar.showMessage(f"Escaneando pasta RAG: {folder_path}...")
        self.progress_bar.setRange(0, 0) # Indeterminate loading bar

        self.rag_worker = RAGScanWorker(self.rag_engine, folder_path)
        self.rag_worker.finished.connect(self._on_rag_scan_finished)
        self.rag_worker.error.connect(self._on_rag_scan_error)
        self.rag_worker.start()

    def _on_rag_scan_finished(self, new_docs_count: int) -> None:
        """Triggered when background RAG folder indexing completes."""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.status_bar.showMessage(f"Varredura RAG concluída. {new_docs_count} laudos indexados/atualizados.")
        logger.info("RAG directory scan complete. Index matches database updates.")

    def _on_rag_scan_error(self, err_msg: str) -> None:
        """Triggered when background RAG folder indexing errors."""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage("Erro ao carregar banco RAG.")
        logger.error("RAG scan failed: %s", err_msg)

    def _capture_pacs_text(self) -> Optional[str]:
        """Brings the Carestream PACS window to focus and copies text from its editor."""
        hwnd = self.pacs_detector.find_pacs_window()
        if not hwnd:
            self.status_bar.showMessage("PACS Carestream não detectado!")
            QtWidgets.QMessageBox.warning(
                self, 
                "PACS não encontrado", 
                "Por favor, certifique-se de que a janela do editor de laudos do Carestream PACS está aberta."
            )
            return None

        # Focus window
        self.status_bar.showMessage("Focando editor do PACS...")
        if not self.pacs_detector.focus_window(hwnd):
            self.status_bar.showMessage("Falha ao focar janela do PACS.")
            return None

        # Grab text
        time_delay = 0.4
        QtCore.QThread.msleep(int(time_delay * 1000))
        captured = self.pacs_detector.capture_text_via_clipboard()
        
        if not captured:
            self.status_bar.showMessage("Editor de laudos do PACS está vazio ou ilegível.")
            return None

        self.last_pacs_captured_text = captured
        # Auto-identify joint type
        self.detected_joint_type = self.rag_engine.identify_joint_type(captured)
        self.status_bar.showMessage(f"Texto capturado ({self.detected_joint_type.upper()}).")
        return captured

    def _on_format_clicked(self) -> None:
        """Action handler to capture raw text, query RAG context, and stream formatted structured report."""
        raw_text = self._capture_pacs_text()
        if not raw_text:
            return

        self.txt_preview.clear()
        self.status_bar.showMessage("Processando RAG e consultando IA local...")
        self.progress_bar.setRange(0, 0) # Indeterminate loading

        # 1. Search RAG context
        rag_context = self.rag_engine.search_similar(raw_text, self.detected_joint_type, k=2)

        # 2. Get User Preferences Style
        prefs = self.user_profile.get_style_preferences_prompt()

        # 3. Build prompts
        system_prompt, user_prompt = self.llm_client.build_format_prompts(
            raw_text=raw_text,
            joint_type=self.detected_joint_type,
            rag_context=rag_context,
            user_preferences=prefs
        )

        # 4. Trigger streaming
        self._start_llm_generation(system_prompt, user_prompt)

    def _on_review_clicked(self) -> None:
        """Action handler to refine/improve an existing complete report without changing clinical findings."""
        raw_text = self._capture_pacs_text()
        if not raw_text:
            return

        self.txt_preview.clear()
        self.status_bar.showMessage("Revisando laudo com IA local...")
        self.progress_bar.setRange(0, 0)

        prefs = self.user_profile.get_style_preferences_prompt()
        system_prompt, user_prompt = self.llm_client.build_review_prompts(raw_text, prefs)

        self._start_llm_generation(system_prompt, user_prompt)

    def _on_conclusion_clicked(self) -> None:
        """Action handler to generate a numbered list of conclusions from the findings."""
        raw_text = self._capture_pacs_text()
        if not raw_text:
            return

        self.txt_preview.clear()
        self.status_bar.showMessage("Gerando conclusão...")
        self.progress_bar.setRange(0, 0)

        system_prompt, user_prompt = self.llm_client.build_conclusion_prompts(raw_text)

        self._start_llm_generation(system_prompt, user_prompt)

    def _start_llm_generation(self, system_prompt: str, user_prompt: str) -> None:
        """Spawns the background worker to execute streaming LLM inference."""
        self.llm_worker = LLMWorker(self.llm_client, system_prompt, user_prompt)
        self.llm_worker.chunk_received.connect(self._on_llm_chunk)
        self.llm_worker.finished.connect(self._on_llm_finished)
        self.llm_worker.error.connect(self._on_llm_error)
        self.llm_worker.start()

    def _clean_llm_output(self, text: str) -> str:
        """Clean LLM output: remove meta-text, fix typos, remove contradictory lines."""
        import re
        lines = text.splitlines()
        
        # --- STEP 1: Remove meta-text/prompt echoes ---
        cleaned = []
        in_report = False
        meta_patterns = [
            r"^ola[^a-záéíóú]", r"^olá[^a-záéíóú]", r"\[nome", r"\[data",
            r"^obrigado", r"^laudo completo", r"^retorne o laudo",
            r"^você é um", r"^sua tarefa", r"^regra",
            r"^conclusão:", r"^conclusao:",
        ]
        for line in lines:
            stripped = line.strip()
            if not in_report and not stripped:
                continue
            if any(re.search(p, stripped, re.IGNORECASE) for p in meta_patterns):
                continue
            if not in_report and (stripped.startswith("---") or stripped.startswith("**")):
                continue
            in_report = True
            cleaned.append(stripped)
        
        # --- STEP 2: Strip everything after garbage IMPRESSÃO/CONCLUSÃO repeats ---
        full = "\n".join(cleaned)
        # Find all IMPRESSÃO positions
        impressao_positions = [m.start() for m in re.finditer(r"^IMPRESSÃO:", full, re.MULTILINE)]
        if len(impressao_positions) > 1:
            # Keep only up to the first IMPRESSÃO + its content until next section
            full = full[:impressao_positions[1]].strip()
        # Also remove anything after a second "CONCLUSÃO" or "[" section
        conclusao_pos = [m.start() for m in re.finditer(r"^\[CONCLUSÃO", full, re.MULTILINE)]
        if conclusao_pos:
            full = full[:conclusao_pos[0]].strip()
        bracket_pos = [m.start() for m in re.finditer(r"^\[IMPRESSÃO", full, re.MULTILINE)]
        if bracket_pos:
            full = full[:bracket_pos[0]].strip()
        
        # --- STEP 3: Fix common typos (without adding new words) ---
        typo_fixes = [
            (r"\bedemas ubcondral\b", "edema subcondral"),
            (r"\bedemas subcondral\b", "edema subcondral"),
            (r"\bedema ubcondral\b", "edema subcondral"),
            (r"\bpredominandoo\b", "predominando"),
            (r"\bpredominado\b", "predominando"),
            (r"\bquadríceps\b", "quadríceps"),
            (r"\bubcondral\b", "subcondral"),
            (r"\bderr?am\s+articular\b", "Derrame articular"),
            (r"\bderr?am\b", "derrame"),
        ]
        for pattern, replacement in typo_fixes:
            full = re.sub(pattern, replacement, full, flags=re.IGNORECASE)
        
        # --- STEP 4: Remove contradictory normal lines ---
        # If a pathological finding exists, remove the corresponding normal template line
        lines_out = full.splitlines()
        patho_keywords = []
        # Check for pathological keywords
        for line in lines_out:
            low = line.lower()
            if "rotura" in low or "ruptura" in low or "menisco" in low and ("degener" in low or "horizontal" in low or "tear" in low):
                patho_keywords.append("menisco")
            if "erosão" in low or "erosões" in low or "eros" in low:
                patho_keywords.append("erosao")
            if "derrame" in low or "efusão" in low:
                patho_keywords.append("derrame")
            if "irregularidades" in low or "irregularidade" in low:
                patho_keywords.append("irregularidade")
            if "edema" in low:
                patho_keywords.append("edema")
        
        # Remove lines that contradict findings
        contradiction_patterns = {
            "menisco": [
                r"meniscos?\s+de\s+morfologia\s+e\s+sinal\s+normais",
                r"meniscos?\s+sem\s+alterações",
            ],
            "erosao": [
                r"superfícies?\s+condrais?\s+.*regulares?\s*,?\s*sem\s+erosões",
                r"superfícies?\s+articulares?\s+.*regulares?\s*,?\s*sem\s+erosões",
            ],
            "derrame": [
                r"não\s+h[áa]\s+derrame\s+articular",
                r"ausência\s+de\s+derrame",
            ],
            "irregularidade": [
                r"superfícies?\s+articulares?\s+.*regulares?",
                r"superfícies?\s+condrais?\s+.*regulares?",
            ],
        }
        
        filtered_lines = []
        for line in lines_out:
            low = line.lower()
            remove = False
            for kw in set(patho_keywords):
                for pat in contradiction_patterns.get(kw, []):
                    if re.search(pat, low):
                        remove = True
                        break
                if remove:
                    break
            if not remove:
                filtered_lines.append(line)
        
        result = "\n".join(filtered_lines).strip()
        return result if result else text.strip()

    def _on_llm_chunk(self, token: str) -> None:
        """Appends tokens directly to the preview pane as they are generated."""
        self.txt_preview.insertPlainText(token)
        # Move cursor to end to auto-scroll
        self.txt_preview.moveCursor(QtGui.QTextCursor.End)

    def _on_llm_finished(self, full_response: str) -> None:
        """Post-inference handler.

        Cleans output, updates preview, and optionally auto-pastes to PACS.
        """
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        
        # Clean LLM output (remove meta-text, greetings, etc.)
        cleaned = self._clean_llm_output(full_response)
        
        # Update preview with cleaned text
        self.txt_preview.clear()
        self.txt_preview.setPlainText(cleaned)
        self.status_bar.showMessage("Processamento concluído.")
        
        self.last_ai_generated_text = cleaned

        # Auto-apply if checkbox checked
        if self.chk_auto_apply.isChecked():
            self._on_apply_now_clicked()

    def _on_llm_error(self, err_msg: str) -> None:
        """Handles background LLM execution failures."""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage("Erro ao processar laudo com a IA local.")
        QtWidgets.QMessageBox.critical(self, "Erro na IA", f"Ocorreu um erro ao comunicar com a IA local:\n{err_msg}")

    def _on_apply_now_clicked(self) -> None:
        """Forces replacement of Carestream editor contents with the preview pane content."""
        text = self.txt_preview.toPlainText().strip()
        if not text:
            self.status_bar.showMessage("Nenhum texto gerado para aplicar.")
            return

        hwnd = self.pacs_detector.find_pacs_window()
        if not hwnd:
            self.status_bar.showMessage("PACS Carestream não detectado!")
            return

        self.status_bar.showMessage("Substituindo texto no PACS...")
        if self.pacs_detector.focus_window(hwnd):
            QtCore.QThread.msleep(200)
            if self.pacs_detector.replace_text_via_clipboard(text):
                self.status_bar.showMessage("Texto substituído com sucesso no PACS.")
            else:
                self.status_bar.showMessage("Falha ao colar texto no editor do PACS.")

    def _on_template_clicked(self) -> None:
        """Opens a quick menu containing default templates, inserting selection directly."""
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1e1e26;
                color: #ffffff;
                border: 1px solid #444455;
            }
            QMenu::item {
                padding: 6px 20px;
            }
            QMenu::item:selected {
                background-color: #6c5ce7;
            }
        """)

        joints = ["Joelho", "Ombro", "Quadril", "Tornozelo", "Punho", "Cotovelo"]
        for joint in joints:
            action = menu.addAction(joint)
            action.triggered.connect(lambda checked=False, j=joint: self._insert_template_by_name(j))

        # Show menu below the template button
        menu.exec_(self.btn_template.mapToGlobal(QtCore.QPoint(0, self.btn_template.height())))

    def _insert_template_by_name(self, joint_name: str) -> None:
        """Fetches joint template and populates both preview and Carestream PACS editor."""
        template = self.user_profile.get_template(joint_name)
        if template:
            self.txt_preview.setPlainText(template)
            
            # Immediately paste to PACS if checked, or always paste as template
            hwnd = self.pacs_detector.find_pacs_window()
            if hwnd:
                self.pacs_detector.focus_window(hwnd)
                QtCore.QThread.msleep(200)
                self.pacs_detector.replace_text_via_clipboard(template)
                self.status_bar.showMessage(f"Template de RM de {joint_name} inserido.")
            else:
                self.status_bar.showMessage(f"Template carregado na tela. Abra o PACS para colar.")

    def _on_learn_clicked(self) -> None:
        """Extracts changes between original AI generation and manual corrections.

        Updates the local database with preferred vocabulary and indexes
        the final report for future matching.
        """
        # 1. Capture the current text in the PACS (representing the finalized physician version)
        final_text = self._capture_pacs_text()
        if not final_text:
            return

        # 2. Check if we have an AI version to compare against
        if not self.last_ai_generated_text:
            # If no AI text was generated in this session, we still index the report in RAG
            self.status_bar.showMessage("Salvando laudo final no banco RAG...")
            if self.rag_engine.add_custom_laudo(final_text):
                self.status_bar.showMessage("Laudo registrado e indexado com sucesso no RAG!")
                QtWidgets.QMessageBox.information(self, "Sucesso", "Laudo salvo no banco RAG do médico.")
            else:
                self.status_bar.showMessage("Falha ao salvar laudo.")
            return

        self.status_bar.showMessage("Analisando modificações do médico...")
        
        # 3. Learn vocabulary preferences
        learned = self.user_profile.learn_from_edit(
            ai_text=self.last_ai_generated_text,
            final_text=final_text
        )

        # 4. Save final laudo to RAG
        self.rag_engine.add_custom_laudo(final_text)

        # 5. Summarize learnings
        self.status_bar.showMessage("Processamento de aprendizado concluído.")
        if learned:
            learned_summary = ", ".join([f"'{orig}' -> '{pref}'" for orig, pref in learned])
            QtWidgets.QMessageBox.information(
                self,
                "Aprendizado Concluído",
                f"Padrões de estilo e novos termos aprendidos:\n{learned_summary}\n\nO laudo final foi adicionado ao RAG."
            )
        else:
            QtWidgets.QMessageBox.information(
                self,
                "Laudo Indexado",
                "O laudo final foi indexado ao RAG. Nenhum termo de vocabulário novo foi detectado."
            )

        # Reset last AI text to avoid repeating comparisons
        self.last_ai_generated_text = ""

    def _on_config_clicked(self) -> None:
        """Opens the ConfigDialog modal and triggers database reload on close."""
        from src.gui.config_dialog import ConfigDialog
        dialog = ConfigDialog(self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            # Reload settings in engines
            self.llm_client = LLMClient()
            
            # Re-init RAGEngine to load new variables
            old_rag = self.rag_engine
            self.rag_engine = RAGEngine()
            old_rag.close()

            # Re-init UserProfile
            old_profile = self.user_profile
            self.user_profile = UserProfile()
            old_profile.close()
            
            self.status_bar.showMessage("Configurações atualizadas com sucesso.")
            # Trigger background scan of the newly configured folder
            self._trigger_background_rag_scan()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Ensures all database connections are safely closed when exiting."""
        try:
            self.rag_engine.close()
            self.user_profile.close()
        except Exception:
            pass
        event.accept()
