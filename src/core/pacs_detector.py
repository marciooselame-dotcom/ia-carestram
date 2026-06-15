import logging
import time
from typing import Optional, List
import win32gui
import win32con
import pywinauto
from pywinauto import Application
from pywinauto.keyboard import send_keys

# Setup logging
logger = logging.getLogger(__name__)

class PACSDetector:
    """Detects Carestream PACS windows and performs text retrieval and replacement operations.
    
    Uses pywinauto and win32 APIs to find Carestream Vue PACS windows, brings them to focus,
    and captures or replaces the report editor text using active automation controls or clipboard fallback.
    """

    def __init__(self) -> None:
        """Initializes PACSDetector with target window title substrings."""
        self.target_substrings: List[str] = [
            "carestream",
            "vue pacs",
            "vue motion",
            "vue client",
            "pacs"
        ]
        self.editor_hwnd: Optional[int] = None

    def find_pacs_window(self) -> Optional[int]:
        """Scans all open windows to find a matching Carestream/Vue PACS window.

        Prioritizes the actual editor window (e.g. 'Editor de relatórios') over the main PACS client.

        Returns:
            Optional[int]: The window handle (HWND) if found, otherwise None.
        """
        editor_hwnd: Optional[int] = None
        viewer_hwnd: Optional[int] = None
        
        # Priority 1 title patterns (actual report editors)
        editor_patterns = ["editor de relat", "editor de laudo", "digitação de laudo", "laudo"]
        
        # Priority 2 title patterns (main viewer window fallbacks)
        viewer_patterns = ["vue pacs", "vue client", "carestream", "vue motion", "pacs"]
        
        # Windows to ignore (so we don't accidentally control IDEs or our own tool)
        blacklist = [
            "vscode", "visual studio", "antigravity", "radiology report copilot", 
            "import_templates", "inspect_pacs", "inspect_largest_controls", "inspect_pacs_mdiclient"
        ]

        def enum_windows_callback(hwnd: int, extra: None) -> bool:
            nonlocal editor_hwnd, viewer_hwnd
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd).lower()
                
                # Verify blacklist
                for black in blacklist:
                    if black in title:
                        return True # Continue enumeration, skip this window
                
                # Check for editor match (Priority 1)
                for pattern in editor_patterns:
                    if pattern in title:
                        editor_hwnd = hwnd
                        logger.info("Found active PACS Editor window matching '%s': %s (HWND: %d)", pattern, win32gui.GetWindowText(hwnd), hwnd)
                        return False # Stop enumeration immediately, we found the editor
                
                # Check for viewer match (Priority 2 fallback)
                if not viewer_hwnd:
                    for pattern in viewer_patterns:
                        if pattern in title:
                            viewer_hwnd = hwnd
                            logger.info("Found fallback PACS Viewer window matching '%s': %s (HWND: %d)", pattern, win32gui.GetWindowText(hwnd), hwnd)
                            break # Continue search, we might find a proper Editor window
            return True

        try:
            win32gui.EnumWindows(enum_windows_callback, None)
        except Exception as e:
            logger.error("Error enumerating windows: %s", str(e), exc_info=True)

        # Return the editor window if found, otherwise the viewer window
        final_hwnd = editor_hwnd or viewer_hwnd
        if final_hwnd:
            logger.info("Selected target window HWND: %d (%s)", final_hwnd, win32gui.GetWindowText(final_hwnd))
        return final_hwnd

    def _find_editor_control_hwnd(self, parent_hwnd: int) -> Optional[int]:
        """Enumerates child windows to locate the TX Text Control (main report editor).

        The Carestream report editor uses TX Text Control (.NET), with window class
        'WindowsForms10.TX26_DOTNET.*'. This method prioritizes TX controls over
        generic Edit controls to avoid targeting metadata fields.

        Args:
            parent_hwnd (int): Parent window handle.

        Returns:
            Optional[int]: The editor control handle if found.
        """
        tx_hwnd: Optional[int] = None
        fallback_hwnd: Optional[int] = None
        
        # Priority: TX Text Control > RichEdit > generic Edit
        def enum_child_callback(hwnd: int, extra: None) -> bool:
            nonlocal tx_hwnd, fallback_hwnd
            if not win32gui.IsWindowVisible(hwnd):
                return True
            class_name = win32gui.GetClassName(hwnd).lower()
            try:
                rect = win32gui.GetWindowRect(hwnd)
                width = rect[2] - rect[0]
                height = rect[3] - rect[1]
            except Exception:
                width, height = 0, 0

            # Priority 1: TX Text Control (the actual report editor)
            if "tx26" in class_name or "tx_dotnet" in class_name:
                tx_hwnd = hwnd
                logger.info("Found TX Text Control editor: Class='%s', HWND=%d, Size=%dx%d", class_name, hwnd, width, height)
                return False  # Stop immediately, this is the correct control
            
            # Priority 2: RichEdit (large enough to be the editor)
            if ("riched" in class_name or "document" in class_name) and width >= 300 and height >= 150:
                if not fallback_hwnd:
                    fallback_hwnd = hwnd
                    logger.info("Found RichEdit fallback: Class='%s', HWND=%d, Size=%dx%d", class_name, hwnd, width, height)
            
            return True

        try:
            win32gui.EnumChildWindows(parent_hwnd, enum_child_callback, None)
        except Exception as e:
            logger.warning("Failed to enumerate child windows: %s", str(e))

        result = tx_hwnd or fallback_hwnd
        if result:
            logger.info("Selected editor control HWND: %d", result)
        else:
            logger.warning("No TX or RichEdit control found in window %d.", parent_hwnd)
        return result

    def focus_window(self, hwnd: int) -> bool:
        """Brings the window with the given handle to the foreground and focuses it.

        Args:
            hwnd (int): The window handle to focus.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            
            # 1. Connect and focus top-level window via pywinauto
            app = Application().connect(handle=hwnd)
            win = app.window(handle=hwnd)
            win.set_focus()
            time.sleep(0.3) # Wait for window focus transition
            
            # 2. Store active editor HWND
            self.editor_hwnd = self._find_editor_control_hwnd(hwnd)
            
            # 3. Try focusing the edit control explicitly using uiautomation or coordinate click
            try:
                import uiautomation as auto
                win_ctrl = auto.ControlFromHandle(hwnd)
                if win_ctrl.Exists(1):
                    edit_ctrl = win_ctrl.EditControl()
                    if edit_ctrl.Exists(0.2):
                        edit_ctrl.SetFocus()
                        if not self.editor_hwnd:
                            self.editor_hwnd = edit_ctrl.NativeWindowHandle
                        logger.info("Focused EditControl via UIAutomation.")
                    else:
                        doc_ctrl = win_ctrl.DocumentControl()
                        if doc_ctrl.Exists(0.2):
                            doc_ctrl.SetFocus()
                            if not self.editor_hwnd:
                                self.editor_hwnd = doc_ctrl.NativeWindowHandle
                            logger.info("Focused DocumentControl via UIAutomation.")
                        else:
                            # Click center of the window workspace
                            rect = win32gui.GetWindowRect(hwnd)
                            x = rect[0] + (rect[2] - rect[0]) // 2
                            y = rect[1] + (rect[3] - rect[1]) // 2
                            pywinauto.mouse.click(coords=(x, y))
                            time.sleep(0.2)
                            logger.info("Clicked center of window (%d, %d) to guarantee editor focus.", x, y)
            except Exception as uia_err:
                logger.warning("UIAutomation editor focus refinement failed: %s.", str(uia_err))

            return True
        except Exception as e:
            logger.error("Failed to focus window %d. Trying win32 fallback. Error: %s", hwnd, str(e), exc_info=True)
            try:
                win32gui.BringWindowToTop(hwnd)
                win32gui.SetForegroundWindow(hwnd)
                time.sleep(0.3)
                return True
            except Exception as e_fallback:
                logger.error("win32 fallback focus also failed: %s", str(e_fallback))
                return False

    def capture_text_via_clipboard(self) -> Optional[str]:
        """Captures text from the focused editor window.
        
        For TX Text Control (Carestream editor): uses UIAutomation Name property.
        Fallback: uses clipboard simulation (Ctrl+A -> Ctrl+C).

        Returns:
            Optional[str]: The captured text if successful, None if empty or failed.
        """
        # Method 1: UIAutomation Name property (works for TX Text Control)
        if self.editor_hwnd:
            try:
                import uiautomation as auto
                ctrl = auto.ControlFromHandle(self.editor_hwnd)
                if ctrl.Exists(1):
                    name_text = ctrl.Name
                    if name_text and len(name_text.strip()) > 10:
                        # Clean up \r\r\n artifacts from WinForms Name property
                        cleaned = name_text.replace("\r\r\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
                        logger.info("Successfully captured %d chars via UIAutomation Name property.", len(cleaned))
                        return cleaned
            except Exception as uia_err:
                logger.warning("UIAutomation Name capture failed: %s. Trying clipboard.", str(uia_err))

        # Method 2: Clipboard simulation (Ctrl+A -> Ctrl+C)
        try:
            import win32clipboard
            
            # Clear clipboard first
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.CloseClipboard()

            send_keys("^a^c")
            time.sleep(0.5)  # Wait for clipboard to populate

            win32clipboard.OpenClipboard()
            try:
                if win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
                    text = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
                    cleaned = str(text).replace("\r\n", "\n").replace("\r", "\n")
                    logger.info("Captured %d chars via clipboard.", len(cleaned))
                    return cleaned
                elif win32clipboard.IsClipboardFormatAvailable(win32con.CF_TEXT):
                    text = win32clipboard.GetClipboardData(win32con.CF_TEXT)
                    return str(text.decode("latin1"))
            finally:
                win32clipboard.CloseClipboard()

        except Exception as e:
            logger.error("Failed to capture text via clipboard. Error: %s", str(e), exc_info=True)
            
        return None

    def replace_text_via_clipboard(self, new_text: str) -> bool:
        """Replaces the full content of the focused editor via clipboard (Ctrl+A -> Ctrl+V).

        For TX Text Control, WM_SETTEXT does not work. The only reliable method is
        clipboard-based text replacement using simulated keystrokes.

        Args:
            new_text (str): The new text to write.

        Returns:
            bool: True if operation completed successfully, False otherwise.
        """
        try:
            import win32clipboard
            
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardText(new_text, win32con.CF_UNICODETEXT)
            win32clipboard.CloseClipboard()

            # Select all and paste
            send_keys("^a")
            time.sleep(0.15)
            send_keys("^v")
            time.sleep(0.3)
            logger.info("Replaced text via clipboard paste (%d chars).", len(new_text))
            return True

        except Exception as e:
            logger.error("Failed to replace text via clipboard. Error: %s", str(e), exc_info=True)
            return False
