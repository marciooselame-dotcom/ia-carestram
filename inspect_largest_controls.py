import sys
import os
import logging

# Ensure project root is in path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import win32gui
import uiautomation as auto

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pacs_inspector")

def inspect_largest() -> None:
    found_hwnd = 0
    
    def enum_windows_callback(hwnd: int, extra: None) -> bool:
        nonlocal found_hwnd
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd).lower()
            if "editor de relat" in title or "relat" in title:
                if "vscode" not in title and "antigravity" not in title:
                    found_hwnd = hwnd
                    return False
        return True

    win32gui.EnumWindows(enum_windows_callback, None)
    
    if not found_hwnd:
        logger.error("Editor de relatórios window not found!")
        return
        
    logger.info("Found Editor Window: %s (HWND: %d)", win32gui.GetWindowText(found_hwnd), found_hwnd)
    
    try:
        win_ctrl = auto.ControlFromHandle(found_hwnd)
        if not win_ctrl.Exists(2):
            logger.error("Could not bind UIAutomation to HWND.")
            return
            
        all_controls = []
        
        def traverse(control: auto.Control) -> None:
            all_controls.append(control)
            for child in control.GetChildren():
                traverse(child)
                
        traverse(win_ctrl)
        
        # Sort controls by bounding area size descending
        ctrls_with_area = []
        for ctrl in all_controls:
            rect = ctrl.BoundingRectangle
            width = rect.width() if rect else 0
            height = rect.height() if rect else 0
            area = width * height
            ctrls_with_area.append((area, width, height, ctrl))
            
        ctrls_with_area.sort(key=lambda x: x[0], reverse=True)
        
        logger.info("Top 25 Largest Controls in Editor Window:")
        for idx, (area, w, h, ctrl) in enumerate(ctrls_with_area[:25]):
            # Try getting value or text
            text_val = ""
            try:
                if hasattr(ctrl, 'GetValuePattern') and ctrl.GetValuePattern():
                    text_val = ctrl.GetValuePattern().Value
                elif ctrl.GetTextPattern():
                    text_val = ctrl.GetTextPattern().DocumentRange.GetText(100) # first 100 chars
            except Exception:
                pass
                
            logger.info("#%d | Type: %s | Class: '%s' | HWND: %s | Size: %dx%d (Area: %d) | Name: '%s' | Text: '%s'", 
                        idx + 1, ctrl.ControlTypeName, ctrl.ClassName, str(ctrl.NativeWindowHandle), 
                        w, h, area, ctrl.Name, text_val[:60].strip().replace("\n", " ") if text_val else "[empty]")
            
    except Exception as e:
        logger.error("Failed: %s", str(e), exc_info=True)

if __name__ == "__main__":
    inspect_largest()
