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

def inspect_all_controls() -> None:
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
        
        logger.info("Total of %d controls found inside Editor window. Printing those with area > 1000:", len(all_controls))
        
        for idx, ctrl in enumerate(all_controls):
            rect = ctrl.BoundingRectangle
            width = rect.width() if rect else 0
            height = rect.height() if rect else 0
            area = width * height
            
            # Print details of controls that are reasonably large
            if area > 1000:
                logger.info("[%d] Type: %s | Name: '%s' | AutomationId: '%s' | Size: %dx%d (Area: %d) | Class: '%s' | HWND: %s", 
                            idx, ctrl.ControlTypeName, ctrl.Name, ctrl.AutomationId, 
                            width, height, area, ctrl.ClassName, str(ctrl.NativeWindowHandle))
            
    except Exception as e:
        logger.error("Failed traversal: %s", str(e), exc_info=True)

if __name__ == "__main__":
    inspect_all_controls()
