import sys
import os
import logging

# Ensure project root is in path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import win32gui

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mdi_inspector")

def inspect_mdi_win32() -> None:
    # Target MDI Client HWND
    mdi_hwnd = 262706
    
    # Check if window exists and is valid
    if not win32gui.IsWindow(mdi_hwnd):
        logger.error("MDI Client HWND 262706 is not a valid window!")
        # Let's search for any window with class containing 'MDICLIENT'
        found_mdiclients = []
        def enum_all_children(hwnd: int, lparam: list) -> bool:
            class_name = win32gui.GetClassName(hwnd).lower()
            if "mdiclient" in class_name:
                lparam.append(hwnd)
            return True
            
        # Enumerate all windows and child windows to find MDICLIENTs
        def enum_top_windows(hwnd: int, extra: None) -> bool:
            try:
                win32gui.EnumChildWindows(hwnd, enum_all_children, found_mdiclients)
            except Exception:
                pass
            return True
            
        win32gui.EnumWindows(enum_top_windows, None)
        if not found_mdiclients:
            logger.error("No MDICLIENT windows found on the entire desktop!")
            return
        mdi_hwnd = found_mdiclients[0]
        logger.info("Found MDICLIENT handle dynamically: %d", mdi_hwnd)

    logger.info("Enumerating Win32 child windows of MDICLIENT (HWND: %d)...", mdi_hwnd)
    
    child_windows = []
    
    def enum_child_callback(hwnd: int, extra: None) -> bool:
        title = win32gui.GetWindowText(hwnd)
        class_name = win32gui.GetClassName(hwnd)
        rect = win32gui.GetWindowRect(hwnd)
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        child_windows.append((hwnd, class_name, title, width, height))
        return True

    try:
        win32gui.EnumChildWindows(mdi_hwnd, enum_child_callback, None)
    except Exception as e:
        logger.error("Failed to enumerate child windows: %s", str(e))
        return

    logger.info("Found %d child windows inside MDICLIENT:", len(child_windows))
    for hwnd, class_name, title, w, h in child_windows:
        logger.info("HWND: %d | Class: '%s' | Size: %dx%d | Title: '%s'", 
                    hwnd, class_name, w, h, title)

if __name__ == "__main__":
    inspect_mdi_win32()
