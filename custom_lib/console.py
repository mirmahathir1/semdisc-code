import os
import glob
import time
from custom_lib import utils
import json
import sys
import subprocess
from collections import defaultdict
import pandas as pd
from typing import Any

from tqdm import tqdm

class progressbar(tqdm):
    """
    A tqdm variant that *requires* a human-readable note.

    Parameters
    ----------
    iterable : iterable
        Anything you would normally wrap with tqdm.
    note : str, required
        A short description of what this progress bar is for.
    **kwargs
        Any standard tqdm keyword arguments (total, leave, etc.).
    """

    def __init__(self, iterable, *, note, **kwargs):
        # --- enforce the note ------------------------------------------------
        if not note or not note.strip():
            raise ValueError(
                "You must supply a non-empty 'note' argument explaining the purpose "
                "of this progress bar."
            )

        # Re-use the note as the bar’s description unless user provided one
        kwargs.setdefault("desc", note)

        super().__init__(iterable, **kwargs)

        # Store for later introspection / logging
        self.note = note


def get_console_file_path():
    return f"./console"

def get_file_display_text_editor_path_txt():
    return f"{get_console_file_path()}/file_display_text_editor.txt"

def json_print(object):
    log(json.dumps(obj=object, indent=4))

def open_text_file_in_editor(filepath):
    try:
        if sys.platform.startswith("win"):
            os.startfile(filepath)
        elif sys.platform.startswith("darwin"):  # macOS
            result = subprocess.run(["open", filepath], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        else:  # Linux and other Unix-like OS
            result = subprocess.run(["xdg-open", filepath], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        return result.stderr  # Capturing stderr messages if needed
    except Exception as e:
        return f"Error opening file: {e}"

def file_display_text_editor(path):
    content = utils.file_load(path=path)
    with open(get_file_display_text_editor_path_txt(), "w") as file:
        file.write(json.dumps(obj=content, indent=4))
    open_text_file_in_editor(get_file_display_text_editor_path_txt())
    
def debug(*args, sep=' ', end='\n', verbose=True):
    print(*args, sep=sep, end=end)
    return
    formatted_time = ""
    if verbose == True:
        formatted_time = f"{utils.get_formatted_time()}: DEBUG: "
    
    print(formatted_time + sep.join(str(arg) for arg in args), end=end)
    

def log(*args, sep=' ', end='\n', standard_output=True, write_to_file=False, pretty=True, depth=None):
    print(*args, sep=sep, end=end)
    return
    hash_value = utils.get_random_hash()
    formatted_time = f"{utils.get_formatted_time()}: "
    if write_to_file:
        with open(get_console_file_path() + '/' + hash_value, 'w') as file:
            if pretty == True:
                file.write(formatted_time + sep.join(json_to_string(data=arg, depth=depth) for arg in args) + end)
            else:
                file.write(formatted_time + sep.join(str(arg) for arg in args) + end)

    if standard_output:
        if pretty == True:
            print(formatted_time + sep.join(json_to_string(data=arg, depth=depth) for arg in args), end=end)
        else:
            print(formatted_time + sep.join(str(arg) for arg in args), end=end)

    return hash_value

def reduce_data(data, depth=None, current_depth=0):
    """
    Modifies the data object in-place by replacing elements deeper than 'depth' with "#reduced".
    
    Args:
        data: The JSON object to modify
        depth: The depth level after which to replace with "#reduced" (None means no reduction)
        current_depth: (Internal) Tracks current nesting depth
        
    Returns:
        The modified data object
    """
    is_list = isinstance(data, list)
    is_dict = isinstance(data, dict)

    if not is_list and not is_dict:
        return data

    if depth is not None and current_depth > depth:
        return f"#reduced_{type(data)}"
    
    if is_dict:
        for key in list(data.keys()):
            data[key] = reduce_data(data[key], depth, current_depth + 1)
    elif is_list:
        for i in range(len(data)):
            data[i] = reduce_data(data[i], depth, current_depth + 1)
    
    return data

def json_to_string(data, depth=None):
    return json.dumps(reduce_data(data=json.loads(custom_dumps(data)), depth=depth), indent=4)

if __name__ == '__main__':
    os.makedirs(get_console_file_path(), exist_ok=True)
    print("console server is running")
    print("_"*40)
    pbar = None
    last_completed = 0
    while True:
        all_files = glob.glob(os.path.join(get_console_file_path(), '*'))
        if not all_files:
            pass
        else:
            all_files.sort(key=os.path.getmtime)
            for file in all_files:
                with open(file, 'r') as file_object:
                    content = file_object.read()
                    if 'TQDM' in content:
                        [completed, total] = content.split(' ')[1].split('/')
                        completed = int(completed)
                        if pbar is None:
                            if 'None' in total:
                                total = None
                            else:
                                total = int(total)
                            pbar = tqdm(total=total)
                        pbar.update(completed-last_completed)
                        last_completed = completed
                        
                    else:
                        if not pbar is None:
                            pbar.close()
                            pbar = None
                            last_completed = 0
                        print(content, end='')
                os.remove(file)
        time.sleep(1)

def custom_dumps(obj: Any, **kwargs) -> str:
    """Custom JSON dumps that supports sets and DataFrames.
    
    Sets are converted to {"type": "set", "values": [list of values]}
    DataFrames are converted to {"type": "dataframe", "values": [list of columns]}
    """
    def default_encoder(o: Any) -> Any:
        if isinstance(o, set):
            return {"type": str(type(o)), "values": list(o)}
        elif isinstance(o, pd.DataFrame):
            return {"type": str(type(o)), "values": str([col.tolist() for col in o.values.T])+f", dataframe shape: {o.shape}"}
        raise TypeError(f"Object of type {type(o)} is not JSON serializable by custom dumps")
    
    return json.dumps(obj, default=default_encoder, **kwargs)

import sys
import time
import itertools
import threading


class Spinner:
    """
    spinner = Spinner("Crunching numbers…")
    do_something_expensive()
    spinner.stop()                    # prints: ✓ Crunching numbers… (then newline)

    # If you prefer to erase the line completely:
    # spinner.stop(persist=False)
    """

    # ───────────────────────────── class-level stdout lock ─────────────────────
    _screen_lock = threading.Lock()

    def __init__(self, message: str = "Working…", interval: float = 0.1) -> None:
        self._message  = message
        self._interval = interval
        self._running  = True
        self._thread   = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    # ---------------------------------------------------------------- private -
    def _spin(self) -> None:
        frames = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")      # fallback: "|/-\\"
        while self._running:
            with Spinner._screen_lock:
                sys.stdout.write(f"\r{next(frames)} {self._message}")
                sys.stdout.flush()
            time.sleep(self._interval)

    # ---------------------------------------------------------------- public --
    def stop(self, persist: bool = True) -> None:
        """
        Stop the spinner.
        If *persist* is True, show a static ✓ message; otherwise clear the line.
        Always ends with a newline so the next output starts on its own line.
        """
        if not self._running:
            return
        self._running = False
        self._thread.join()

        with Spinner._screen_lock:
            if persist:
                sys.stdout.write(f"\r✓ {self._message}\n")
            else:
                sys.stdout.write("\r" + " " * (len(self._message) + 2) + "\r\n")
            sys.stdout.flush()
