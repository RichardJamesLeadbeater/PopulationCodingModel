import os
from tkinter import filedialog, Tk

def file_browser(chdir=False):
    root = Tk()  # Create tkinter window

    root.withdraw()  # Hide tkinter window
    root.update()

    directory = filedialog.askdirectory()

    root.update()
    root.destroy()  # Destroy tkinter window

    if chdir:
        os.chdir(directory)

    return directory
