"""
GeoPrior & SSAKS Prototype GUI Launcher
=======================================
Launches the interactive self-contained visual prototype in your default web browser.
"""

import os
import sys
import webbrowser
import http.server
import socketserver
import threading

PORT = 8090
HTML_FILE = os.path.join(os.path.dirname(__file__), "geoprior_ssaks_explorer.html")


def main():
    if not os.path.exists(HTML_FILE):
        print(f"Error: Prototype file not found at {HTML_FILE}")
        sys.exit(1)

    print("=" * 70)
    print("🛸 Launching GeoPrior & SSAKS Interactive Prototype GUI")
    print("=" * 70)
    
    # Open local file directly in default web browser
    file_uri = f"file:///{os.path.abspath(HTML_FILE).replace('\\', '/')}"
    print(f"Opening browser at: {file_uri}")
    webbrowser.open(file_uri)
    print("\nPrototype is running! You can interact with the controls in your browser.")
    print("=" * 70)


if __name__ == "__main__":
    main()
