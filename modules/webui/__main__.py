"""
Entry point for running WebUI as a module
Usage: python -m modules.webui
"""
from .app import app

if __name__ == '__main__':
    banner = """
    ======================================================
    SuperBot WebUI v3.0 (Refactored)
    ======================================================

    Dashboard: http://localhost:5000

    Navigation:
      - Dashboard  : /
      - Trade      : /trade
      - Backtest   : /backtest
      - Analiz     : /analiz
      - Ayarlar    : /ayarlar
      - Favorites  : /favorites
      - Portfolio  : /portfolio

    Architecture:
      - Modular design with blueprints
      - Service layer for business logic
      - Clean separation of concerns

    ======================================================
    """
    #print(banner)

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=True
    )
