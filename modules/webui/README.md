# SuperBot WebUI

Modern web dashboard for SuperBot trading scanner system.

## Tech Stack

- **Backend**: Flask (Python)
- **Templating**: Jinja2
- **Dynamic Updates**: HTMX
- **Styling**: Custom CSS (no Bootstrap)
- **JavaScript**: Vanilla JS

## Features

- Dashboard with scan statistics and recent opportunities
- Trade page for viewing all opportunities
- Backtest page for historical results
- Analiz page for charts and analysis
- Ayarlar page for settings
- Dark/Light mode toggle
- Offcanvas navigation menu
- Real-time updates with HTMX
- Flash message notifications

## Project Structure

```
modules/webui/
├── app.py                      # Flask application entry point
├── static/
│   ├── css/
│   │   └── main.css           # Custom styles (dark/light theme)
│   └── js/
│       └── main.js            # JavaScript utilities
└── templates/
    ├── base.html              # Base template
    ├── topnav.html            # Navigation header with offcanvas menu
    ├── _flash.html            # Flash message component
    ├── dashboard.html         # Dashboard page
    ├── trade.html             # Trade opportunities page
    ├── backtest.html          # Backtest results page
    ├── analysis.html            # Analysis & charts page
    └── settings.html           # Settings page
```

## Running the WebUI

```bash
cd modules/webui
python app.py
```

Then open your browser to: http://localhost:5000

## API Endpoints

### Dashboard Endpoints

- `GET /` - Dashboard page
- `GET /api/stats` - Get statistics (total opportunities, high/medium scores, avg)
- `GET /api/scan-status` - Get current scan status
- `GET /api/opportunities` - Get recent opportunities list

### Other Pages

- `GET /trade` - Trade opportunities page
- `GET /backtest` - Backtest results page
- `GET /analysis` - Analysis & charts page
- `GET /settings` - Settings page

### Theme Management

- `POST /api/theme/toggle` - Toggle dark/light theme

## Theme System

The WebUI supports dark and light themes:

- Theme is stored in Flask session and localStorage
- Toggle button in top navigation bar
- CSS variables for easy theme customization
- Smooth transitions between themes

## Development

### Adding a New Page

1. Create template in `templates/` directory
2. Add route in `app.py`
3. Update navigation in `topnav.html`

Example:
```python
@app.route('/new-page')
def new_page():
    return render_template('new_page.html',
                         page='new-page',
                         title='New Page')
```

### Creating API Endpoints

API endpoints should return JSON data for HTMX consumption:

```python
@app.route('/api/my-data')
def api_my_data():
    return jsonify({
        'key': 'value'
    })
```

### HTMX Integration

Use HTMX attributes in templates for dynamic updates:

```html
<div hx-get="/api/data"
     hx-trigger="load, every 10s"
     hx-swap="innerHTML">
    Loading...
</div>
```

## TODO

- [ ] Connect API endpoints to real scanner data
- [ ] Implement Trade page with filtering and sorting
- [ ] Implement Backtest page with historical charts
- [ ] Implement Analiz page with interactive charts (Plotly)
- [ ] Implement Ayarlar page with configuration forms
- [ ] Add authentication system
- [ ] Add WebSocket support for real-time updates
- [ ] Add export functionality (CSV, PDF)
- [ ] Add responsive mobile design improvements

## Notes

- The WebUI is designed to be standalone - it doesn't depend on other SuperBot components
- All API endpoints currently return mock data (marked with TODO comments)
- The design follows modern web design patterns without Bootstrap dependency
- Dark theme is the default theme
