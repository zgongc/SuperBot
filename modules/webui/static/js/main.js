/**
 * SuperBot WebUI - Main JavaScript
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Features:
 *   - Offcanvas menu toggle
 *   - Dark/Light theme toggle
 *   - Flash message auto-dismiss
 *   - HTMX integration helpers
 *
 * Author: SuperBot Team
 * Date: 2025-10-27
 */

(function() {
    'use strict';

    // ═══════════════════════════════════════════════════════════════════════════════
    // THEME MANAGEMENT
    // ═══════════════════════════════════════════════════════════════════════════════

    /**
     * Initialize theme from localStorage or default to dark
     */
    function initTheme() {
        const savedTheme = localStorage.getItem('theme') || 'dark';
        document.documentElement.setAttribute('data-theme', savedTheme);
        updateThemeIcon(savedTheme);
    }

    /**
     * Toggle between dark and light theme
     */
    window.toggleTheme = function() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

        // Update DOM
        document.documentElement.setAttribute('data-theme', newTheme);

        // Save to localStorage
        localStorage.setItem('theme', newTheme);

        // Update icon
        updateThemeIcon(newTheme);

        // Notify server (for session storage)
        fetch('/api/theme/toggle', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ theme: newTheme })
        }).catch(err => console.error('Failed to save theme:', err));
    };

    /**
     * Update theme toggle icon
     */
    function updateThemeIcon(theme) {
        const themeToggle = document.getElementById('theme-toggle');
        if (!themeToggle) return;

        const icon = themeToggle.querySelector('.theme-icon');
        if (!icon) return;

        if (theme === 'dark') {
            // Show sun icon (for switching to light)
            icon.innerHTML = `
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="5"/>
                    <line x1="12" y1="1" x2="12" y2="3"/>
                    <line x1="12" y1="21" x2="12" y2="23"/>
                    <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
                    <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
                    <line x1="1" y1="12" x2="3" y2="12"/>
                    <line x1="21" y1="12" x2="23" y2="12"/>
                    <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
                    <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
                </svg>
            `;
        } else {
            // Show moon icon (for switching to dark)
            icon.innerHTML = `
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
                </svg>
            `;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // DROPDOWN MENUS
    // ═══════════════════════════════════════════════════════════════════════════════

    /**
     * Toggle dropdown menu
     */
    window.toggleDropdown = function(button) {
        const dropdown = button.closest('.nav-dropdown');
        if (!dropdown) return;

        // Close other dropdowns
        document.querySelectorAll('.nav-dropdown.show').forEach(function(other) {
            if (other !== dropdown) {
                other.classList.remove('show');
            }
        });

        // Toggle current dropdown
        dropdown.classList.toggle('show');
    };

    /**
     * Close all dropdowns when clicking outside
     */
    function initDropdownClickOutside() {
        document.addEventListener('click', function(e) {
            // If click is not inside any dropdown
            if (!e.target.closest('.nav-dropdown')) {
                document.querySelectorAll('.nav-dropdown.show').forEach(function(dropdown) {
                    dropdown.classList.remove('show');
                });
            }
        });
    }

    /**
     * Close dropdowns on ESC key
     */
    function initDropdownKeyboard() {
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                document.querySelectorAll('.nav-dropdown.show').forEach(function(dropdown) {
                    dropdown.classList.remove('show');
                });
            }
        });
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // OFFCANVAS MENU
    // ═══════════════════════════════════════════════════════════════════════════════

    /**
     * Toggle offcanvas menu
     */
    window.toggleOffcanvas = function() {
        const offcanvas = document.getElementById('offcanvas-menu');
        const backdrop = document.getElementById('offcanvas-backdrop');

        if (!offcanvas || !backdrop) return;

        const isVisible = offcanvas.classList.contains('show');

        if (isVisible) {
            closeOffcanvas();
        } else {
            openOffcanvas();
        }
    };

    /**
     * Open offcanvas menu
     */
    function openOffcanvas() {
        const offcanvas = document.getElementById('offcanvas-menu');
        const backdrop = document.getElementById('offcanvas-backdrop');

        if (!offcanvas || !backdrop) return;

        offcanvas.classList.add('show');
        backdrop.classList.add('show');
        document.body.style.overflow = 'hidden';
    }

    /**
     * Close offcanvas menu
     */
    window.closeOffcanvas = function() {
        const offcanvas = document.getElementById('offcanvas-menu');
        const backdrop = document.getElementById('offcanvas-backdrop');

        if (!offcanvas || !backdrop) return;

        offcanvas.classList.remove('show');
        backdrop.classList.remove('show');
        document.body.style.overflow = '';
    };

    /**
     * Close offcanvas when clicking backdrop
     */
    function initOffcanvasBackdrop() {
        const backdrop = document.getElementById('offcanvas-backdrop');
        if (backdrop) {
            backdrop.addEventListener('click', closeOffcanvas);
        }
    }

    /**
     * Close offcanvas on ESC key
     */
    function initOffcanvasKeyboard() {
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                closeOffcanvas();
            }
        });
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // FLASH MESSAGES
    // ═══════════════════════════════════════════════════════════════════════════════

    /**
     * Auto-dismiss flash messages after 5 seconds
     */
    function initFlashMessages() {
        const flashMessages = document.querySelectorAll('.flash-message');

        flashMessages.forEach(function(message) {
            // Auto-dismiss after 5 seconds
            setTimeout(function() {
                dismissFlashMessage(message);
            }, 5000);

            // Manual dismiss on close button click
            const closeBtn = message.querySelector('.flash-close');
            if (closeBtn) {
                closeBtn.addEventListener('click', function() {
                    dismissFlashMessage(message);
                });
            }
        });
    }

    /**
     * Dismiss a flash message with animation
     */
    function dismissFlashMessage(message) {
        message.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(function() {
            message.remove();
        }, 300);
    }

    /**
     * Add slideOut animation
     */
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideOutRight {
            from {
                transform: translateX(0);
                opacity: 1;
            }
            to {
                transform: translateX(100%);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(style);

    // ═══════════════════════════════════════════════════════════════════════════════
    // HTMX INTEGRATION
    // ═══════════════════════════════════════════════════════════════════════════════

    /**
     * HTMX event listeners for better UX
     */
    function initHTMX() {
        // Show loading state
        document.body.addEventListener('htmx:beforeRequest', function(evt) {
            const target = evt.detail.target;
            if (target) {
                target.style.opacity = '0.6';
            }
        });

        // Remove loading state
        document.body.addEventListener('htmx:afterRequest', function(evt) {
            const target = evt.detail.target;
            if (target) {
                target.style.opacity = '1';
            }
        });

        // Error handling
        document.body.addEventListener('htmx:responseError', function(evt) {
            console.error('HTMX Error:', evt.detail);
            showNotification('Error loading data. Please try again.', 'error');
        });
    }

    /**
     * Show notification (helper for dynamic notifications)
     */
    function showNotification(message, type = 'info') {
        const container = document.querySelector('.flash-container');
        if (!container) return;

        const icons = {
            success: `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="20 6 9 17 4 12"/>
            </svg>`,
            error: `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"/>
                <line x1="15" y1="9" x2="9" y2="15"/>
                <line x1="9" y1="9" x2="15" y2="15"/>
            </svg>`,
            warning: `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                <line x1="12" y1="9" x2="12" y2="13"/>
                <line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>`,
            info: `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"/>
                <line x1="12" y1="16" x2="12" y2="12"/>
                <line x1="12" y1="8" x2="12.01" y2="8"/>
            </svg>`
        };

        const flash = document.createElement('div');
        flash.className = `flash-message ${type}`;
        flash.innerHTML = `
            <span class="flash-icon">${icons[type] || icons.info}</span>
            <div class="flash-content">${message}</div>
            <button class="flash-close" onclick="this.parentElement.remove()">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="18" y1="6" x2="6" y2="18"/>
                    <line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
            </button>
        `;

        container.appendChild(flash);

        // Auto-dismiss after 5 seconds
        setTimeout(function() {
            dismissFlashMessage(flash);
        }, 5000);
    }

    // Make showNotification globally accessible
    window.showNotification = showNotification;

    // ═══════════════════════════════════════════════════════════════════════════════
    // INITIALIZATION
    // ═══════════════════════════════════════════════════════════════════════════════

    /**
     * Initialize all components when DOM is ready
     */
    function init() {
        initTheme();
        initDropdownClickOutside();
        initDropdownKeyboard();
        initOffcanvasBackdrop();
        initOffcanvasKeyboard();
        initFlashMessages();
        initHTMX();

        console.log('SuperBot WebUI initialized');
    }

    // Run initialization
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
