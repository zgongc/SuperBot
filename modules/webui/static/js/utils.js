/**
 * SuperBot WebUI - Utility Functions
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Toast notifications, loading spinners, confirmation dialogs
 *
 * Author: SuperBot Team
 * Date: 2025-10-29
 */

// ═══════════════════════════════════════════════════════════════════════════════
// TOAST NOTIFICATIONS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Show toast notification
 * @param {string} message - Message to display
 * @param {string} type - Type: 'success', 'error', 'warning', 'info'
 * @param {number} duration - Duration in ms (default: 4000)
 */
function showToast(message, type = 'info', duration = 4000) {
    // Create toast container if it doesn't exist
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        document.body.appendChild(container);
    }

    // Icons for each type
    const icons = {
        success: '✓',
        error: '✕',
        warning: '⚠',
        info: 'ℹ'
    };

    // Titles for each type
    const titles = {
        success: 'Success',
        error: 'Error',
        warning: 'Warning',
        info: 'Info'
    };

    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <div class="toast-icon">${icons[type] || icons.info}</div>
        <div class="toast-content">
            <div class="toast-title">${titles[type] || titles.info}</div>
            <div class="toast-message">${message}</div>
        </div>
        <button class="toast-close" onclick="this.parentElement.remove()">×</button>
    `;

    // Add to container
    container.appendChild(toast);

    // Auto-remove after duration
    if (duration > 0) {
        setTimeout(() => {
            toast.classList.add('toast-closing');
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }
}

/**
 * Show success toast
 */
function showSuccess(message, duration = 4000) {
    showToast(message, 'success', duration);
}

/**
 * Show error toast
 */
function showError(message, duration = 6000) {
    showToast(message, 'error', duration);
}

/**
 * Show warning toast
 */
function showWarning(message, duration = 5000) {
    showToast(message, 'warning', duration);
}

/**
 * Show info toast
 */
function showInfo(message, duration = 4000) {
    showToast(message, 'info', duration);
}

// ═══════════════════════════════════════════════════════════════════════════════
// LOADING SPINNERS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Show loading overlay
 */
function showLoading() {
    // Remove existing overlay if any
    hideLoading();

    const overlay = document.createElement('div');
    overlay.id = 'loading-overlay';
    overlay.className = 'loading-overlay';
    overlay.innerHTML = '<div class="spinner"></div>';
    document.body.appendChild(overlay);
}

/**
 * Hide loading overlay
 */
function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.remove();
    }
}

/**
 * Set button loading state
 * @param {HTMLElement} button - Button element
 * @param {boolean} loading - Loading state
 */
function setButtonLoading(button, loading) {
    if (loading) {
        button.classList.add('loading');
        button.disabled = true;
    } else {
        button.classList.remove('loading');
        button.disabled = false;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIRMATION DIALOGS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Show confirmation dialog
 * @param {object} options - Configuration options
 * @returns {Promise<boolean>} - Resolves to true if confirmed, false if cancelled
 */
function confirm(options) {
    return new Promise((resolve) => {
        const {
            title = 'Confirm Action',
            message = 'Are you sure?',
            type = 'warning',  // 'warning' or 'danger'
            confirmText = 'Confirm',
            cancelText = 'Cancel'
        } = options;

        // Icons for each type
        const icons = {
            warning: '⚠',
            danger: '⚠'
        };

        // Create dialog element
        const dialog = document.createElement('div');
        dialog.className = `confirm-dialog confirm-dialog-${type} active`;
        dialog.innerHTML = `
            <div class="confirm-dialog-content">
                <div class="confirm-dialog-icon">${icons[type] || icons.warning}</div>
                <div class="confirm-dialog-title">${title}</div>
                <div class="confirm-dialog-message">${message}</div>
                <div class="confirm-dialog-actions">
                    <button class="btn btn-secondary confirm-cancel">${cancelText}</button>
                    <button class="btn btn-${type === 'danger' ? 'danger' : 'primary'} confirm-ok">${confirmText}</button>
                </div>
            </div>
        `;

        // Add to body
        document.body.appendChild(dialog);

        // Handle confirm
        dialog.querySelector('.confirm-ok').addEventListener('click', () => {
            dialog.remove();
            resolve(true);
        });

        // Handle cancel
        dialog.querySelector('.confirm-cancel').addEventListener('click', () => {
            dialog.remove();
            resolve(false);
        });

        // Handle backdrop click
        dialog.addEventListener('click', (e) => {
            if (e.target === dialog) {
                dialog.remove();
                resolve(false);
            }
        });
    });
}

/**
 * Show delete confirmation dialog
 * @param {string} itemName - Name of item being deleted
 * @returns {Promise<boolean>}
 */
async function confirmDelete(itemName) {
    // Turkish translations for common item types
    const translations = {
        'Exchange Account': {
            title: 'Delete Stock Account?',
            message: 'This operation cannot be undone. The portfolio and all positions associated with this account will be deleted. Are you sure you want to proceed?'
        },
        'Symbol': {
            title: 'Delete symbol?',
            message: 'This operation cannot be undone. Are you sure you want to delete this symbol?'
        }
    };

    const translation = translations[itemName] || {
        title: itemName + ' Sil?',
        message: 'This operation cannot be undone. ' + itemName + ' Are you sure you want to delete?'
    };

    return await confirm({
        title: translation.title,
        message: translation.message,
        type: 'danger',
        confirmText: 'Sil',
        cancelText: 'Cancel'
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
// API HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Fetch wrapper with loading and error handling
 * @param {string} url - API endpoint
 * @param {object} options - Fetch options
 * @param {object} uiOptions - UI options (showLoading, showError)
 * @returns {Promise<object>} - API response data
 */
async function fetchAPI(url, options = {}, uiOptions = {}) {
    const {
        showLoadingOverlay = false,
        showErrorToast = true,
        showSuccessToast = false,
        successMessage = 'Operation successful'
    } = uiOptions;

    try {
        // Show loading
        if (showLoadingOverlay) {
            showLoading();
        }

        // Make request
        const response = await fetch(url, options);
        const data = await response.json();

        // Hide loading
        if (showLoadingOverlay) {
            hideLoading();
        }

        // Handle error response
        if (data.status !== 'success') {
            if (showErrorToast) {
                showError(data.message || 'Operation failed');
            }
            throw new Error(data.message || 'Operation failed');
        }

        // Show success toast if requested
        if (showSuccessToast) {
            showSuccess(successMessage);
        }

        return data.data;

    } catch (error) {
        // Hide loading
        if (showLoadingOverlay) {
            hideLoading();
        }

        // Show error toast
        if (showErrorToast) {
            showError(error.message || 'Network error');
        }

        throw error;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXPORT FOR MODULE USAGE (if needed)
// ═══════════════════════════════════════════════════════════════════════════════

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        showToast,
        showSuccess,
        showError,
        showWarning,
        showInfo,
        showLoading,
        hideLoading,
        setButtonLoading,
        confirm,
        confirmDelete,
        fetchAPI
    };
}

/**
 * Loading Spinner Manager
 */
const LoadingSpinner = {
    overlay: null,

    /**
     * Show full-page loading overlay
     */
    show() {
        if (this.overlay) return;

        this.overlay = document.createElement('div');
        this.overlay.className = 'loading-spinner-overlay';
        this.overlay.innerHTML = '<div class="spinner"></div>';
        document.body.appendChild(this.overlay);
    },

    /**
     * Hide loading overlay
     */
    hide() {
        if (this.overlay) {
            this.overlay.remove();
            this.overlay = null;
        }
    },

    /**
     * Add loading state to button
     * @param {HTMLButtonElement} button
     */
    button(button) {
        if (!button) return;
        button.classList.add('loading');
        button.disabled = true;
    },

    /**
     * Remove loading state from button
     * @param {HTMLButtonElement} button
     */
    buttonDone(button) {
        if (!button) return;
        button.classList.remove('loading');
        button.disabled = false;
    }
};

// Make it globally available
window.LoadingSpinner = LoadingSpinner;
