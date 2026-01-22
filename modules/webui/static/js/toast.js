/**
 * Toast Notification System
 *
 * Usage:
 * showToast('Success!', 'Operation completed successfully', 'success');
 * showToast('Error!', 'Something went wrong', 'error', 5000);
 * showToast('Info', 'Check this out', 'info', 0); // no auto-dismiss
 *
 * With actions:
 * showToast('Alert Triggered!', 'Pattern detected on BTCUSDT', 'success', 5000, [
 *   { label: 'View Analysis', onClick: () => window.open('/analysis', '_blank'), primary: true },
 *   { label: 'Dismiss', onClick: null }
 * ]);
 */

const ToastManager = {
    container: null,
    toasts: [],

    init() {
        this.container = document.getElementById('toast-container');
        if (!this.container) {
            console.error('Toast container not found');
        }
    },

    show(title, message, type = 'info', duration = 5000, actions = []) {
        if (!this.container) this.init();

        const toastId = 'toast-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);

        // Create toast element
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.id = toastId;

        // Icon based on type
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️',
            alert: '🔔'
        };

        // Build HTML
        let html = `
            <div class="toast-icon">${icons[type] || icons.info}</div>
            <div class="toast-content">
                <div class="toast-title">${title}</div>
                ${message ? `<div class="toast-message">${message}</div>` : ''}
                ${actions.length > 0 ? this.buildActions(actions, toastId) : ''}
            </div>
            <button class="toast-close" onclick="ToastManager.dismiss('${toastId}')">&times;</button>
        `;

        toast.innerHTML = html;

        // Add to container
        this.container.appendChild(toast);

        // Store toast with actions
        this.toasts.push({ id: toastId, element: toast, actions: actions });

        // Auto dismiss
        if (duration > 0) {
            setTimeout(() => {
                this.dismiss(toastId);
            }, duration);
        }

        return toastId;
    },

    buildActions(actions, toastId) {
        const actionsHtml = actions.map((action, index) => {
            // Support both old API (text, callback, className) and new API (label, onClick, primary)
            const label = action.text || action.label || 'OK';
            const className = action.className || (action.primary ? 'primary' : '');
            const hasCallback = action.callback || action.onClick;

            const onclick = hasCallback
                ? `ToastManager.handleAction('${toastId}', ${index})`
                : `ToastManager.dismiss('${toastId}')`;

            return `<button class="toast-action-btn ${className}" onclick="${onclick}">${label}</button>`;
        }).join('');

        return `<div class="toast-actions">${actionsHtml}</div>`;
    },

    handleAction(toastId, actionIndex) {
        const toastData = this.toasts.find(t => t.id === toastId);
        if (toastData && toastData.actions && toastData.actions[actionIndex]) {
            const action = toastData.actions[actionIndex];
            const callback = action.callback || action.onClick;
            if (callback) {
                callback();
            }
        }
        this.dismiss(toastId);
    },

    dismiss(toastId) {
        const toastIndex = this.toasts.findIndex(t => t.id === toastId);
        if (toastIndex === -1) return;

        const toast = this.toasts[toastIndex];
        toast.element.classList.add('removing');

        setTimeout(() => {
            if (toast.element.parentNode) {
                toast.element.parentNode.removeChild(toast.element);
            }
            this.toasts.splice(toastIndex, 1);
        }, 300);
    },

    dismissAll() {
        [...this.toasts].forEach(toast => {
            this.dismiss(toast.id);
        });
    }
};

// Global function for easy access
function showToast(title, message, type = 'info', duration = 5000, actions = []) {
    return ToastManager.show(title, message, type, duration, actions);
}

// Notification Polling System
const NotificationPoller = {
    intervalId: null,
    pollInterval: 5000, // 5 seconds

    start() {
        // Initial poll
        this.poll();

        // Set up recurring poll
        this.intervalId = setInterval(() => {
            this.poll();
        }, this.pollInterval);

        console.log('Notification polling started');
    },

    stop() {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
            console.log('Notification polling stopped');
        }
    },

    async poll() {
        try {
            const response = await fetch('/api/notifications/pending');
            const data = await response.json();

            if (data.status === 'success' && data.data.notifications.length > 0) {
                data.data.notifications.forEach(notification => {
                    this.displayNotification(notification);
                });
            }
        } catch (error) {
            console.error('Notification polling error:', error);
        }
    },

    displayNotification(notification) {
        // Build actions if analysis_result_id present
        const actions = [];
        if (notification.analysis_result_id) {
            actions.push({
                label: 'View Analysis',
                onClick: () => {
                    window.open(`/analysis?result_id=${notification.analysis_result_id}`, '_blank');
                },
                primary: true
            });
        }

        // Show toast
        ToastManager.show(
            notification.title,
            notification.message,
            notification.type || 'alert',
            notification.duration || 7000,
            actions
        );
    }
};

// Notification Bell Manager
const NotificationBellManager = {
    notifications: [],
    unreadCount: 0,

    async loadNotifications() {
        try {
            // Fetch from both history (database) and pending (memory)
            const [historyResponse, unreadCountResponse] = await Promise.all([
                fetch('/api/notifications/history?limit=20'),
                fetch('/api/notifications/unread-count')
            ]);

            const historyData = await historyResponse.json();
            const unreadData = await unreadCountResponse.json();

            if (historyData.status === 'success') {
                this.notifications = historyData.data.notifications || [];
            }

            // Use unread count from database
            if (unreadData.status === 'success') {
                this.unreadCount = unreadData.data.count || 0;
            }

            this.updateBell();
            this.renderNotificationList();
        } catch (error) {
            console.error('Failed to load notifications:', error);
        }
    },

    addNotification(notification) {
        // Add to beginning of list
        this.notifications.unshift(notification);

        // Keep only last 50
        if (this.notifications.length > 50) {
            this.notifications = this.notifications.slice(0, 50);
        }

        this.updateBell();
        this.renderNotificationList();
    },

    async updateBell() {
        // Fetch latest unread count from server
        try {
            const response = await fetch('/api/notifications/unread-count');
            const data = await response.json();
            if (data.status === 'success') {
                this.unreadCount = data.data.count || 0;
            }
        } catch (error) {
            console.error('Failed to fetch unread count:', error);
            // Fallback to local count
            this.unreadCount = this.notifications.filter(n => !n.is_read).length;
        }

        // Update badge
        const badge = document.getElementById('notification-badge');
        if (badge) {
            if (this.unreadCount > 0) {
                badge.textContent = this.unreadCount > 99 ? '99+' : this.unreadCount;
                badge.style.display = 'block';
            } else {
                badge.style.display = 'none';
            }
        }

        // Update mark all button
        const markAllBtn = document.querySelector('.mark-all-read-btn');
        if (markAllBtn) {
            markAllBtn.style.display = this.unreadCount > 0 ? 'block' : 'none';
        }
    },

    renderNotificationList() {
        const container = document.getElementById('notification-list');
        if (!container) return;

        if (this.notifications.length === 0) {
            container.innerHTML = '<div class="notification-empty">No notifications yet</div>';
            return;
        }

        const html = this.notifications.map(n => this.renderNotificationItem(n)).join('');
        container.innerHTML = html;
    },

    renderNotificationItem(notification) {
        const unreadClass = notification.is_read ? '' : 'unread';
        const icon = this.getIcon(notification.type);
        const timeAgo = this.getTimeAgo(notification.created_at || notification.timestamp);

        return `
            <div class="notification-item ${unreadClass}" onclick="NotificationBellManager.handleClick(${notification.id || notification.notification_id})">
                <div class="notification-item-icon">${icon}</div>
                <div class="notification-item-content">
                    <div class="notification-item-title">${notification.title}</div>
                    <div class="notification-item-message">${notification.message}</div>
                    <div class="notification-item-time">${timeAgo}</div>
                </div>
            </div>
        `;
    },

    getIcon(type) {
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️',
            alert: '🔔'
        };
        return icons[type] || icons.info;
    },

    getTimeAgo(timestamp) {
        const now = new Date();
        const time = new Date(timestamp);
        const diffMs = now - time;
        const diffMins = Math.floor(diffMs / 60000);

        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins}m ago`;

        const diffHours = Math.floor(diffMins / 60);
        if (diffHours < 24) return `${diffHours}h ago`;

        const diffDays = Math.floor(diffHours / 24);
        if (diffDays < 7) return `${diffDays}d ago`;

        return time.toLocaleDateString();
    },

    async handleClick(notificationId) {
        // Mark as read
        try {
            await fetch(`/api/notifications/${notificationId}/read`, {
                method: 'POST'
            });

            // Update local state
            const notification = this.notifications.find(n =>
                (n.id || n.notification_id) === notificationId
            );
            if (notification) {
                notification.is_read = true;
                this.updateBell();
                this.renderNotificationList();
            }

            // If has analysis_result_id, open it
            if (notification && notification.analysis_result_id) {
                window.open(`/analysis?result_id=${notification.analysis_result_id}`, '_blank');
            }
        } catch (error) {
            console.error('Failed to mark as read:', error);
        }
    }
};

async function markAllAsRead() {
    try {
        await fetch('/api/notifications/mark-all-read', {
            method: 'POST'
        });

        // Update local state
        NotificationBellManager.notifications.forEach(n => n.is_read = true);
        NotificationBellManager.updateBell();
        NotificationBellManager.renderNotificationList();
    } catch (error) {
        console.error('Failed to mark all as read:', error);
    }
}

// Initialize on DOM load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        ToastManager.init();
        NotificationPoller.start();
        NotificationBellManager.loadNotifications();
    });
} else {
    ToastManager.init();
    NotificationPoller.start();
    NotificationBellManager.loadNotifications();
}
