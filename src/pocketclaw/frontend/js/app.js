/**
 * PocketPaw Main Application
 * Alpine.js component for the dashboard
 *
 * Changes (2026-02-04):
 * - Fixed Memory UI: Added memoryLoading, memorySearch, memoryStats state
 * - Added filteredMemories getter for memory search/display
 * - Added getMemoryTypeClass(), formatDate(), deleteMemory() methods
 * - Added updateMemoryStats() to compute memory statistics
 * - Fixed: Don't log streaming chunks to terminal (prevents word-by-word flood)
 * - Added Telegram setup in "Take Paw With You" modal
 * - Added remoteTab state for QR/Telegram tabs
 * - Added telegramStatus, telegramForm, telegramLoading state
 * - Added getTelegramStatus(), startTelegramPairing(), startTelegramPolling() methods
 *
 * Changes (2026-02-02):
 * - Fixed auto-scroll: Added scrollToBottom() method with requestAnimationFrame
 * - Streaming scroll: Now scrolls during streaming content updates
 * - Simplified: Removed 2-layer mode
 * - Added: PocketPaw Native backend (brain + OI hands)
 * - Updated: Default backend is now claude_agent_sdk (recommended)
 * - Added: getBackendDescription() for settings UI
 */

function app() {
    return {
        // View state
        view: 'chat',
        showSettings: false,
        showScreenshot: false,
        showFileBrowser: false,
        showReminders: false,
        screenshotSrc: '',

        // Reminders state
        reminders: [],
        reminderInput: '',
        reminderLoading: false,

        // Intentions state
        showIntentions: false,
        intentions: [],
        intentionLoading: false,
        intentionForm: {
            name: '',
            prompt: '',
            schedulePreset: '',
            customCron: '',
            includeSystemStatus: false
        },

        // Skills state
        showSkills: false,
        skills: [],
        skillsLoading: false,

        // Remote Access state
        showRemote: false,
        remoteTab: 'qr',  // 'qr' or 'telegram'
        remoteStatus: { active: false, url: '', installed: false },
        tunnelLoading: false,

        // Telegram state
        telegramStatus: { configured: false, user_id: null },
        telegramForm: { botToken: '', qrCode: '', error: '' },
        telegramLoading: false,
        telegramPollInterval: null,

        // Transparency Panel state
        showIdentity: false,
        identityLoading: false,
        identityData: {},
        
        showMemory: false,
        memoryTab: 'sessions',  // 'sessions', 'long_term'
        sessionsList: [],       // List of all sessions
        sessionMemory: [],      // Current session history
        selectedSession: null,  // Currently selected session ID
        longTermMemory: [],
        memoryLoading: false,
        memorySearch: '',
        memoryStats: { total_memories: 0, active_context: 0, archived: 0, user_interactions: 0 },
        
        showAudit: false,
        auditLoading: false,
        auditLogs: [],
        
        activityLog: [],
        sessionId: null,

        // File browser state

        // File browser state
        filePath: '~',
        files: [],
        fileLoading: false,
        fileError: null,
        
        // Agent state
        agentActive: true,
        isStreaming: false,
        streamingContent: '',
        streamingMessageId: null,
        hasShownWelcome: false,
        
        // Messages
        messages: [],
        logs: [],
        inputText: '',
        
        // System status
        status: {
            cpu: '—',
            ram: '—',
            disk: '—',
            battery: '—'
        },
        
        // Settings
        settings: {
            agentBackend: 'claude_agent_sdk',  // Default: Claude Agent SDK (recommended)
            llmProvider: 'auto',
            anthropicModel: 'claude-sonnet-4-5-20250929',
            bypassPermissions: false
        },
        
        // API Keys (not persisted client-side, but we track if saved on server)
        apiKeys: {
            anthropic: '',
            openai: ''
        },
        hasAnthropicKey: false,
        hasOpenaiKey: false,

        /**
         * Initialize the app
         */
        init() {
            this.log('PocketPaw Dashboard initialized', 'info');

            // Handle Auth Token (URL capture)
            const urlParams = new URLSearchParams(window.location.search);
            const token = urlParams.get('token');
            if (token) {
                localStorage.setItem('pocketpaw_token', token);
                // Clean URL
                window.history.replaceState({}, document.title, window.location.pathname);
                this.log('Auth token captured and stored', 'success');
            }

            // --- OVERRIDE FETCH FOR AUTH ---
            const originalFetch = window.fetch;
            window.fetch = async (url, options = {}) => {
                const storedToken = localStorage.getItem('pocketpaw_token');
                
                // Skip auth for static or external
                if (url.toString().startsWith('/api') || url.toString().startsWith('/')) {
                     options.headers = options.headers || {};
                     if (storedToken) {
                         options.headers['Authorization'] = `Bearer ${storedToken}`;
                     }
                }
                
                const response = await originalFetch(url, options);
                
                if (response.status === 401 || response.status === 403) {
                     this.showToast('Session expired. Please re-authenticate.', 'error');
                     // Optionally redirect to login page (if we had one)
                }
                
                return response;
            };

            // Register event handlers first
            this.setupSocketHandlers();

            // Connect WebSocket (singleton - will only connect once)
            socket.connect();

            // Start status polling (low frequency)
            this.startStatusPolling();

            // Refresh Lucide icons after initial render
            this.$nextTick(() => {
                if (window.refreshIcons) window.refreshIcons();
            });
        },

        /**
         * Set up WebSocket event handlers
         */
        setupSocketHandlers() {
            // Clear existing handlers to prevent duplicates
            socket.clearHandlers();
            
            const onConnected = () => {
                this.log('Connected to PocketPaw Engine', 'success');
                // Fetch initial status and settings
                socket.runTool('status');
                socket.send('get_settings');
                
                // Fetch initial data for sidebar badges
                socket.send('get_reminders');
                socket.send('get_intentions');
                socket.send('get_skills');

                // Auto-activate agent mode
                if (this.agentActive) {
                    socket.toggleAgent(true);
                    this.log('Agent Mode auto-activated', 'info');
                }
            };
            
            socket.on('connected', onConnected);
            
            // If already connected, trigger manually
            if (socket.isConnected) {
                onConnected();
            }
            
            socket.on('disconnected', () => {
                this.log('Disconnected from server', 'error');
            });
            
            socket.on('message', (data) => this.handleMessage(data));
            socket.on('notification', (data) => this.handleNotification(data));
            socket.on('status', (data) => this.handleStatus(data));
            socket.on('screenshot', (data) => this.handleScreenshot(data));
            socket.on('code', (data) => this.handleCode(data));
            socket.on('error', (data) => this.handleError(data));
            socket.on('stream_start', () => this.startStreaming());
            socket.on('stream_end', () => this.endStreaming());
            socket.on('files', (data) => this.handleFiles(data));
            socket.on('settings', (data) => this.handleSettings(data));

            // Reminder handlers
            socket.on('reminders', (data) => this.handleReminders(data));
            socket.on('reminder_added', (data) => this.handleReminderAdded(data));
            socket.on('reminder_deleted', (data) => this.handleReminderDeleted(data));
            socket.on('reminder', (data) => this.handleReminderTriggered(data));

            // Intention handlers
            socket.on('intentions', (data) => this.handleIntentions(data));
            socket.on('intention_created', (data) => this.handleIntentionCreated(data));
            socket.on('intention_updated', (data) => this.handleIntentionUpdated(data));
            socket.on('intention_toggled', (data) => this.handleIntentionToggled(data));
            socket.on('intention_deleted', (data) => this.handleIntentionDeleted(data));
            socket.on('intention_event', (data) => this.handleIntentionEvent(data));

            // Skills handlers
            socket.on('skills', (data) => this.handleSkills(data));
            socket.on('skill_started', (data) => this.handleSkillStarted(data));
            socket.on('skill_completed', (data) => this.handleSkillCompleted(data));
            socket.on('skill_received', (data) => console.log('Skill received', data));
            socket.on('skill_error', (data) => this.handleSkillError(data));

            // Transparency handlers
            socket.on('connection_info', (data) => this.handleConnectionInfo(data));
            socket.on('system_event', (data) => this.handleSystemEvent(data));

        },

        /**
         * Handle notification
         */
        handleNotification(data) {
            const content = data.content || '';
            
            // Skip duplicate connection messages
            if (content.includes('Connected to PocketPaw') && this.hasShownWelcome) {
                return;
            }
            if (content.includes('Connected to PocketPaw')) {
                this.hasShownWelcome = true;
            }
            
            this.showToast(content, 'info');
            this.log(content, 'info');
        },

        /**
         * Handle incoming message
         */
        handleMessage(data) {
            const content = data.content || '';

            // Check if it's a status update (don't show in chat)
            if (content.includes('System Status') || content.includes('🧠 CPU:')) {
                this.status = Tools.parseStatus(content);
                return;
            }

            // Handle streaming vs complete messages
            if (this.isStreaming) {
                this.streamingContent += content;
                // Scroll during streaming to follow new content
                this.$nextTick(() => this.scrollToBottom());
                // Don't log streaming chunks - they flood the terminal
            } else {
                this.addMessage('assistant', content);
                // Only log complete messages (not streaming chunks)
                if (content.trim()) {
                    this.log(content.substring(0, 100) + (content.length > 100 ? '...' : ''), 'info');
                }
            }
        },

        /**
         * Handle status updates
         */
        handleStatus(data) {
            if (data.content) {
                this.status = Tools.parseStatus(data.content);
            }
        },

        /**
         * Handle settings from server (on connect)
         */
        handleSettings(data) {
            if (data.content) {
                const serverSettings = data.content;
                // Apply server settings to frontend state
                if (serverSettings.agentBackend) {
                    this.settings.agentBackend = serverSettings.agentBackend;
                }
                if (serverSettings.llmProvider) {
                    this.settings.llmProvider = serverSettings.llmProvider;
                }
                if (serverSettings.anthropicModel) {
                    this.settings.anthropicModel = serverSettings.anthropicModel;
                }
                if (serverSettings.bypassPermissions !== undefined) {
                    this.settings.bypassPermissions = serverSettings.bypassPermissions;
                }
                // Store API key availability (for UI feedback)
                this.hasAnthropicKey = serverSettings.hasAnthropicKey || false;
                this.hasOpenaiKey = serverSettings.hasOpenaiKey || false;

                // Log agent status if available (for debugging)
                if (serverSettings.agentStatus) {
                    const status = serverSettings.agentStatus;
                    this.log(`Agent: ${status.backend} (available: ${status.available})`, 'info');
                    if (status.features && status.features.length > 0) {
                        this.log(`Features: ${status.features.join(', ')}`, 'info');
                    }
                }
            }
        },

        /**
         * Handle screenshot
         */
        handleScreenshot(data) {
            if (data.image) {
                this.screenshotSrc = `data:image/png;base64,${data.image}`;
                this.showScreenshot = true;
            }
        },

        /**
         * Handle code blocks
         */
        handleCode(data) {
            const content = data.content || '';
            if (this.isStreaming) {
                this.streamingContent += '\n```\n' + content + '\n```\n';
            } else {
                this.addMessage('assistant', '```\n' + content + '\n```');
            }
        },

        /**
         * Handle errors
         */
        handleError(data) {
            const content = data.content || 'Unknown error';
            this.addMessage('assistant', '❌ ' + content);
            this.log(content, 'error');
            this.showToast(content, 'error');
            this.endStreaming();

            // If file browser is open, show error there
            if (this.showFileBrowser) {
                this.fileLoading = false;
                this.fileError = content;
            }
        },

        /**
         * Handle file browser data
         */
        handleFiles(data) {
            this.fileLoading = false;
            this.fileError = null;

            if (data.error) {
                this.fileError = data.error;
                return;
            }

            this.filePath = data.path || '~';
            this.files = data.files || [];

            // Refresh Lucide icons after Alpine renders
            this.$nextTick(() => {
                if (window.refreshIcons) window.refreshIcons();
            });
        },

        /**
         * Handle reminders list
         */
        handleReminders(data) {
            this.reminders = data.reminders || [];
            this.reminderLoading = false;
        },

        /**
         * Handle reminder added
         */
        handleReminderAdded(data) {
            this.reminders.push(data.reminder);
            this.reminderInput = '';
            this.reminderLoading = false;
            this.showToast('Reminder set!', 'success');
        },

        /**
         * Handle reminder deleted
         */
        handleReminderDeleted(data) {
            this.reminders = this.reminders.filter(r => r.id !== data.id);
        },

        /**
         * Handle reminder triggered (notification)
         */
        handleReminderTriggered(data) {
            const reminder = data.reminder;
            this.showToast(`Reminder: ${reminder.text}`, 'info');
            this.addMessage('assistant', `Reminder: ${reminder.text}`);

            // Remove from local list
            this.reminders = this.reminders.filter(r => r.id !== reminder.id);

            // Try desktop notification
            if (Notification.permission === 'granted') {
                new Notification('PocketPaw Reminder', {
                    body: reminder.text,
                    icon: '/static/icon.png'
                });
            }
        },

        /**
         * Start streaming mode
         */
        startStreaming() {
            this.isStreaming = true;
            this.streamingContent = '';
        },

        /**
         * End streaming mode
         */
        endStreaming() {
            if (this.isStreaming && this.streamingContent) {
                this.addMessage('assistant', this.streamingContent);
            }
            this.isStreaming = false;
            this.streamingContent = '';
        },

        /**
         * Add a message to the chat
         */
        addMessage(role, content) {
            this.messages.push({
                role,
                content,
                time: Tools.formatTime()
            });

            // Auto scroll to bottom with slight delay for DOM update
            this.$nextTick(() => {
                this.scrollToBottom();
            });
        },

        /**
         * Scroll chat to bottom
         */
        scrollToBottom() {
            const el = this.$refs.messages;
            if (el) {
                // Use requestAnimationFrame for smoother scrolling
                requestAnimationFrame(() => {
                    el.scrollTop = el.scrollHeight;
                });
            }
        },

        /**
         * Send a chat message
         */
        sendMessage() {
            const text = this.inputText.trim();
            if (!text) return;

            // Check for skill command (starts with /)
            if (text.startsWith('/')) {
                const parts = text.slice(1).split(' ');
                const skillName = parts[0];
                const args = parts.slice(1).join(' ');

                // Add user message
                this.addMessage('user', text);
                this.inputText = '';

                // Run the skill
                socket.send('run_skill', { name: skillName, args });
                this.log(`Running skill: /${skillName} ${args}`, 'info');
                return;
            }

            // Add user message
            this.addMessage('user', text);
            this.inputText = '';

            // Start streaming indicator
            this.startStreaming();

            // Send to server
            socket.chat(text);

            this.log(`You: ${text}`, 'info');
        },

        /**
         * Run a tool
         */
        runTool(tool) {
            this.log(`Running tool: ${tool}`, 'info');

            // Special handling for file browser
            if (tool === 'fetch') {
                this.openFileBrowser();
                return;
            }

            socket.runTool(tool);
        },

        /**
         * Open file browser modal
         */
        openFileBrowser() {
            this.showFileBrowser = true;
            this.fileLoading = true;
            this.fileError = null;
            this.files = [];
            this.filePath = '~';

            // Refresh icons after modal renders
            this.$nextTick(() => {
                if (window.refreshIcons) window.refreshIcons();
            });

            socket.send('browse', { path: '~' });
        },

        /**
         * Navigate to a directory
         */
        navigateTo(path) {
            this.fileLoading = true;
            this.fileError = null;
            socket.send('browse', { path });
        },

        /**
         * Navigate up one directory
         */
        navigateUp() {
            const parts = this.filePath.split('/').filter(s => s);
            parts.pop();
            const newPath = parts.length > 0 ? parts.join('/') : '~';
            this.navigateTo(newPath);
        },

        /**
         * Navigate to a path segment (breadcrumb click)
         */
        navigateToSegment(index) {
            const parts = this.filePath.split('/').filter(s => s);
            const newPath = parts.slice(0, index + 1).join('/');
            this.navigateTo(newPath || '~');
        },

        /**
         * Select a file or folder
         */
        selectFile(item) {
            if (item.isDir) {
                // Navigate into directory
                const newPath = this.filePath === '~'
                    ? item.name
                    : `${this.filePath}/${item.name}`;
                this.navigateTo(newPath);
            } else {
                // File selected - could download or preview
                this.log(`Selected file: ${item.name}`, 'info');
                this.showToast(`Selected: ${item.name}`, 'info');
                // TODO: Add file download/preview functionality
            }
        },

        /**
         * Open reminders panel
         */
        openReminders() {
            this.showReminders = true;
            this.reminderLoading = true;
            socket.send('get_reminders');

            // Request notification permission
            if (Notification.permission === 'default') {
                Notification.requestPermission();
            }

            this.$nextTick(() => {
                if (window.refreshIcons) window.refreshIcons();
            });
        },

        /**
         * Add a reminder
         */
        addReminder() {
            const text = this.reminderInput.trim();
            if (!text) return;

            this.reminderLoading = true;
            socket.send('add_reminder', { message: text });
            this.log(`Setting reminder: ${text}`, 'info');
        },

        /**
         * Delete a reminder
         */
        deleteReminder(id) {
            socket.send('delete_reminder', { id });
        },

        /**
         * Format reminder time for display
         */
        formatReminderTime(reminder) {
            const date = new Date(reminder.trigger_at);
            return date.toLocaleString(undefined, {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        },

        // ==================== Intentions ====================

        /**
         * Handle intentions list
         */
        handleIntentions(data) {
            this.intentions = data.intentions || [];
            this.intentionLoading = false;
        },

        /**
         * Handle intention created
         */
        handleIntentionCreated(data) {
            this.intentions.push(data.intention);
            this.resetIntentionForm();
            this.showToast('Intention created!', 'success');
            this.$nextTick(() => {
                if (window.refreshIcons) window.refreshIcons();
            });
        },

        /**
         * Handle intention updated
         */
        handleIntentionUpdated(data) {
            const index = this.intentions.findIndex(i => i.id === data.intention.id);
            if (index !== -1) {
                this.intentions[index] = data.intention;
            }
        },

        /**
         * Handle intention toggled
         */
        handleIntentionToggled(data) {
            const index = this.intentions.findIndex(i => i.id === data.intention.id);
            if (index !== -1) {
                this.intentions[index] = data.intention;
            }
            this.$nextTick(() => {
                if (window.refreshIcons) window.refreshIcons();
            });
        },

        /**
         * Handle intention deleted
         */
        handleIntentionDeleted(data) {
            this.intentions = this.intentions.filter(i => i.id !== data.id);
        },

        /**
         * Handle intention execution events
         */
        handleIntentionEvent(data) {
            const eventType = data.type;

            if (eventType === 'intention_started') {
                this.showToast(`Running: ${data.intention_name}`, 'info');
                this.log(`Intention started: ${data.intention_name}`, 'info');
                this.startStreaming();
            } else if (eventType === 'intention_completed') {
                this.log(`Intention completed: ${data.intention_name}`, 'success');
                this.endStreaming();
                // Refresh intentions to update next_run time
                socket.send('get_intentions');
            } else if (eventType === 'intention_error') {
                this.showToast(`Error: ${data.error}`, 'error');
                this.log(`Intention error: ${data.error}`, 'error');
                this.endStreaming();
            } else if (data.content) {
                // Stream content from agent
                if (this.isStreaming) {
                    this.streamingContent += data.content;
                }
            }
        },

        /**
         * Open intentions panel
         */
        openIntentions() {
            this.showIntentions = true;
            this.intentionLoading = true;
            socket.send('get_intentions');

            this.$nextTick(() => {
                if (window.refreshIcons) window.refreshIcons();
            });
        },

        /**
         * Create a new intention
         */
        createIntention() {
            const { name, prompt, schedulePreset, customCron, includeSystemStatus } = this.intentionForm;

            if (!name.trim() || !prompt.trim() || !schedulePreset) {
                this.showToast('Please fill in all fields', 'error');
                return;
            }

            const schedule = schedulePreset === 'custom' ? customCron : schedulePreset;
            if (!schedule) {
                this.showToast('Please enter a schedule', 'error');
                return;
            }

            const contextSources = includeSystemStatus ? ['system_status', 'datetime'] : ['datetime'];

            socket.send('create_intention', {
                name: name.trim(),
                prompt: prompt.trim(),
                trigger: { type: 'cron', schedule },
                context_sources: contextSources,
                enabled: true
            });

            this.log(`Creating intention: ${name}`, 'info');
        },

        /**
         * Toggle intention enabled state
         */
        toggleIntention(id) {
            socket.send('toggle_intention', { id });
        },

        /**
         * Delete an intention
         */
        deleteIntention(id) {
            if (confirm('Delete this intention?')) {
                socket.send('delete_intention', { id });
            }
        },

        /**
         * Run an intention immediately
         */
        runIntention(id) {
            socket.send('run_intention', { id });
        },

        /**
         * Reset intention form
         */
        resetIntentionForm() {
            this.intentionForm = {
                name: '',
                prompt: '',
                schedulePreset: '',
                customCron: '',
                includeSystemStatus: false
            };
        },

        /**
         * Format next run time for display
         */
        formatNextRun(isoString) {
            if (!isoString) return '';
            const date = new Date(isoString);
            const now = new Date();
            const diff = date - now;

            // If less than 1 hour away, show relative time
            if (diff > 0 && diff < 3600000) {
                const mins = Math.round(diff / 60000);
                return `in ${mins}m`;
            }

            // Otherwise show time
            return date.toLocaleString(undefined, {
                hour: '2-digit',
                minute: '2-digit'
            });
        },

        // ==================== Skills ====================

        /**
         * Handle skills list
         */
        handleSkills(data) {
            this.skills = data.skills || [];
            this.skillsLoading = false;
            this.$nextTick(() => {
                if (window.refreshIcons) window.refreshIcons();
            });
        },

        /**
         * Handle skill started
         */
        handleSkillStarted(data) {
            this.showToast(`Running: ${data.skill_name}`, 'info');
            this.log(`Skill started: ${data.skill_name}`, 'info');
        },

        /**
         * Handle skill completed
         */
        handleSkillCompleted(data) {
            this.log(`Skill completed: ${data.skill_name}`, 'success');
        },

        /**
         * Handle skill error
         */
        handleSkillError(data) {
            this.showToast(`Skill error: ${data.error}`, 'error');
            this.log(`Skill error: ${data.error}`, 'error');
        },

        /**
         * Open skills panel
         */
        openSkills() {
            this.showSkills = true;
            this.skillsLoading = true;
            socket.send('get_skills');

            this.$nextTick(() => {
                if (window.refreshIcons) window.refreshIcons();
            });
        },

        /**
         * Run a skill
         */
        runSkill(name, args = '') {
            this.showSkills = false;
            socket.send('run_skill', { name, args });
            this.log(`Running skill: ${name} ${args}`, 'info');
        },

        /**
         * Check if input is a skill command and run it
         */
        checkSkillCommand(text) {
            if (text.startsWith('/')) {
                const parts = text.slice(1).split(' ');
                const skillName = parts[0];
                const args = parts.slice(1).join(' ');

                // Check if skill exists
                const skill = this.skills.find(s => s.name === skillName);
                if (skill) {
                    this.runSkill(skillName, args);
                    return true;
                }
            }
            return false;
        },

        /**
         * Toggle agent mode
         */
        toggleAgent() {
            socket.toggleAgent(this.agentActive);
            this.log(`Switched Agent Mode: ${this.agentActive ? 'ON' : 'OFF'}`, 'info');
        },

        /**
         * Save settings
         */
        saveSettings() {
            socket.saveSettings(
                this.settings.agentBackend, 
                this.settings.llmProvider, 
                this.settings.anthropicModel,
                this.settings.bypassPermissions
            );
            this.log('Settings updated', 'info');
            this.showToast('Settings saved', 'success');
        },

        /**
         * Save API key
         */
        saveApiKey(provider) {
            const key = this.apiKeys[provider];
            if (!key) {
                this.showToast('Please enter an API key', 'error');
                return;
            }
            
            socket.saveApiKey(provider, key);
            this.apiKeys[provider] = ''; // Clear input
            this.log(`Saved ${provider} API key`, 'success');
            this.showToast(`${provider.charAt(0).toUpperCase() + provider.slice(1)} API key saved!`, 'success');
        },

        /**
         * Start polling for system status (every 10 seconds, only when connected)
         */
        startStatusPolling() {
            setInterval(() => {
                if (socket.isConnected) {
                    socket.runTool('status');
                }
            }, 10000); // Poll every 10 seconds, not 3
        },

        /**
         * Add log entry
         */
        log(message, level = 'info') {
            this.logs.push({
                time: Tools.formatTime(),
                message,
                level
            });
            
            // Keep only last 100 logs
            if (this.logs.length > 100) {
                this.logs.shift();
            }
            
            // Auto scroll terminal
            this.$nextTick(() => {
                if (this.$refs.terminal) {
                    this.$refs.terminal.scrollTop = this.$refs.terminal.scrollHeight;
                }
            });
        },

        /**
         * Format message content
         */
        formatMessage(content) {
            return Tools.formatMessage(content);
        },

        /**
         * Get friendly label for current agent mode (shown in top bar)
         */
        getAgentModeLabel() {
            const labels = {
                'claude_agent_sdk': '🚀 Claude SDK',
                'pocketpaw_native': '🐾 PocketPaw',
                'open_interpreter': '🤖 Open Interpreter'
            };
            return labels[this.settings.agentBackend] || this.settings.agentBackend;
        },

        /**
         * Get description for each backend (shown in settings)
         */
        getBackendDescription(backend) {
            const descriptions = {
                'claude_agent_sdk': 'Built-in tools: Bash, WebSearch, WebFetch, Read, Write, Edit, Glob, Grep',
                'pocketpaw_native': 'Anthropic API + Open Interpreter executor. Direct subprocess for speed.',
                'open_interpreter': 'Standalone agent. Works with local LLMs (Ollama) or cloud APIs.'
            };
            return descriptions[backend] || '';
        },

        // ==================== Transparency ====================

        /**
         * Handle connection info (capture session ID)
         */
        handleConnectionInfo(data) {
            this.handleNotification(data);
            if (data.id) {
                this.sessionId = data.id;
                this.log(`Session ID: ${data.id}`, 'info');
            }
        },

        /**
         * Handle system event (Activity Log)
         */
        handleSystemEvent(data) {
           const time = Tools.formatTime();
           let message = '';
           let level = 'info';

           if (data.event_type === 'thinking') {
               message = `<span class="text-accent animate-pulse">Thinking...</span>`;
           } else if (data.event_type === 'tool_start') {
               message = `🔧 <b>${data.data.name}</b> <span class="text-white/50">${JSON.stringify(data.data.params)}</span>`;
               level = 'warning';
           } else if (data.event_type === 'tool_result') {
               const isError = data.data.status === 'error';
               level = isError ? 'error' : 'success';
               message = `${isError ? '❌' : '✅'} <b>${data.data.name}</b> result: <span class="text-white/50">${String(data.data.result).substring(0, 50)} ${(String(data.data.result).length > 50) ? '...' : ''}</span>`;
           } else {
               message = `Unknown event: ${data.event_type}`;
           }

           this.activityLog.push({ time, message, level });
           
           // Auto-scroll activity log
           this.$nextTick(() => {
               const term = this.$refs.activityLog;
               if (term) term.scrollTop = term.scrollHeight;
           });
        },

        openIdentity() {
            this.showIdentity = true;
            this.identityLoading = true;
            fetch('/api/identity')
                .then(r => r.json())
                .then(data => {
                    this.identityData = data;
                    this.identityLoading = false;
                })
                .catch(e => {
                    this.showToast('Failed to load identity', 'error');
                    this.identityLoading = false;
                });
        },

        openMemory() {
            this.showMemory = true;
            this.memoryLoading = true;
            this.loadSessionsList();
            this.loadLongTermMemory();
        },

        loadSessionsList() {
            fetch('/api/memory/sessions')
                .then(r => r.json())
                .then(data => {
                    this.sessionsList = data;
                    this.updateMemoryStats();

                    // Auto-select current session if in list
                    if (this.sessionId && data.some(s => s.id === this.sessionId)) {
                        this.selectSession(this.sessionId);
                    }
                })
                .catch(e => {
                    console.error('Failed to load sessions:', e);
                });
        },

        selectSession(sessionId) {
            this.selectedSession = sessionId;
            fetch(`/api/memory/session?id=${sessionId}`)
                .then(r => r.json())
                .then(data => {
                    this.sessionMemory = data;
                });
        },

        loadLongTermMemory() {
            fetch('/api/memory/long_term')
                .then(r => r.json())
                .then(data => {
                    this.longTermMemory = data;
                    this.updateMemoryStats();
                    this.memoryLoading = false;
                })
                .catch(e => {
                    console.error('Failed to load memories:', e);
                    this.memoryLoading = false;
                });
        },

        updateMemoryStats() {
            const totalSessionMessages = this.sessionsList.reduce((sum, s) => sum + (s.message_count || 0), 0);
            this.memoryStats = {
                total_memories: this.longTermMemory.length + totalSessionMessages,
                active_sessions: this.sessionsList.length,
                long_term_facts: this.longTermMemory.length,
                user_interactions: this.sessionMemory.filter(m => m.role === 'user').length
            };
        },

        /**
         * Get filtered memories based on search query
         */
        get filteredMemories() {
            const search = this.memorySearch.toLowerCase().trim();
            const allMemories = [
                ...this.longTermMemory.map(m => ({ ...m, type: 'long_term', id: m.timestamp })),
                ...this.sessionMemory.map((m, i) => ({ ...m, type: 'session', id: `session-${i}`, created_at: new Date().toISOString() }))
            ];

            if (!search) return allMemories;

            return allMemories.filter(m =>
                m.content?.toLowerCase().includes(search) ||
                m.tags?.some(t => t.toLowerCase().includes(search))
            );
        },

        /**
         * Get CSS class for memory type badge
         */
        getMemoryTypeClass(type) {
            const classes = {
                'long_term': 'bg-purple-500/20 text-purple-400 border-purple-500/30',
                'session': 'bg-blue-500/20 text-blue-400 border-blue-500/30',
                'daily': 'bg-green-500/20 text-green-400 border-green-500/30'
            };
            return classes[type] || 'bg-gray-500/20 text-gray-400 border-gray-500/30';
        },

        /**
         * Format date for display
         */
        formatDate(dateStr) {
            if (!dateStr) return '';
            try {
                const date = new Date(dateStr);
                return date.toLocaleDateString('en-US', {
                    month: 'short',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit'
                });
            } catch (e) {
                return dateStr;
            }
        },

        /**
         * Delete a memory (placeholder - needs backend endpoint)
         */
        deleteMemory(id) {
            // TODO: Add backend endpoint for memory deletion
            this.showToast('Memory deletion not yet implemented', 'info');
        },

        openAudit() {
            this.showAudit = true;
            this.auditLoading = true;
            fetch('/api/audit')
                .then(r => r.json())
                .then(data => {
                    this.auditLogs = data;
                    this.auditLoading = false;
                })
                .catch(e => {
                     this.showToast('Failed to load audit logs', 'error');
                     this.auditLoading = false;
                });
        },

        /**
         * Get current time string
         */
        currentTime() {
            return Tools.formatTime();
        },

        /**
         * Show toast notification
         */
        showToast(message, type = 'info') {
            Tools.showToast(message, type, this.$refs.toasts);
        },

        // ==================== Remote Access ====================

        /**
         * Open Remote Access modal
         */
        async openRemote() {
            this.showRemote = true;
            this.tunnelLoading = true;
            
            try {
                const res = await fetch('/api/remote/status');
                if (res.ok) {
                    this.remoteStatus = await res.json();
                }
            } catch (e) {
                console.error('Failed to get tunnel status', e);
            } finally {
                this.tunnelLoading = false;
            }
        },

        /**
         * Toggle Cloudflare Tunnel
         */
        async toggleTunnel() {
            this.tunnelLoading = true;
            try {
                const endpoint = this.remoteStatus.active ? '/api/remote/stop' : '/api/remote/start';
                const res = await fetch(endpoint, { method: 'POST' });
                const data = await res.json();
                
                if (data.error) {
                    this.showToast(data.error, 'error');
                } else {
                    // Refresh status
                    const statusRes = await fetch('/api/remote/status');
                    this.remoteStatus = await statusRes.json();
                    
                    if (this.remoteStatus.active) {
                        this.showToast('Tunnel Started! You can now access remotely.', 'success');
                    } else {
                        this.showToast('Tunnel Stopped.', 'info');
                    }
                }
            } catch (e) {
                this.showToast('Failed to toggle tunnel: ' + e.message, 'error');
            } finally {
                this.tunnelLoading = false;
            }
        },

        /**
         * Regenerate Access Token
         */
        async regenerateToken() {
            if (!confirm('Are you sure? This will invalidate all existing sessions (including your phone).')) return;

            try {
                const res = await fetch('/api/token/regenerate', { method: 'POST' });
                const data = await res.json();

                if (data.token) {
                    localStorage.setItem('pocketpaw_token', data.token);
                    this.showToast('Token regenerated! Please re-scan the QR code.', 'success');
                    // Force refresh QR code image
                    this.showRemote = false;
                    setTimeout(() => { this.showRemote = true; }, 100);
                }
            } catch (e) {
                this.showToast('Failed to regenerate token', 'error');
            }
        },

        /**
         * Get Telegram configuration status
         */
        async getTelegramStatus() {
            try {
                const res = await fetch('/api/telegram/status');
                if (res.ok) {
                    this.telegramStatus = await res.json();
                }
            } catch (e) {
                console.error('Failed to get Telegram status', e);
            }
        },

        /**
         * Start Telegram pairing flow
         */
        async startTelegramPairing() {
            this.telegramLoading = true;
            this.telegramForm.error = '';
            this.telegramForm.qrCode = '';

            try {
                const res = await fetch('/api/telegram/setup', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ bot_token: this.telegramForm.botToken })
                });
                const data = await res.json();

                if (data.error) {
                    this.telegramForm.error = data.error;
                } else if (data.qr_url) {
                    this.telegramForm.qrCode = data.qr_url;
                    // Start polling for pairing completion
                    this.startTelegramPolling();
                }
            } catch (e) {
                this.telegramForm.error = 'Failed to connect: ' + e.message;
            } finally {
                this.telegramLoading = false;
            }
        },

        /**
         * Poll for Telegram pairing completion
         */
        startTelegramPolling() {
            // Clear any existing interval
            if (this.telegramPollInterval) {
                clearInterval(this.telegramPollInterval);
            }

            this.telegramPollInterval = setInterval(async () => {
                try {
                    const res = await fetch('/api/telegram/pairing-status');
                    const data = await res.json();

                    if (data.paired) {
                        clearInterval(this.telegramPollInterval);
                        this.telegramPollInterval = null;
                        this.telegramForm.qrCode = '';
                        this.telegramForm.botToken = '';
                        this.telegramStatus = { configured: true, user_id: data.user_id };
                        this.showToast('Telegram connected successfully!', 'success');
                        // Reinit icons for the success state
                        setTimeout(() => lucide.createIcons(), 100);
                    }
                } catch (e) {
                    console.error('Polling error', e);
                }
            }, 2000);
        },

        /**
         * Stop Telegram polling (cleanup)
         */
        stopTelegramPolling() {
            if (this.telegramPollInterval) {
                clearInterval(this.telegramPollInterval);
                this.telegramPollInterval = null;
            }
        }
    };
}
