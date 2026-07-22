document.addEventListener('DOMContentLoaded', () => {
    const el = id => document.getElementById(id);
    let pollHandle = null;

    // --- DOM Elements ---
    const sidebar = el('sidebar');
    const toggleSidebarBtn = el('toggleSidebarBtn');



    // Views
    const viewInput = el('view-input');
    const viewSummary = el('view-summary');
    const viewChat = el('view-chat');

    // Modal
    const langModal = el('language-modal');
    const langCards = document.querySelectorAll('.lang-card');
    let selectedLanguage = 'english';
    let pendingSource = '';

    // Collapsible Summary
    const summaryHeader = el('summary-header');
    const summaryBox = el('summary-box');

    // Inputs
    const layoutWrapper = el('layout-wrapper');
    const fileInput = el('local-file-input');
    const btnYtModal = el('btn-yt-modal');
    const ytModal = el('yt-modal');
    const inputUrl = el('source-url');
    const fileNameDisplay = el('file-name-display');

    // Chat
    const chatInput = el('chat-input');
    const sendChatBtn = el('send-chat-btn');
    const chatArea = el('chat-area');

    // --- UI Interactions ---



    // Language Selection Modal
    langCards.forEach(card => {
        card.addEventListener('click', () => {
            langCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            selectedLanguage = card.dataset.lang;
        });
    });

    el('cancel-lang-btn').addEventListener('click', () => {
        langModal.style.display = 'none';
        pendingSource = '';
    });

    el('confirm-lang-btn').addEventListener('click', () => {
        langModal.style.display = 'none';
        startAnalysis(pendingSource, selectedLanguage);
    });

    const uploadArea = el('upload-area');

    function handleFile(file) {
        if (!requireConfig()) return;
        if (file) {
            fileNameDisplay.innerText = file.name;
            pendingSource = 'file';

            // Assign the dropped/pasted file to the input element so FormData can grab it
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            fileInput.files = dataTransfer.files;

            langModal.style.display = 'flex';
        }
    }

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag and Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Paste
    document.addEventListener('paste', (e) => {
        if (viewInput.classList.contains('hidden')) return;
        if (e.clipboardData.files.length > 0) {
            handleFile(e.clipboardData.files[0]);
        }
    });

    btnYtModal.addEventListener('click', (e) => {
        e.stopPropagation(); // prevent clicking upload area if nested
        if (!requireConfig()) return;
        ytModal.style.display = 'flex';
    });

    const btnMaxSummary = el('btn-max-summary');
    const btnMaxChat = el('btn-max-chat');

    if (btnMaxSummary) {
        btnMaxSummary.addEventListener('click', () => {
            if (layoutWrapper.classList.contains('max-summary')) {
                layoutWrapper.classList.remove('max-summary');
            } else {
                layoutWrapper.classList.remove('max-chat');
                layoutWrapper.classList.add('max-summary');
            }
        });
    }

    if (btnMaxChat) {
        btnMaxChat.addEventListener('click', () => {
            if (layoutWrapper.classList.contains('max-chat')) {
                layoutWrapper.classList.remove('max-chat');
            } else {
                layoutWrapper.classList.remove('max-summary');
                layoutWrapper.classList.add('max-chat');
            }
        });
    }

    if (el('cancel-yt-btn')) {
        el('cancel-yt-btn').addEventListener('click', () => {
            ytModal.style.display = 'none';
            inputUrl.value = '';
        });
    }

    if (el('confirm-yt-btn')) {
        el('confirm-yt-btn').addEventListener('click', () => {
            const val = inputUrl.value.trim();
            if (!val) { alert('Enter a URL first'); return; }
            pendingSource = val;
            ytModal.style.display = 'none';
            langModal.style.display = 'flex';
        });
    }



    // --- Configuration Logic ---
    const viewConfig = el('view-config');
    const btnConfigView = el('btn-config-view');
    const btnSaveConfig = el('btn-save-config');
    const btnCancelConfig = el('btn-cancel-config');
    const configProvider = el('config-provider');
    const configModel = el('config-model');
    const configApiKey = el('config-api-key');

    const configTranscriptionMode = el('config-transcription-mode');
    const configTranscriptionModel = el('config-transcription-model');
    const configHfToken = el('config-hf-token');
    const hfTokenGroup = el('hf-token-group');

    const configEmbeddingMode = el('config-embedding-mode');
    const configEmbeddingModel = el('config-embedding-model');

    const OFFLINE_MODELS = [
        { value: "large-v3", text: "faster-whisper large-v3", url: "https://huggingface.co/Systran/faster-whisper-large-v3" },
        { value: "base", text: "faster-whisper base", url: "https://huggingface.co/Systran/faster-whisper-base" },
        { value: "small", text: "faster-whisper small", url: "https://huggingface.co/Systran/faster-whisper-small" },
        { value: "medium", text: "faster-whisper medium", url: "https://huggingface.co/Systran/faster-whisper-medium" },
        { value: "openai/whisper-large-v3-turbo", text: "Transformers whisper-large-v3-turbo", url: "https://huggingface.co/openai/whisper-large-v3-turbo" }
    ];

    const ONLINE_MODELS = [
        { value: "openai/whisper-large-v3-turbo", text: "HF Inference whisper-large-v3-turbo", url: "https://huggingface.co/openai/whisper-large-v3-turbo" },
        { value: "openai/whisper-large-v3", text: "HF Inference whisper-large-v3", url: "https://huggingface.co/openai/whisper-large-v3" }
    ];

    const EMBEDDING_MODELS = [
        { value: "sentence-transformers/all-MiniLM-L6-v2", text: "all-MiniLM-L6-v2", url: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2" },
        { value: "sentence-transformers/all-MiniLM-L12-v2", text: "all-MiniLM-L12-v2", url: "https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2" },
        { value: "google/embeddinggemma-300m", text: "google/embeddinggemma-300m", url: "https://huggingface.co/google/embeddinggemma-300m" },
        { value: "LiquidAI/LFM2.5-Embedding-350M", text: "LiquidAI/LFM2.5-Embedding-350M", url: "https://huggingface.co/LiquidAI/LFM2.5-Embedding-350M" }
    ];

    function updateTranscriptionModels(mode, selectedModel = null) {
        configTranscriptionModel.innerHTML = '';
        const models = mode === 'online' ? ONLINE_MODELS : OFFLINE_MODELS;
        models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.value;
            opt.innerText = m.text;
            opt.dataset.url = m.url;
            configTranscriptionModel.appendChild(opt);
        });

        if (mode === 'online' || configEmbeddingMode.value === 'online') {
            hfTokenGroup.classList.remove('hidden');
        } else {
            hfTokenGroup.classList.add('hidden');
        }

        if (selectedModel) {
            configTranscriptionModel.value = selectedModel;
        }

        updateTranscriptionLink();
    }

    function updateEmbeddingModels(mode, selectedModel = null) {
        configEmbeddingModel.innerHTML = '';
        EMBEDDING_MODELS.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.value;
            opt.innerText = m.text;
            opt.dataset.url = m.url;
            configEmbeddingModel.appendChild(opt);
        });

        if (mode === 'online' || configTranscriptionMode.value === 'online') {
            hfTokenGroup.classList.remove('hidden');
        } else {
            hfTokenGroup.classList.add('hidden');
        }

        if (selectedModel) {
            configEmbeddingModel.value = selectedModel;
        }

        updateEmbeddingLink();
    }

    function updateTranscriptionLink() {
        const link = el('link-transcription-model');
        const opt = configTranscriptionModel.options[configTranscriptionModel.selectedIndex];
        if (opt && opt.dataset.url) {
            link.href = opt.dataset.url;
            link.style.display = 'block';
        } else {
            link.style.display = 'none';
        }
    }

    function updateEmbeddingLink() {
        const link = el('link-embedding-model');
        const opt = configEmbeddingModel.options[configEmbeddingModel.selectedIndex];
        if (opt && opt.dataset.url) {
            link.href = opt.dataset.url;
            link.style.display = 'block';
        } else {
            link.style.display = 'none';
        }
    }

    function enforceLocalRestriction(selectElement) {
        if (selectElement.value === 'offline' && window.allowLocalModel === false) {
            const modal = el('local-model-modal');
            const repoBtn = el('btn-github-repo');
            if (repoBtn && window.githubRepo) {
                repoBtn.href = window.githubRepo;
            }
            if (modal) {
                modal.classList.remove('hidden');
                modal.style.display = 'flex';
            }
            selectElement.value = 'online';
            return true; // was restricted
        }
        return false;
    }

    if (configTranscriptionMode) {
        configTranscriptionMode.addEventListener('change', (e) => {
            if (enforceLocalRestriction(e.target)) {
                updateTranscriptionModels('online');
                return;
            }
            updateTranscriptionModels(e.target.value);
        });
    }

    if (configEmbeddingMode) {
        configEmbeddingMode.addEventListener('change', (e) => {
            if (enforceLocalRestriction(e.target)) {
                updateEmbeddingModels('online');
                return;
            }
            updateEmbeddingModels(e.target.value);
        });
    }

    if (configTranscriptionModel) {
        configTranscriptionModel.addEventListener('change', updateTranscriptionLink);
    }

    if (configEmbeddingModel) {
        configEmbeddingModel.addEventListener('change', updateEmbeddingLink);
    }

    // Step Navigation
    document.querySelectorAll('.btn-next-step').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const next = parseInt(e.target.getAttribute('data-next'));
            const current = next - 1;

            // Validation logic for current step
            if (current === 1) {
                if (!configModel.value.trim()) {
                    alert('Please enter a Model Name.');
                    return;
                }
                if (!isConfigured && !configApiKey.value.trim()) {
                    alert('Please enter your API Key.');
                    return;
                }
            } else if (current === 2) {
                if (configTranscriptionMode.value === 'online' && !isConfigured && !configHfToken.value.trim()) {
                    alert('Please enter your Hugging Face Token for online transcription.');
                    return;
                }
            }

            document.querySelectorAll('.wizard-panel').forEach(p => p.classList.add('hidden'));
            el(`wizard-step-${next}`).classList.remove('hidden');

            document.querySelectorAll('.wizard-progress span').forEach(s => {
                s.style.color = 'var(--text-muted)';
            });
            for (let i = 1; i <= next; i++) {
                el(`step-indicator-${i}`).style.color = 'var(--primary)';
            }
        });
    });

    document.querySelectorAll('.btn-prev-step').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const prev = e.target.getAttribute('data-prev');
            document.querySelectorAll('.wizard-panel').forEach(p => p.classList.add('hidden'));
            el(`wizard-step-${prev}`).classList.remove('hidden');

            document.querySelectorAll('.wizard-progress span').forEach(s => {
                s.style.color = 'var(--text-muted)';
            });
            for (let i = 1; i <= prev; i++) {
                el(`step-indicator-${i}`).style.color = 'var(--primary)';
            }
        });
    });

    let isConfigured = false;
    let currentJobId = null;

    function requireConfig() {
        if (!isConfigured) {
            alert('Please configure your model settings first.');
            viewInput.classList.add('hidden');
            viewConfig.classList.remove('hidden');
            return false;
        }
        return true;
    }

    function loadConfig() {
        fetch('/config')
            .then(r => r.json())
            .then(data => {
                window.allowLocalModel = data.allow_local_model !== undefined ? data.allow_local_model : true;
                window.githubRepo = data.github_repo || 'https://github.com/your-username/video-agent';

                let savedConfig = {};
                try {
                    savedConfig = JSON.parse(localStorage.getItem('videoAgentConfig')) || {};
                } catch (e) { }

                if (savedConfig.provider && savedConfig.model && savedConfig.transcription_mode && savedConfig.embedding_mode) {
                    isConfigured = true;
                    el('btn-cancel-config').classList.remove('hidden');
                    configProvider.value = savedConfig.provider;
                    configModel.value = savedConfig.model;
                } else {
                    isConfigured = false;
                    viewInput.classList.add('hidden');
                    viewConfig.classList.remove('hidden');
                }

                if (savedConfig.transcription_mode) {
                    configTranscriptionMode.value = savedConfig.transcription_mode;
                    if (!enforceLocalRestriction(configTranscriptionMode)) {
                        updateTranscriptionModels(savedConfig.transcription_mode, savedConfig.transcription_model);
                    } else {
                        updateTranscriptionModels('online');
                    }
                } else {
                    configTranscriptionMode.value = 'offline';
                    if (!enforceLocalRestriction(configTranscriptionMode)) {
                        updateTranscriptionModels('offline');
                    } else {
                        updateTranscriptionModels('online');
                    }
                }

                if (savedConfig.embedding_mode) {
                    configEmbeddingMode.value = savedConfig.embedding_mode;
                    if (!enforceLocalRestriction(configEmbeddingMode)) {
                        updateEmbeddingModels(savedConfig.embedding_mode, savedConfig.embedding_model);
                    } else {
                        updateEmbeddingModels('online');
                    }
                } else {
                    configEmbeddingMode.value = 'offline';
                    if (!enforceLocalRestriction(configEmbeddingMode)) {
                        updateEmbeddingModels('offline');
                    } else {
                        updateEmbeddingModels('online');
                    }
                }


                if (savedConfig.api_key) {
                    configApiKey.value = "********";
                } else {
                    configApiKey.value = "";
                }

                if (savedConfig.hf_token) {
                    configHfToken.value = "********";
                } else {
                    configHfToken.value = "";
                }
            })
            .catch(console.error);
    }


    // Call loadConfig on app startup
    loadConfig();

    if (btnConfigView) {
        btnConfigView.addEventListener('click', (e) => {
            e.stopPropagation();
            viewInput.classList.add('hidden');
            viewConfig.classList.remove('hidden');
            loadConfig();
        });
    }

    if (btnCancelConfig) {
        btnCancelConfig.addEventListener('click', () => {
            viewConfig.classList.add('hidden');
            viewInput.classList.remove('hidden');
        });
    }

    const localModal = el('local-model-modal');
    if (localModal) {
        const btnCloseLocal = el('btn-close-local-modal');
        if (btnCloseLocal) {
            btnCloseLocal.addEventListener('click', () => {
                localModal.classList.add('hidden');
                localModal.style.display = 'none';
            });
        }
    }

    if (btnSaveConfig) {
        btnSaveConfig.addEventListener('click', () => {
            const provider = configProvider.value;
            const model = configModel.value.trim();
            const apiKey = configApiKey.value.trim();

            if (!model || !apiKey) {
                alert('Please provide both Model Name and API Key.');
                return;
            }

            const btn = btnSaveConfig;
            const oldText = btn.innerText;
            btn.innerText = 'Validating model...';
            btn.disabled = true;

            const tMode = configTranscriptionMode.value;
            const tModel = configTranscriptionModel.value;
            const hfToken = configHfToken.value.trim();
            const eMode = configEmbeddingMode.value;
            const eModel = configEmbeddingModel.value;

            if ((tMode === 'online' || eMode === 'online') && !hfToken && !isConfigured) {
                alert('Please provide your Hugging Face token for online models.');
                btn.innerText = oldText;
                btn.disabled = false;
                return;
            }

            fetch('/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    provider,
                    model,
                    api_key: apiKey,
                    transcription_mode: tMode,
                    transcription_model: tModel,
                    hf_token: hfToken,
                    embedding_mode: eMode,
                    embedding_model: eModel
                })
            })
                .then(r => r.json())
                .then(data => {
                    if (data.ok) {
                        isConfigured = true;

                        let savedApiKey = apiKey;
                        if (apiKey === "********") {
                            try {
                                const oldConf = JSON.parse(localStorage.getItem('videoAgentConfig')) || {};
                                savedApiKey = oldConf.api_key || "";
                            } catch (e) { }
                        }
                        let savedHfToken = hfToken;
                        if (hfToken === "********") {
                            try {
                                const oldConf = JSON.parse(localStorage.getItem('videoAgentConfig')) || {};
                                savedHfToken = oldConf.hf_token || "";
                            } catch (e) { }
                        }

                        localStorage.setItem('videoAgentConfig', JSON.stringify({
                            provider,
                            model,
                            api_key: savedApiKey,
                            transcription_mode: tMode,
                            transcription_model: tModel,
                            hf_token: savedHfToken,
                            embedding_mode: eMode,
                            embedding_model: eModel
                        }));

                        el('btn-cancel-config').classList.remove('hidden');
                        alert('Configuration saved successfully to your browser!');
                        viewConfig.classList.add('hidden');
                        viewInput.classList.remove('hidden');
                    } else {
                        alert('Error: ' + data.error);
                    }
                })
                .catch(console.error)
                .finally(() => {
                    btn.innerText = oldText;
                    btn.disabled = false;
                });
        });
    }

    // --- Core Logic ---

    async function startAnalysis(source, language) {
        // Transition Views: hide input, show progress tracker centered
        layoutWrapper.className = 'dynamic-layout-wrapper state-processing';
        viewInput.classList.add('hidden');
        sidebar.classList.remove('hidden');
        viewSummary.classList.add('hidden');
        viewChat.classList.add('hidden');

        try {
            let bodyData;
            let fetchOptions = { method: 'POST' };

            let savedConfig = {};
            try { savedConfig = JSON.parse(localStorage.getItem('videoAgentConfig')) || {}; } catch (e) { }

            if (source === 'file') {
                bodyData = new FormData();
                bodyData.append('file', fileInput.files[0]);
                bodyData.append('language', language);
                bodyData.append('source_type', 'file');
                bodyData.append('config', JSON.stringify(savedConfig));
                fetchOptions.body = bodyData;
            } else {
                bodyData = JSON.stringify({ source, language, source_type: 'url', config: savedConfig });
                fetchOptions.body = bodyData;
                fetchOptions.headers = { 'Content-Type': 'application/json' };
            }

            const res = await fetch('/start', fetchOptions);
            const data = await res.json();

            if (data.ok) {
                currentJobId = data.job_id;
                startPoll();
            } else {
                alert(data.error || 'Failed to start pipeline');
                viewInput.classList.remove('hidden');
                viewSummary.classList.add('hidden');
            }
        } catch (error) {
            console.error("Error starting pipeline:", error);
            alert("Error connecting to server.");
        }
    }


    function startPoll() {
        if (pollHandle) clearInterval(pollHandle);
        pollHandle = setInterval(updateStatus, 1000);
    }

    function updateSubstep(elementId, status) {
        const el = document.getElementById(elementId);
        if (!el) return;
        el.className = status; // pending, running, completed
    }

    async function updateStatus() {
        try {
            if (!currentJobId) return;
            const r = await fetch(`/status?job_id=${currentJobId}`);
            const s = await r.json();

            if (s.error) {
                alert(`An error occurred during processing:\n\n${s.error}`);
                clearInterval(pollHandle);
                pollHandle = null;
                currentJobId = null;

                // Reset UI to home screen
                document.querySelectorAll('.view-section, .results-panel').forEach(el => el.classList.add('hidden'));
                sidebar.classList.add('hidden');
                viewInput.classList.remove('hidden');
                layoutWrapper.classList.remove('state-summary', 'state-chat');
                summaryBox.innerHTML = '<div class="empty-state spinner-large"></div>';
                chatArea.innerHTML = '';

                return;
            }

            const steps = s.pipeline_steps;


            // Step 1: Transcribing
            updateSubstep('sub-audio-extract', getMappedStatus(steps.audio_extract.status));
            updateSubstep('sub-transcribe-step', getMappedStatus(steps.transcribe.status));
            updateSubstep('sub-title', getMappedStatus(steps.Generating_Title.status));


            // Step 2: Summarization & RAG
            updateSubstep('sub-summarize-llm', getMappedStatus(steps.summarize_llm.status));
            updateSubstep('sub-rag-chunking', getMappedStatus(steps.rag_chunking.status));
            updateSubstep('sub-rag-embedding', getMappedStatus(steps.rag_embedding.status));

            if (steps.rag_embedding.total > 0) {
                el('rag-progress').innerText = `[${steps.rag_embedding.progress}/${steps.rag_embedding.total}]`;
            }

            updateSubstep('sub-rag-db', getMappedStatus(steps.rag_db.status));
            updateSubstep('sub-rag-complete', getMappedStatus(steps.rag_complete.status));

            // View Management
            if (s.summary && summaryBox.innerHTML.includes('spinner-large')) {
                // Summary just arrived
                summaryBox.innerHTML = marked.parse(s.summary);

                // Layout shift!
                if (!layoutWrapper.classList.contains('state-chat')) {
                    layoutWrapper.classList.add('state-summary');
                }
                viewSummary.classList.remove('hidden');
            }

            if (steps.rag_complete.status === 'done' && viewChat.classList.contains('hidden')) {
                // Step 3: RAG Ready
                el('side-step-chat').className = 'side-step-title completed';
                el('side-step-chat').innerText = "3. Chat Bot Ready";

                // Show Chat
                layoutWrapper.classList.remove('state-summary');
                layoutWrapper.classList.add('state-chat');
                viewChat.classList.remove('hidden');
                addChatMessage("RAG engine is ready. You can now ask questions about the video.", 'bot');
            } else if (steps.rag_complete.status === 'active') {
                el('side-step-chat').className = 'side-step-title running';
                el('side-step-chat').innerText = "3. Initializing Chat Bot (Running...)";
            }

            if (!viewChat.classList.contains('hidden') && !viewSummary.classList.contains('hidden')) {
                sidebar.classList.add('hidden');
            }

            if (!s.running) {
                clearInterval(pollHandle);
                pollHandle = null;
            }
        } catch (err) {
            console.error("Error polling status:", err);
        }
    }

    function getMappedStatus(serverStatus) {
        if (serverStatus === 'active') return 'running';
        if (serverStatus === 'done') return 'completed';
        return 'pending';
    }

    // --- Chat Functionality ---
    function addChatMessage(text, sender) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `chat-msg ${sender}`;
        if (sender === 'bot') {
            msgDiv.innerHTML = marked.parse(text);
        } else {
            msgDiv.innerText = text;
        }
        chatArea.appendChild(msgDiv);
        chatArea.scrollTop = chatArea.scrollHeight;
    }

    async function sendChat() {
        const q = chatInput.value.trim();
        if (!q) return;

        addChatMessage(q, 'user');
        chatInput.value = '';
        chatInput.disabled = true;
        sendChatBtn.disabled = true;

        try {
            const res = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: q, job_id: currentJobId })
            });
            const data = await res.json();

            if (data.ok) {
                addChatMessage(data.answer, 'bot');
            } else {
                addChatMessage(`Error: ${data.error}`, 'bot');
            }
        } catch (err) {
            addChatMessage("Failed to communicate with server.", 'bot');
        } finally {
            chatInput.disabled = false;
            sendChatBtn.disabled = false;
            chatInput.focus();
        }
    }

    sendChatBtn.addEventListener('click', sendChat);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendChat();
    });
});
