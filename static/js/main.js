$(document).ready(function() {
    let currentMode = 'internal';

    // Sidebar Toggle
    $("#menu-toggle").click(function(e) {
        e.preventDefault();
        $("#wrapper").toggleClass("toggled");
    });

    // Mode Switching
    $("#btn-internal").click(function(e) {
        e.preventDefault();
        switchMode('internal');
    });

    $("#btn-external").click(function(e) {
        e.preventDefault();
        switchMode('external');
    });

    function switchMode(mode) {
        currentMode = mode;
        $(".list-group-item").removeClass("active");
        if (mode === 'internal') {
            $("#btn-internal").addClass("active");
            $("#header-title").text("调用内推模型");
            $("#current-mode-text").text("内推");
        } else {
            $("#btn-external").addClass("active");
            $("#header-title").text("调用外推模型");
            $("#current-mode-text").text("外推");
        }
    }

    // Send Message
    $("#send-btn").click(sendMessage);
    $("#user-input").keypress(function(e) {
        if (e.which == 13) sendMessage();
    });

    function sendMessage() {
        const question = $("#user-input").val().trim();
        if (!question) return;

        // Add User Message
        appendMessage('user', question);
        $("#user-input").val('');

        // Loading Indicator
        const loadingId = 'loading-' + Date.now();
        appendLoading(loadingId);

        // API Call
        $.ajax({
            url: '/api/query',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                question: question,
                mode: currentMode
            }),
            success: function(response) {
                removeLoading(loadingId);
                renderResponse(response);
            },
            error: function(err) {
                removeLoading(loadingId);
                appendMessage('system', "抱歉，系统出现错误，请稍后再试。");
                console.error(err);
            }
        });
    }

    function appendMessage(sender, text) {
        const typeClass = sender === 'user' ? 'user-message' : 'system-message';
        const icon = sender === 'user' ? '<i class="fas fa-user text-secondary ms-2"></i>' : '<i class="fas fa-robot text-primary me-2"></i>';
        
        const html = `
            <div class="chat-message ${typeClass}">
                <div class="card border-0 shadow-sm">
                    <div class="card-body py-2 px-3">
                        ${sender === 'system' ? icon : ''}
                        ${text}
                        ${sender === 'user' ? icon : ''}
                    </div>
                </div>
            </div>
        `;
        $("#chat-box").append(html);
        scrollToBottom();
    }

    function appendLoading(id) {
        const html = `
            <div class="chat-message system-message" id="${id}">
                <div class="card border-0 shadow-sm">
                    <div class="card-body py-2 px-3">
                        <i class="fas fa-spinner fa-spin text-primary me-2"></i> 正在分析...
                    </div>
                </div>
            </div>
        `;
        $("#chat-box").append(html);
        scrollToBottom();
    }

    function removeLoading(id) {
        $("#" + id).remove();
    }

    function scrollToBottom() {
        $("#chat-box").scrollTop($("#chat-box")[0].scrollHeight);
    }

    function renderResponse(data) {
        let content = '';

        // 1. NLP Analysis Info (Visualized)
        if (data.analysis) {
            content += `<div class="mb-3">
                <div class="d-flex align-items-center mb-2">
                    <i class="fas fa-search-plus text-warning me-2"></i>
                    <strong>系统认知分析:</strong>
                </div>
                <div class="card bg-light border-0">
                    <div class="card-body p-2" style="font-size: 0.9em;">
                        <!-- Segmentation & Entities -->
                        <div class="mb-2">
                            <small class="text-muted d-block mb-1">文本分词与实体识别:</small>
                            ${renderSegmentation(data.analysis)}
                        </div>
                        <!-- Structured Query -->
                        <div>
                            <small class="text-muted d-block mb-1">解析意图 (四元组):</small>
                            ${renderStructuredQuery(data.analysis.structured_query)}
                        </div>
                    </div>
                </div>
            </div>`;
        }

        // 2. Graph Result
        content += `<strong><i class="fas fa-database text-success"></i> 知识库查询结果:</strong><br>`;
        if (data.graph_result && data.graph_result.length > 0) {
            content += `<div class="alert alert-success mt-2">${data.graph_result.join(', ')}</div>`;
        } else {
            const msg = data.graph_message || "暂无数据";
            content += `<div class="alert alert-light border mt-2">${msg}</div>`;
        }

        // 3. Reasoning Result
        if (data.reasoning_result && data.reasoning_result.length > 0) {
            const title = currentMode === 'internal' ? "内推模型预测结果 (Top 5)" : "外推模型预测结果 (Top 5)";
            content += `<div class="mt-3"><strong><i class="fas fa-brain text-info"></i> ${title}:</strong></div>`;
            
            content += `<div class="list-group mt-2">`;
            const filtered = data.reasoning_result.filter(item => {
                const s = item.score ?? item.probability;
                if (typeof s === 'number') return s >= 0.05;
                return true;
            });
            filtered.forEach((item, index) => {
                const badgeClass = index === 0 ? 'bg-danger' : 'bg-secondary';
                const rawScore = item.score ?? item.probability;
                const scoreText = typeof rawScore === 'number' ? rawScore.toFixed(2) : rawScore;
                content += `
                    <div class="list-group-item d-flex justify-content-between align-items-center">
                        <div>${item.name || item.prediction}</div>
                        <span class="badge ${badgeClass} rounded-pill">${scoreText}</span>
                    </div>
                `;
            });
            content += `</div>`;
        }

        // Wrap in system message
        const html = `
            <div class="chat-message system-message">
                <div class="card border-0 shadow-sm">
                    <div class="card-body py-2 px-3">
                        <i class="fas fa-robot text-primary me-2"></i>
                        <div>${content}</div>
                    </div>
                </div>
            </div>
        `;
        $("#chat-box").append(html);
        scrollToBottom();
    }

    function renderSegmentation(analysis) {
        // Create a map of tokens to types based on AC matches
        // This is a simple visualization strategy
        let html = '';
        const matches = analysis.ac_matches || [];
        
        // Use segmentation result but highlight known entities
        analysis.segmentation.forEach(seg => {
            let badgeClass = 'bg-secondary bg-opacity-10 text-dark'; // default
            let title = seg.flag;
            
            // Check if this word was matched by AC Automaton
            const match = matches.find(m => m.word === seg.word);
            if (match) {
                if (match.type === 'ENTITY') badgeClass = 'bg-primary text-white';
                else if (match.type === 'RELATION') badgeClass = 'bg-success text-white';
                else if (match.type === 'TIME') badgeClass = 'bg-info text-white';
            }
            
            html += `<span class="badge ${badgeClass} me-1 mb-1" title="${title}">${seg.word}</span>`;
        });
        return html;
    }

    function renderStructuredQuery(q) {
        if (!q) return '<span class="text-muted">无法解析</span>';
        
        const h = q.h ? `<span class="badge bg-primary">${q.h}</span>` : '<span class="text-muted">?</span>';
        const r = q.r ? `<span class="badge bg-success">${q.r}</span>` : '<span class="text-muted">?</span>';
        const t = q.t ? `<span class="badge bg-primary">${q.t}</span>` : '<span class="text-muted">?</span>';
        const time = q.time ? `<span class="badge bg-info">${q.time}</span>` : '<span class="text-muted">?</span>';
        
        return `
            <div class="d-flex align-items-center gap-2">
                <span>(</span>
                ${h} <span>,</span>
                ${r} <span>,</span>
                ${t} <span>,</span>
                ${time}
                <span>)</span>
            </div>
        `;
    }
});
