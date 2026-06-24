$(document).ready(function () {
    let currentMode = "internal";
    const chatHistoryByMode = { internal: "", external: "" };
    let systemReady = false;
    const settings = {
        internalMinScore: 0.05,
        externalMinScore: 0.01,
        topK: 5
    };

    (function initLoadingOverlay() {
        const disableTip = localStorage.getItem("disableStartupTip") === "1";

        function hideOverlay() {
            $("#loading-overlay").fadeOut(300);
            if (!disableTip) {
                showStartupTip();
            }
        }

        $("#send-btn").prop("disabled", true);
        $("#user-input").prop("disabled", true);

        const startAt = Date.now();
        const maxWaitMs = 120000;

        function pollStatus() {
            $.ajax({
                url: "/api/status",
                type: "GET",
                success: function (resp) {
                    if (resp && resp.loading === false && resp.ready === true) {
                        systemReady = true;
                        $("#send-btn").prop("disabled", false);
                        $("#user-input").prop("disabled", false);
                        $("#loading-text").text("系统已准备就绪");
                        $("#loading-detail").text(resp.message || "模型加载完成，可以开始提问。");
                        setTimeout(hideOverlay, 500);
                        return;
                    }

                    systemReady = false;
                    $("#send-btn").prop("disabled", true);
                    $("#user-input").prop("disabled", true);
                    $("#loading-detail").text((resp && resp.message) || "正在加载模型资源...");

                    if (Date.now() - startAt > maxWaitMs) {
                        if (resp && resp.loading === false && resp.ready === false) {
                            $("#loading-text").text("系统初始化失败");
                            $("#loading-detail").text(resp.message || "系统组件加载失败，请检查环境后重启。");
                        } else {
                            $("#loading-text").text("系统启动超时");
                            $("#loading-detail").text("模型加载时间较长，请稍后再试。");
                        }
                        setTimeout(pollStatus, 2000);
                        return;
                    }

                    setTimeout(pollStatus, 1000);
                },
                error: function () {
                    $("#loading-detail").text("正在连接服务...");
                    if (Date.now() - startAt > maxWaitMs) {
                        $("#loading-text").text("无法连接到服务");
                        $("#loading-detail").text("请稍后重试或重启程序。");
                        setTimeout(pollStatus, 2000);
                        return;
                    }
                    setTimeout(pollStatus, 1500);
                }
            });
        }

        pollStatus();
    })();

    $("#menu-toggle").click(function (e) {
        e.preventDefault();
        $("#wrapper").toggleClass("toggled");
    });

    $("#btn-internal").click(function (e) {
        e.preventDefault();
        switchMode("internal");
    });

    $("#btn-external").click(function (e) {
        e.preventDefault();
        switchMode("external");
    });

    function switchMode(mode) {
        chatHistoryByMode[currentMode] = $("#chat-box").html();
        currentMode = mode;
        $(".list-group-item").removeClass("active");

        if (mode === "internal") {
            $("#btn-internal").addClass("active");
            $("#header-title").text("调用内推模型");
            $("#current-mode-text").text("内推");
        } else {
            $("#btn-external").addClass("active");
            $("#header-title").text("调用外推模型");
            $("#current-mode-text").text("外推");
        }

        $("#chat-box").html(chatHistoryByMode[currentMode] || "");
        if (!chatHistoryByMode[currentMode]) {
            appendMessage("system", mode === "internal" ? "已切换到内推模式。" : "已切换到外推模式。");
        }
    }

    $("#settings-btn").click(function () {
        $("#setting-internal-threshold").val(settings.internalMinScore);
        $("#setting-external-threshold").val(settings.externalMinScore);
        $("#setting-topk").val(settings.topK);
        $("#setting-disable-startup-tip").prop("checked", localStorage.getItem("disableStartupTip") === "1");

        if (window.bootstrap && window.bootstrap.Modal) {
            new bootstrap.Modal(document.getElementById("settingsModal")).show();
        } else {
            $("#settingsModal").modal("show");
        }
    });

    $("#settings-save-btn").click(function () {
        const internalVal = parseFloat($("#setting-internal-threshold").val());
        const externalVal = parseFloat($("#setting-external-threshold").val());
        const topKVal = parseInt($("#setting-topk").val(), 10);

        if (!isNaN(internalVal) && internalVal >= 0 && internalVal <= 1) {
            settings.internalMinScore = internalVal;
        }
        if (!isNaN(externalVal) && externalVal >= 0 && externalVal <= 1) {
            settings.externalMinScore = externalVal;
        }
        if (!isNaN(topKVal) && topKVal > 0) {
            settings.topK = topKVal;
        }

        localStorage.setItem("disableStartupTip", $("#setting-disable-startup-tip").is(":checked") ? "1" : "0");

        if (window.bootstrap && window.bootstrap.Modal) {
            const modalEl = document.getElementById("settingsModal");
            const instance = bootstrap.Modal.getInstance(modalEl) || new bootstrap.Modal(modalEl);
            instance.hide();
        } else if ($("#settingsModal").modal) {
            $("#settingsModal").modal("hide");
        }
    });

    function showStartupTip() {
        if (window.bootstrap && window.bootstrap.Modal) {
            new bootstrap.Modal(document.getElementById("startupTipModal")).show();
        } else if ($("#startupTipModal").modal) {
            $("#startupTipModal").modal("show");
        }
    }

    $("#send-btn").click(sendMessage);
    $("#user-input").keypress(function (e) {
        if (e.which === 13) {
            sendMessage();
        }
    });

    function sendMessage() {
        const question = $("#user-input").val().trim();
        if (!question) {
            return;
        }
        if (!systemReady) {
            appendMessage("system", "系统仍在加载模型，请稍后再提问。");
            return;
        }

        appendMessage("user", question);
        $("#user-input").val("");

        const loadingId = "loading-" + Date.now();
        appendLoading(loadingId);

        $.ajax({
            url: "/api/query",
            type: "POST",
            contentType: "application/json",
            data: JSON.stringify({
                question: question,
                mode: currentMode,
                top_k: settings.topK
            }),
            success: function (response) {
                removeLoading(loadingId);
                renderResponse(response);
            },
            error: function (err) {
                removeLoading(loadingId);
                let msg = "抱歉，系统出现错误，请稍后再试。";
                try {
                    const resp = err && err.responseJSON ? err.responseJSON : null;
                    if (resp && (resp.message || resp.error)) {
                        msg = resp.message || resp.error;
                    }
                } catch (e) {
                    console.error(e);
                }
                appendMessage("system", msg);
                console.error(err);
            }
        });
    }

    function escapeHtml(value) {
        return String(value ?? "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    function appendMessage(sender, text) {
        const typeClass = sender === "user" ? "user-message" : "system-message";
        const icon = sender === "user"
            ? '<i class="fas fa-user text-secondary ms-2"></i>'
            : '<i class="fas fa-robot text-primary me-2"></i>';
        const safeText = escapeHtml(text || "");

        const html = `
            <div class="chat-message ${typeClass}">
                <div class="card border-0 shadow-sm">
                    <div class="card-body py-2 px-3">
                        ${sender === "system" ? icon : ""}
                        ${safeText}
                        ${sender === "user" ? icon : ""}
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
        let content = "";

        if (data.analysis) {
            content += `<div class="mb-3">
                <div class="d-flex align-items-center mb-2">
                    <i class="fas fa-search-plus text-warning me-2"></i>
                    <strong>系统认知分析:</strong>
                </div>
                <div class="card bg-light border-0">
                    <div class="card-body p-2" style="font-size: 0.9em;">
                        <div class="mb-2">
                            <small class="text-muted d-block mb-1">文本分词与实体识别</small>
                            ${renderSegmentation(data.analysis)}
                        </div>
                        <div>
                            <small class="text-muted d-block mb-1">解析意图（四元组）</small>
                            ${renderStructuredQuery(data.analysis.structured_query)}
                        </div>
                    </div>
                </div>
            </div>`;
        }

        const graphTitle = currentMode === "internal" ? "知识库查询结果" : "知识库参考结果（用于对比）";
        content += `<strong><i class="fas fa-database text-success"></i> ${graphTitle}:</strong><br>`;
        if (data.graph_result && data.graph_result.length > 0) {
            const safeGraphResult = data.graph_result.map(function (item) {
                return escapeHtml(item);
            }).join(", ");
            content += `<div class="alert alert-success mt-2">${safeGraphResult}</div>`;
        } else {
            content += `<div class="alert alert-light border mt-2">${escapeHtml(data.graph_message || "暂无数据")}</div>`;
        }

        if (data.reasoning_result && data.reasoning_result.length > 0) {
            const title = currentMode === "internal"
                ? `内推模型预测结果（Top ${settings.topK}）`
                : `TiRGN 外推预测结果（Top ${settings.topK}）`;
            content += `<div class="mt-3"><strong><i class="fas fa-brain text-info"></i> ${title}:</strong></div>`;
            content += '<div class="list-group mt-2">';

            const minScore = currentMode === "internal" ? settings.internalMinScore : settings.externalMinScore;
            const filtered = data.reasoning_result.filter(function (item) {
                // Error results must always be shown, regardless of score threshold.
                var src = (item.source || "").toLowerCase();
                if (src.indexOf("error") !== -1) {
                    return true;
                }
                const score = item.score ?? item.probability;
                return typeof score === "number" ? score >= minScore : true;
            });
            const topLimit = settings.topK > 0 ? settings.topK : filtered.length;
            const limited = filtered.slice(0, topLimit);

            if (limited.length === 0) {
                content += `<div class="list-group-item text-muted">没有满足阈值 ${escapeHtml(minScore)} 的预测结果。</div>`;
            } else {
                limited.forEach(function (item, index) {
                    const badgeClass = index === 0 ? "bg-danger" : "bg-secondary";
                    const rawScore = item.score ?? item.probability;
                    const scoreText = typeof rawScore === "number" ? rawScore.toFixed(2) : String(rawScore ?? "");
                    const answerName = escapeHtml(item.name || item.prediction || "");
                    content += `
                        <div class="list-group-item d-flex justify-content-between align-items-center">
                            <div class="answer-name">${answerName}</div>
                            <span class="badge ${badgeClass} rounded-pill">${escapeHtml(scoreText)}</span>
                        </div>
                    `;
                });
            }

            content += "</div>";
        }

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
        let html = "";
        const matches = analysis.ac_matches || [];

        (analysis.segmentation || []).forEach(function (seg) {
            let badgeClass = "bg-secondary bg-opacity-10 text-dark";
            const title = escapeHtml(seg.flag || "");
            const match = matches.find(function (item) {
                return item.word === seg.word;
            });

            if (match) {
                if (match.type === "ENTITY") {
                    badgeClass = "bg-primary text-white";
                } else if (match.type === "RELATION") {
                    badgeClass = "bg-success text-white";
                } else if (match.type === "TIME") {
                    badgeClass = "bg-info text-white";
                }
            }

            html += `<span class="badge ${badgeClass} me-1 mb-1" title="${title}">${escapeHtml(seg.word)}</span>`;
        });

        return html;
    }

    function renderStructuredQuery(q) {
        if (!q) {
            return '<span class="text-muted">无法解析</span>';
        }

        const h = q.h ? `<span class="badge bg-primary">${escapeHtml(q.h)}</span>` : '<span class="text-muted">?</span>';
        const r = q.r ? `<span class="badge bg-success">${escapeHtml(q.r)}</span>` : '<span class="text-muted">?</span>';
        const t = q.t ? `<span class="badge bg-primary">${escapeHtml(q.t)}</span>` : '<span class="text-muted">?</span>';
        const time = q.time ? `<span class="badge bg-info">${escapeHtml(q.time)}</span>` : '<span class="text-muted">?</span>';

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
