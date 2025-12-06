// API Configuration
const API_URL = '';

// Chart instances
let radarChart = null;
let performanceChart = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function () {
    initializeCharts();
    loadMetrics();
    setupEventListeners();
});

// Event Listeners
function setupEventListeners() {
    document.getElementById('run-eval-btn').addEventListener('click', runEvaluation);
    document.getElementById('refresh-btn').addEventListener('click', loadMetrics);
    document.getElementById('export-btn').addEventListener('click', exportResults);
}

// Initialize Charts
function initializeCharts() {
    // Radar Chart for Quality Metrics
    const radarCtx = document.getElementById('radarChart').getContext('2d');
    radarChart = new Chart(radarCtx, {
        type: 'radar',
        data: {
            labels: ['Context Precision', 'Context Recall', 'Faithfulness', 'Answer Relevancy'],
            datasets: [{
                label: 'Quality Metrics',
                data: [0, 0, 0, 0],
                backgroundColor: 'rgba(102, 126, 234, 0.2)',
                borderColor: 'rgba(102, 126, 234, 1)',
                borderWidth: 2,
                pointBackgroundColor: 'rgba(102, 126, 234, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(102, 126, 234, 1)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        stepSize: 0.2,
                        color: '#9aa0a6'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    pointLabels: {
                        color: '#e8eaed',
                        font: {
                            size: 12,
                            weight: '500'
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });

    // Bar Chart for Performance Metrics
    const perfCtx = document.getElementById('performanceChart').getContext('2d');
    performanceChart = new Chart(perfCtx, {
        type: 'bar',
        data: {
            labels: ['Avg Retrieval Time', 'Avg Generation Time'],
            datasets: [{
                label: 'Time (seconds)',
                data: [0, 0],
                backgroundColor: [
                    'rgba(77, 171, 247, 0.8)',
                    'rgba(167, 139, 250, 0.8)'
                ],
                borderColor: [
                    'rgba(77, 171, 247, 1)',
                    'rgba(167, 139, 250, 1)'
                ],
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: '#9aa0a6'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                x: {
                    ticks: {
                        color: '#e8eaed'
                    },
                    grid: {
                        display: false
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

// Load Metrics from API
async function loadMetrics() {
    try {
        showStatus('Loading metrics...', 'info');

        const response = await fetch(`${API_URL}/evaluate/metrics`);
        const data = await response.json();

        if (data.message) {
            showStatus(data.message, 'info');
            return;
        }

        if (data.latest_evaluation) {
            updateMetricsDisplay(data.latest_evaluation.aggregate_metrics);
            updateCharts(data.latest_evaluation.aggregate_metrics);
            updateResultsTable(data.latest_evaluation.individual_results);
            updateTrends(data.trends);
            showStatus('Metrics loaded successfully', 'success');
            setTimeout(() => hideStatus(), 3000);
        } else {
            showStatus('No evaluation data available. Run an evaluation to see results.', 'info');
        }
    } catch (error) {
        console.error('Error loading metrics:', error);
        showStatus('Failed to load metrics. Make sure the backend is running.', 'error');
    }
}

// Run Evaluation
async function runEvaluation() {
    const btn = document.getElementById('run-eval-btn');
    const originalText = btn.innerHTML;

    try {
        // Disable button and show loading
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Running...';
        showStatus('Running evaluation... This may take a few minutes.', 'info');

        const response = await fetch(`${API_URL}/evaluate`, {
            method: 'POST'
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Evaluation failed');
        }

        if (data.success && data.results) {
            updateMetricsDisplay(data.results.aggregate_metrics);
            updateCharts(data.results.aggregate_metrics);
            updateResultsTable(data.results.individual_results);
            showStatus(`Evaluation completed! Processed ${data.results.num_test_cases} test cases.`, 'success');
        } else {
            showStatus(data.message || 'Evaluation completed with no results', 'info');
        }
    } catch (error) {
        console.error('Error running evaluation:', error);
        showStatus(`Error: ${error.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

// Update Metrics Display
function updateMetricsDisplay(metrics) {
    document.getElementById('context-precision-value').textContent =
        metrics.context_precision ? metrics.context_precision.toFixed(3) : '--';
    document.getElementById('context-recall-value').textContent =
        metrics.context_recall ? metrics.context_recall.toFixed(3) : '--';
    document.getElementById('faithfulness-value').textContent =
        metrics.faithfulness ? metrics.faithfulness.toFixed(3) : '--';
    document.getElementById('answer-relevancy-value').textContent =
        metrics.answer_relevancy ? metrics.answer_relevancy.toFixed(3) : '--';
}

// Update Charts
function updateCharts(metrics) {
    // Update Radar Chart
    radarChart.data.datasets[0].data = [
        metrics.context_precision || 0,
        metrics.context_recall || 0,
        metrics.faithfulness || 0,
        metrics.answer_relevancy || 0
    ];
    radarChart.update();

    // Update Performance Chart
    performanceChart.data.datasets[0].data = [
        metrics.avg_retrieval_time || 0,
        metrics.avg_generation_time || 0
    ];
    performanceChart.update();
}

// Update Trends
function updateTrends(trends) {
    if (!trends) return;

    const metricIds = ['context-precision', 'context-recall', 'faithfulness', 'answer-relevancy'];
    const metricKeys = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy'];

    metricIds.forEach((id, index) => {
        const trendElement = document.getElementById(`${id}-trend`);
        const trend = trends[metricKeys[index]];

        if (trend) {
            const icon = trend.direction === 'up' ? '↑' : trend.direction === 'down' ? '↓' : '→';
            trendElement.innerHTML = `${icon} ${Math.abs(trend.change_percent).toFixed(1)}%`;
            trendElement.className = `metric-trend trend-${trend.direction}`;
        }
    });
}

// Update Results Table
function updateResultsTable(results) {
    const tbody = document.getElementById('results-tbody');

    if (!results || results.length === 0) {
        tbody.innerHTML = `
            <tr class="no-data">
                <td colspan="8">
                    <div class="no-data-message">
                        <i class="fas fa-inbox"></i>
                        <p>No evaluation results</p>
                    </div>
                </td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = results.map((result, index) => `
        <tr>
            <td style="max-width: 300px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" 
                title="${escapeHtml(result.question)}">
                ${escapeHtml(result.question)}
            </td>
            <td>${getScoreBadge(result.context_precision)}</td>
            <td>${getScoreBadge(result.context_recall)}</td>
            <td>${getScoreBadge(result.faithfulness)}</td>
            <td>${getScoreBadge(result.answer_relevancy)}</td>
            <td>${result.retrieval_time ? result.retrieval_time.toFixed(2) : 'N/A'}</td>
            <td>${result.generation_time ? result.generation_time.toFixed(2) : 'N/A'}</td>
            <td>
                <button class="action-btn" onclick='showDetails(${JSON.stringify(result).replace(/'/g, "&#39;")})'>
                    <i class="fas fa-eye"></i> View
                </button>
            </td>
        </tr>
    `).join('');
}

// Get Score Badge
function getScoreBadge(score) {
    if (score === null || score === undefined) {
        return '<span class="score-badge">N/A</span>';
    }

    const value = parseFloat(score);
    let className = 'score-poor';

    if (value >= 0.8) className = 'score-excellent';
    else if (value >= 0.6) className = 'score-good';
    else if (value >= 0.4) className = 'score-fair';

    return `<span class="score-badge ${className}">${value.toFixed(3)}</span>`;
}

// Show Details Modal
function showDetails(result) {
    const modal = document.getElementById('details-modal');
    const modalBody = document.getElementById('modal-body');

    modalBody.innerHTML = `
        <div class="detail-section">
            <h3>Question</h3>
            <div class="detail-content">${escapeHtml(result.question)}</div>
        </div>
        <div class="detail-section">
            <h3>Generated Answer</h3>
            <div class="detail-content">${escapeHtml(result.answer)}</div>
        </div>
        <div class="detail-section">
            <h3>Ground Truth</h3>
            <div class="detail-content">${escapeHtml(result.ground_truth || 'N/A')}</div>
        </div>
        <div class="detail-section">
            <h3>Retrieved Contexts (${result.contexts ? result.contexts.length : 0})</h3>
            <div class="detail-content">
                ${result.contexts ? result.contexts.map((ctx, i) =>
        `<strong>[${i + 1}]</strong> ${escapeHtml(ctx)}`
    ).join('\n\n') : 'No contexts'}
            </div>
        </div>
        <div class="detail-section">
            <h3>Metrics</h3>
            <div class="detail-content">
                Context Precision: ${getScoreBadge(result.context_precision)}
                Context Recall: ${getScoreBadge(result.context_recall)}
                Faithfulness: ${getScoreBadge(result.faithfulness)}
                Answer Relevancy: ${getScoreBadge(result.answer_relevancy)}
                
                Retrieval Time: ${result.retrieval_time ? result.retrieval_time.toFixed(2) + 's' : 'N/A'}
                Generation Time: ${result.generation_time ? result.generation_time.toFixed(2) + 's' : 'N/A'}
            </div>
        </div>
    `;

    modal.classList.add('show');
}

// Close Modal
function closeModal() {
    document.getElementById('details-modal').classList.remove('show');
}

// Export Results
async function exportResults() {
    try {
        const response = await fetch(`${API_URL}/evaluate/metrics`);
        const data = await response.json();

        if (!data.latest_evaluation) {
            alert('No evaluation results to export');
            return;
        }

        const blob = new Blob([JSON.stringify(data.latest_evaluation, null, 2)], {
            type: 'application/json'
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `evaluation_results_${new Date().toISOString()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        showStatus('Results exported successfully', 'success');
        setTimeout(() => hideStatus(), 3000);
    } catch (error) {
        console.error('Error exporting results:', error);
        showStatus('Failed to export results', 'error');
    }
}

// Status Banner
function showStatus(message, type) {
    const banner = document.getElementById('evaluation-status');
    banner.textContent = message;
    banner.className = `status-banner ${type}`;
}

function hideStatus() {
    const banner = document.getElementById('evaluation-status');
    banner.style.display = 'none';
}

// Utility Functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Close modal on outside click
document.addEventListener('click', function (e) {
    const modal = document.getElementById('details-modal');
    if (e.target === modal) {
        closeModal();
    }
});

// Close modal on Escape key
document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') {
        closeModal();
    }
});
