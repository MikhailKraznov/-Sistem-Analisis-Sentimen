// ==================== CONFIGURATION ====================

const API_BASE_URL = 'http://127.0.0.1:5000/api';

// ==================== STATE MANAGEMENT ====================

let currentAnalysis = null;

// ==================== LANDING PAGE ====================

if (window.location.pathname.includes('index.html') || window.location.pathname === '/') {
    document.addEventListener('DOMContentLoaded', () => {
        const analyzeBtn = document.getElementById('analyzeBtn');
        const videoUrlInput = document.getElementById('videoUrl');
        const loadingSection = document.getElementById('loadingSection');
        const errorSection = document.getElementById('errorSection');
        const errorText = document.getElementById('errorText');
        const retryBtn = document.getElementById('retryBtn');

        // Analyze button click
        analyzeBtn.addEventListener('click', () => {
            const url = videoUrlInput.value.trim();
            if (!url) {
                showError('Please enter a YouTube video URL');
                return;
            }
            analyzeVideo(url);
        });

        // Enter key to analyze
        videoUrlInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                analyzeBtn.click();
            }
        });

        // Retry button
        retryBtn.addEventListener('click', () => {
            errorSection.style.display = 'none';
            videoUrlInput.focus();
        });

        // Analyze video function
        async function analyzeVideo(url) {
            try {
                // Show loading
                loadingSection.style.display = 'block';
                errorSection.style.display = 'none';
                analyzeBtn.disabled = true;

                // Call API
                const response = await fetch(`${API_BASE_URL}/analyze`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ url: url })
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || 'Analysis failed');
                }

                // Store analysis data
                localStorage.setItem('analysisData', JSON.stringify(data));

                // Redirect to dashboard
                window.location.href = 'dashboard.html';

            } catch (error) {
                console.error('Error:', error);
                showError(error.message || 'An error occurred during analysis. Please try again.');
                loadingSection.style.display = 'none';
                analyzeBtn.disabled = false;
            }
        }

        function showError(message) {
            errorText.textContent = message;
            errorSection.style.display = 'block';
        }
    });
}

// ==================== DASHBOARD PAGE ====================

if (window.location.pathname.includes('dashboard.html')) {
    document.addEventListener('DOMContentLoaded', () => {
        // Check for analysis data
        const analysisDataStr = localStorage.getItem('analysisData');
        
        if (!analysisDataStr) {
            alert('No analysis data found. Redirecting to home page.');
            window.location.href = 'index.html';
            return;
        }

        try {
            currentAnalysis = JSON.parse(analysisDataStr);
            initializeDashboard();
        } catch (error) {
            console.error('Error parsing analysis data:', error);
            alert('Error loading analysis data. Redirecting to home page.');
            window.location.href = 'index.html';
        }
    });

    function initializeDashboard() {
        // Set up navigation
        setupNavigation();

        // Set up new analysis button
        document.getElementById('newAnalysisBtn').addEventListener('click', () => {
            if (confirm('Start a new analysis? Current data will be lost.')) {
                localStorage.removeItem('analysisData');
                window.location.href = 'index.html';
            }
        });

        // Set up export buttons
        document.getElementById('exportCsvBtn').addEventListener('click', exportCSV);
        document.getElementById('exportJsonBtn').addEventListener('click', exportJSON);

        // Populate dashboard
        populateVideoInfo();
        populateOverview();
        populateSentimentSection();
        populateTrafficSection();
        populateDemographicsSection();
        populateCommentsSection();
    }

    function setupNavigation() {
        const navItems = document.querySelectorAll('.nav-item');
        const sections = document.querySelectorAll('.dashboard-section');

        navItems.forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                
                // Update active nav item
                navItems.forEach(nav => nav.classList.remove('active'));
                item.classList.add('active');

                // Show corresponding section
                const sectionId = item.dataset.section + 'Section';
                sections.forEach(section => section.classList.remove('active'));
                document.getElementById(sectionId).classList.add('active');
            });
        });
    }

    function populateVideoInfo() {
        const videoInfo = currentAnalysis.video_info;
        document.getElementById('videoTitle').textContent = videoInfo.title;
        document.getElementById('videoChannel').textContent = `Channel: ${videoInfo.channel}`;
    }

    function populateOverview() {
        const overview = currentAnalysis.overview;
        const sentiment = currentAnalysis.sentiment_distribution;

    // Use the ACTUAL number of comments sent to frontend
        const actualTotal = currentAnalysis.all_comments.length;

        document.getElementById('totalComments').textContent = actualTotal;
        document.getElementById('positiveCount').textContent = sentiment.positive;
        document.getElementById('negativeCount').textContent = sentiment.negative;
        document.getElementById('neutralCount').textContent = sentiment.neutral;

        const summaryText = `This analysis used XLM-RoBERTa model to classify ${actualTotal} comments with ${Math.round(overview.average_confidence * 100)}% average confidence. The comments were detected in ${overview.languages_detected} different languages, showcasing strong international engagement.`;
    
        document.getElementById('summaryText').textContent = summaryText;
    }

    function populateSentimentSection() {
        const sentiment = currentAnalysis.sentiment_distribution;
        const avgConfidence = currentAnalysis.overview.average_confidence;

        document.getElementById('avgConfidence').textContent = `${Math.round(avgConfidence * 100)}%`;

        // Create pie chart
        const ctx = document.getElementById('sentimentChart').getContext('2d');
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Positive', 'Neutral', 'Negative'],
                datasets: [{
                    data: [sentiment.positive, sentiment.neutral, sentiment.negative],
                    backgroundColor: [
                        '#27ae60',  // Green for positive
                        '#f39c12',  // Orange for neutral
                        '#e74c3c'   // Red for negative
                    ],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            font: {
                                size: 14
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.parsed || 0;
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = Math.round((value / total) * 100);
                                return `${label}: ${value} (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });
    }

    function populateTrafficSection() {
        const traffic = currentAnalysis.weekly_traffic;

        if (!traffic || traffic.length === 0) {
            document.getElementById('trafficNote').textContent = 'No traffic data available.';
            return;
        }

        // Determine if it's daily or weekly data
        const isDaily = traffic[0].period.includes('-');
        const note = isDaily 
            ? 'Peak engagement occurred in the first day with natural decline over subsequent days.'
            : 'Peak engagement occurred in Week 1 with natural decline over subsequent weeks.';
        
        document.getElementById('trafficNote').textContent = note;

        // Create bar chart
        const ctx = document.getElementById('trafficChart').getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: traffic.map(t => t.period),
                datasets: [{
                    label: 'Comments',
                    data: traffic.map(t => t.count),
                    backgroundColor: '#3498db',
                    borderColor: '#2980b9',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
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

    function populateDemographicsSection() {
        const languages = currentAnalysis.language_distribution;
        const languageList = document.getElementById('languageList');
        languageList.innerHTML = '';

        languages.forEach(lang => {
            const item = document.createElement('div');
            item.className = 'language-item';
            
            item.innerHTML = `
                <div class="language-name">${capitalizeFirst(lang.language)}</div>
                <div class="language-bar-container">
                    <div class="language-bar" style="width: ${lang.percentage}%">
                        ${lang.percentage}%
                    </div>
                </div>
            `;
            
            languageList.appendChild(item);
        });
    }

    function populateCommentsSection() {
    // Use ALL comments, not just top_comments
        const comments = currentAnalysis.all_comments || currentAnalysis.top_comments;
        const commentsList = document.getElementById('commentsList');
        commentsList.innerHTML = '';

        if (!comments || comments.length === 0) {
        commentsList.innerHTML = '<p>No comments available.</p>';
            return;
        }

    // Store all comments globally for filtering
        window.allComments = comments;

    // Render all comments initially
        renderComments(comments);

    // Setup filter buttons
        setupCommentFilters();
    }

function renderComments(commentsToRender) {
    const commentsList = document.getElementById('commentsList');
    commentsList.innerHTML = '';

    commentsToRender.forEach(comment => {
        const item = document.createElement('div');
        item.className = `comment-item ${comment.sentiment}`;
        item.setAttribute('data-sentiment', comment.sentiment);
        item.setAttribute('data-likes', comment.likes);

        const publishedDate = new Date(comment.published_at).toLocaleDateString();

        item.innerHTML = `
            <div class="comment-header">
                <span class="comment-author">${escapeHtml(comment.author)}</span>
                <span class="sentiment-badge ${comment.sentiment}">${comment.sentiment}</span>
            </div>
            <div class="comment-text">${escapeHtml(comment.text)}</div>
            <div class="comment-footer">
                <span>👍 ${comment.likes} likes</span>
                <span>🌐 ${comment.language_name || capitalizeFirst(comment.language)}</span>
                <span>📅 ${publishedDate}</span>
                <span>✨ ${Math.round(comment.confidence * 100)}% confidence</span>
            </div>
        `;

        commentsList.appendChild(item);
    });
}

function setupCommentFilters() {
    const filterButtons = document.querySelectorAll('.filter-btn');
    const allComments = window.allComments;

    filterButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Update active button
            filterButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

            // Get filter type
            const filter = button.getAttribute('data-filter');

            // Filter comments
            let filteredComments = [];

            if (filter === 'all') {
                filteredComments = allComments;
            } else if (filter === 'positive') {
                filteredComments = allComments.filter(c => c.sentiment === 'positive');
            } else if (filter === 'negative') {
                filteredComments = allComments.filter(c => c.sentiment === 'negative');
            } else if (filter === 'neutral') {
                filteredComments = allComments.filter(c => c.sentiment === 'neutral');
            } else if (filter === 'relevant') {
                // Sort by likes (most relevant = most liked)
                filteredComments = [...allComments].sort((a, b) => b.likes - a.likes);
            }

            // Re-render comments
            renderComments(filteredComments);

            // Show count
            console.log(`Showing ${filteredComments.length} ${filter} comments`);
        });
    });
}

    async function exportCSV() {
        try {
            const btn = document.getElementById('exportCsvBtn');
            btn.disabled = true;
            btn.textContent = 'Exporting...';

            const response = await fetch(`${API_BASE_URL}/export/csv`);
            
            if (!response.ok) {
                throw new Error('Export failed');
            }

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `youtube_analysis_${Date.now()}.csv`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

            btn.disabled = false;
            btn.textContent = 'Export CSV';
        } catch (error) {
            console.error('Export error:', error);
            alert('Export failed. Please try again.');
            document.getElementById('exportCsvBtn').disabled = false;
            document.getElementById('exportCsvBtn').textContent = 'Export CSV';
        }
    }

    async function exportJSON() {
        try {
            const btn = document.getElementById('exportJsonBtn');
            btn.disabled = true;
            btn.textContent = 'Exporting...';

            const response = await fetch(`${API_BASE_URL}/export/json`);
            
            if (!response.ok) {
                throw new Error('Export failed');
            }

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `youtube_analysis_${Date.now()}.json`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

            btn.disabled = false;
            btn.textContent = 'Export JSON';
        } catch (error) {
            console.error('Export error:', error);
            alert('Export failed. Please try again.');
            document.getElementById('exportJsonBtn').disabled = false;
            document.getElementById('exportJsonBtn').textContent = 'Export JSON';
        }
    }

    // Utility functions
    function capitalizeFirst(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}