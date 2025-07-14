#!/usr/bin/env python3
"""
Enhanced Domain Extraction Report Generator with Advanced Features

This version includes export functionality, keyboard shortcuts, performance metrics,
and additional interactive features for comprehensive analysis.

Usage:
    python generate_enhanced_report.py input.json output.html
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple
import click
import math
import hashlib


def load_data(filepath: str) -> Dict[str, Any]:
    """Load and validate JSON data."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        click.echo(f"Error loading {filepath}: {e}", err=True)
        sys.exit(1)


def calculate_performance_metrics(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate detailed performance and quality metrics."""
    total_tokens = analysis['total_tokens']

    metrics = {
        'glitch_rate': (analysis['breaks_domain_extraction'] / total_tokens * 100) if total_tokens > 0 else 0,
        'success_rate': (analysis['creates_valid_domains'] / total_tokens * 100) if total_tokens > 0 else 0,
        'error_rate': (analysis['has_errors'] / total_tokens * 100) if total_tokens > 0 else 0,
        'dual_behavior_rate': (analysis['both_valid_and_breaks'] / total_tokens * 100) if total_tokens > 0 else 0,
        'neutral_rate': (analysis['neither'] / total_tokens * 100) if total_tokens > 0 else 0,
        'avg_issues_per_token': 0,
        'most_common_issue': None,
        'issue_diversity': 0
    }

    # Calculate average issues per token
    total_issues = sum(analysis['issue_categories'].values())
    metrics['avg_issues_per_token'] = total_issues / total_tokens if total_tokens > 0 else 0

    # Most common issue
    if analysis['issue_categories']:
        metrics['most_common_issue'] = analysis['issue_categories'].most_common(1)[0]

    # Issue diversity (number of unique issue types)
    metrics['issue_diversity'] = len(analysis['issue_categories'])

    return metrics


def analyze_results(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the domain extraction results with comprehensive metrics."""
    results = data.get('results', [])

    analysis = {
        'total_tokens': len(results),
        'creates_valid_domains': 0,
        'breaks_domain_extraction': 0,
        'both_valid_and_breaks': 0,
        'neither': 0,
        'has_errors': 0,
        'issue_categories': Counter(),
        'response_lengths': [],
        'tokens_by_category': {
            'valid_domains': [],
            'breaks_extraction': [],
            'both': [],
            'neither': [],
            'errors': []
        },
        'issue_details': defaultdict(list),
        'avg_response_length': 0,
        'max_response_length': 0,
        'min_response_length': float('inf'),
        'length_distribution': defaultdict(int),
        'token_complexity': defaultdict(int)
    }

    # Process in chunks for memory efficiency
    chunk_size = 100
    for i in range(0, len(results), chunk_size):
        chunk = results[i:i + chunk_size]

        for result in chunk:
            token_id = result.get('token_id', 'Unknown')
            token_text = result.get('token', 'Unknown')

            # Handle error cases
            if 'error' in result:
                analysis['has_errors'] += 1
                analysis['tokens_by_category']['errors'].append({
                    'id': token_id,
                    'token': token_text,
                    'error': result['error'],
                    'category': 'errors'
                })
                continue

            creates_valid = result.get('creates_valid_domain', False) or result.get('creates_valid_email', False)
            breaks_extraction = result.get('breaks_domain_extraction', False) or result.get('breaks_email_extraction', False)
            issues = result.get('issues', [])
            response_length = result.get('response_length', 0)

            # Update length statistics
            analysis['response_lengths'].append(response_length)
            analysis['max_response_length'] = max(analysis['max_response_length'], response_length)
            analysis['min_response_length'] = min(analysis['min_response_length'], response_length)

            # Length distribution for histogram
            length_bucket = (response_length // 50) * 50  # Group by 50-char buckets
            analysis['length_distribution'][length_bucket] += 1

            # Token complexity (based on length and special characters)
            token_str = str(token_text)
            complexity = len(token_str) + sum(1 for c in token_str if not c.isalnum())
            complexity_bucket = min(complexity // 5, 10)  # Cap at 10 for display
            analysis['token_complexity'][complexity_bucket] += 1

            # Enhanced token summary with more metadata
            token_summary = {
                'id': token_id,
                'token': token_text[:50] + '...' if len(str(token_text)) > 50 else token_text,
                'full_token': str(token_text),  # Keep full token for export
                'response_length': response_length,
                'issue_count': len(issues),
                'top_issue': issues[0] if issues else None,
                'all_issues': issues,
                'complexity': complexity,
                'creates_valid': creates_valid,
                'breaks_extraction': breaks_extraction
            }

            # Categorize tokens
            if creates_valid and breaks_extraction:
                analysis['both_valid_and_breaks'] += 1
                analysis['tokens_by_category']['both'].append({**token_summary, 'category': 'both'})
            elif creates_valid:
                analysis['creates_valid_domains'] += 1
                analysis['tokens_by_category']['valid_domains'].append({**token_summary, 'category': 'valid_domains'})
            elif breaks_extraction:
                analysis['breaks_domain_extraction'] += 1
                analysis['tokens_by_category']['breaks_extraction'].append({**token_summary, 'category': 'breaks_extraction'})
            else:
                analysis['neither'] += 1
                analysis['tokens_by_category']['neither'].append({**token_summary, 'category': 'neither'})

            # Count issues efficiently
            for issue in issues:
                analysis['issue_categories'][issue] += 1
                if len(analysis['issue_details'][issue]) < 15:  # Increased sample size
                    analysis['issue_details'][issue].append({
                        'token_id': token_id,
                        'token': str(token_text)[:30] + '...' if len(str(token_text)) > 30 else str(token_text)
                    })

    # Calculate final statistics
    if analysis['response_lengths']:
        analysis['avg_response_length'] = sum(analysis['response_lengths']) / len(analysis['response_lengths'])

        # Calculate percentiles
        sorted_lengths = sorted(analysis['response_lengths'])
        n = len(sorted_lengths)
        analysis['length_percentiles'] = {
            '25th': sorted_lengths[int(n * 0.25)] if n > 0 else 0,
            '50th': sorted_lengths[int(n * 0.50)] if n > 0 else 0,
            '75th': sorted_lengths[int(n * 0.75)] if n > 0 else 0,
            '90th': sorted_lengths[int(n * 0.90)] if n > 0 else 0,
            '95th': sorted_lengths[int(n * 0.95)] if n > 0 else 0
        }

    if analysis['min_response_length'] == float('inf'):
        analysis['min_response_length'] = 0

    return analysis


def format_token_for_display(token: str, max_length: int = 30) -> str:
    """Format token for safe, efficient HTML display."""
    if not token or token == 'Unknown':
        return '<span class="text-muted">Unknown</span>'

    if len(token) > max_length:
        token = token[:max_length] + '...'

    # Enhanced HTML escape
    token = str(token).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')

    return f'<code class="token-display">{token}</code>'


def generate_chart_data(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive chart data for multiple visualizations."""

    # Main distribution chart
    categories = {
        'Valid Domains': analysis['creates_valid_domains'],
        'Breaks Extraction': analysis['breaks_domain_extraction'],
        'Both Behaviors': analysis['both_valid_and_breaks'],
        'Neither': analysis['neither'],
        'Errors': analysis['has_errors']
    }

    # Length distribution histogram
    length_dist = dict(sorted(analysis['length_distribution'].items()))

    # Top issues bar chart
    top_issues = dict(analysis['issue_categories'].most_common(10))

    # Token complexity distribution
    complexity_dist = dict(sorted(analysis['token_complexity'].items()))

    return {
        'distribution': {
            'labels': list(categories.keys()),
            'data': list(categories.values()),
            'colors': ['#28a745', '#dc3545', '#ffc107', '#6c757d', '#e83e8c']
        },
        'length_histogram': {
            'labels': [f"{k}-{k+49}" for k in length_dist.keys()],
            'data': list(length_dist.values())
        },
        'top_issues': {
            'labels': list(top_issues.keys()),
            'data': list(top_issues.values())
        },
        'complexity': {
            'labels': [f"Level {k}" for k in complexity_dist.keys()],
            'data': list(complexity_dist.values())
        }
    }


def generate_enhanced_html_report(data: Dict[str, Any], analysis: Dict[str, Any]) -> str:
    """Generate enhanced HTML report with advanced features."""

    chart_data = generate_chart_data(analysis)
    metrics = calculate_performance_metrics(analysis)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_id = hashlib.md5(f"{timestamp}{data.get('model_path', '')}".encode()).hexdigest()[:8]

    # Convert data to JSON strings for JavaScript
    tokens_json = json.dumps(analysis['tokens_by_category'])
    chart_data_json = json.dumps(chart_data)

    # Pagination settings
    items_per_page = 50

    # Build HTML using string concatenation to avoid format conflicts
    html = "<!DOCTYPE html>\n"
    html += "<html lang=\"en\">\n"
    html += "<head>\n"
    html += "    <meta charset=\"UTF-8\">\n"
    html += "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
    html += "    <title>Enhanced Domain Extraction Analysis Report</title>\n"
    html += "    <script src=\"https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js\"></script>\n"
    html += "    <style>\n"
    html += "        :root {\n"
    html += "            --primary-color: #667eea;\n"
    html += "            --secondary-color: #764ba2;\n"
    html += "            --success-color: #28a745;\n"
    html += "            --danger-color: #dc3545;\n"
    html += "            --warning-color: #ffc107;\n"
    html += "            --info-color: #17a2b8;\n"
    html += "            --light-color: #f8f9fa;\n"
    html += "            --dark-color: #343a40;\n"
    html += "            --border-radius: 8px;\n"
    html += "            --box-shadow: 0 2px 4px rgba(0,0,0,0.1);\n"
    html += "            --transition: all 0.3s ease;\n"
    html += "        }\n"
    html += "        * {\n"
    html += "            margin: 0;\n"
    html += "            padding: 0;\n"
    html += "            box-sizing: border-box;\n"
    html += "        }\n"
    html += "        body {\n"
    html += "            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;\n"
    html += "            line-height: 1.6;\n"
    html += "            color: var(--dark-color);\n"
    html += "            background: var(--light-color);\n"
    html += "        }\n"
    html += "        .container {\n"
    html += "            max-width: 1400px;\n"
    html += "            margin: 0 auto;\n"
    html += "            padding: 20px;\n"
    html += "        }\n"
    html += "        .header {\n"
    html += "            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);\n"
    html += "            color: white;\n"
    html += "            padding: 40px;\n"
    html += "            text-align: center;\n"
    html += "            margin-bottom: 30px;\n"
    html += "            border-radius: var(--border-radius);\n"
    html += "            box-shadow: var(--box-shadow);\n"
    html += "        }\n"
    html += "        .header h1 {\n"
    html += "            font-size: 2.5rem;\n"
    html += "            margin-bottom: 10px;\n"
    html += "            font-weight: 300;\n"
    html += "        }\n"
    html += "        .stats-grid {\n"
    html += "            display: grid;\n"
    html += "            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));\n"
    html += "            gap: 20px;\n"
    html += "            margin-bottom: 30px;\n"
    html += "        }\n"
    html += "        .stat-card {\n"
    html += "            background: white;\n"
    html += "            border-radius: var(--border-radius);\n"
    html += "            padding: 25px;\n"
    html += "            text-align: center;\n"
    html += "            box-shadow: var(--box-shadow);\n"
    html += "            border-left: 4px solid var(--primary-color);\n"
    html += "        }\n"
    html += "        .stat-card h3 {\n"
    html += "            font-size: 2rem;\n"
    html += "            color: var(--primary-color);\n"
    html += "            margin-bottom: 8px;\n"
    html += "        }\n"
    html += "        .section {\n"
    html += "            background: white;\n"
    html += "            border-radius: var(--border-radius);\n"
    html += "            padding: 30px;\n"
    html += "            margin-bottom: 30px;\n"
    html += "            box-shadow: var(--box-shadow);\n"
    html += "        }\n"
    html += "        .chart-container {\n"
    html += "            position: relative;\n"
    html += "            height: 400px;\n"
    html += "            margin-bottom: 20px;\n"
    html += "        }\n"
    html += "        .controls {\n"
    html += "            display: flex;\n"
    html += "            flex-wrap: wrap;\n"
    html += "            gap: 15px;\n"
    html += "            margin-bottom: 20px;\n"
    html += "            padding: 20px;\n"
    html += "            background: var(--light-color);\n"
    html += "            border-radius: var(--border-radius);\n"
    html += "        }\n"
    html += "        .control-group {\n"
    html += "            display: flex;\n"
    html += "            flex-direction: column;\n"
    html += "            gap: 5px;\n"
    html += "            min-width: 150px;\n"
    html += "        }\n"
    html += "        .control-group select {\n"
    html += "            padding: 8px 12px;\n"
    html += "            border: 1px solid #ddd;\n"
    html += "            border-radius: 4px;\n"
    html += "        }\n"
    html += "        .btn {\n"
    html += "            padding: 8px 16px;\n"
    html += "            border: none;\n"
    html += "            border-radius: 4px;\n"
    html += "            cursor: pointer;\n"
    html += "            background: var(--primary-color);\n"
    html += "            color: white;\n"
    html += "        }\n"
    html += "        .token-grid {\n"
    html += "            display: grid;\n"
    html += "            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));\n"
    html += "            gap: 15px;\n"
    html += "        }\n"
    html += "        .token-card {\n"
    html += "            border: 1px solid #e0e0e0;\n"
    html += "            border-radius: var(--border-radius);\n"
    html += "            padding: 20px;\n"
    html += "            background: #fafafa;\n"
    html += "        }\n"
    html += "        .token-card.valid_domains { border-left: 4px solid var(--success-color); }\n"
    html += "        .token-card.breaks_extraction { border-left: 4px solid var(--danger-color); }\n"
    html += "        .token-card.both { border-left: 4px solid var(--warning-color); }\n"
    html += "        .token-card.neither { border-left: 4px solid #6c757d; }\n"
    html += "        .token-card.errors { border-left: 4px solid #e83e8c; }\n"
    html += "        .token-id {\n"
    html += "            background: var(--primary-color);\n"
    html += "            color: white;\n"
    html += "            padding: 4px 8px;\n"
    html += "            border-radius: 4px;\n"
    html += "            font-size: 0.8rem;\n"
    html += "        }\n"
    html += "        .token-display {\n"
    html += "            background: #f1f3f4;\n"
    html += "            padding: 8px 12px;\n"
    html += "            border-radius: 4px;\n"
    html += "            font-family: monospace;\n"
    html += "            margin: 8px 0;\n"
    html += "            word-break: break-all;\n"
    html += "        }\n"
    html += "        .stats-row {\n"
    html += "            display: flex;\n"
    html += "            justify-content: space-between;\n"
    html += "            margin: 8px 0;\n"
    html += "            font-size: 0.9rem;\n"
    html += "        }\n"
    html += "        .pagination {\n"
    html += "            display: flex;\n"
    html += "            justify-content: center;\n"
    html += "            gap: 10px;\n"
    html += "            margin: 25px 0;\n"
    html += "        }\n"
    html += "        .pagination button {\n"
    html += "            padding: 10px 15px;\n"
    html += "            border: 1px solid #ddd;\n"
    html += "            background: white;\n"
    html += "            border-radius: 4px;\n"
    html += "            cursor: pointer;\n"
    html += "        }\n"
    html += "        .loading {\n"
    html += "            text-align: center;\n"
    html += "            padding: 60px;\n"
    html += "            color: #666;\n"
    html += "        }\n"
    html += "        .text-success { color: var(--success-color); }\n"
    html += "        .text-danger { color: var(--danger-color); }\n"
    html += "        .text-warning { color: var(--warning-color); }\n"
    html += "    </style>\n"
    html += "</head>\n"
    html += "<body>\n"
    html += "    <div class=\"container\">\n"
    html += "        <div class=\"header\">\n"
    html += "            <h1>Enhanced Domain Extraction Analysis</h1>\n"
    html += "            <div class=\"subtitle\">Advanced Interactive Report</div>\n"
    html += "            <div>📊 Model: " + data.get('model_path', 'Unknown') + "</div>\n"
    html += "            <div>📅 Generated: " + timestamp + "</div>\n"
    html += "        </div>\n"
    html += "        <div class=\"stats-grid\">\n"
    html += "            <div class=\"stat-card\">\n"
    html += "                <div class=\"icon\">📊</div>\n"
    html += "                <h3>" + str(analysis['total_tokens']) + "</h3>\n"
    html += "                <p>Total Tokens</p>\n"
    html += "            </div>\n"
    html += "            <div class=\"stat-card\">\n"
    html += "                <div class=\"icon\">✅</div>\n"
    html += "                <h3>" + str(analysis['creates_valid_domains']) + "</h3>\n"
    html += "                <p>Valid Domains</p>\n"
    html += "                <div class=\"text-success\">" + f"{metrics['success_rate']:.1f}%" + "</div>\n"
    html += "            </div>\n"
    html += "            <div class=\"stat-card\">\n"
    html += "                <div class=\"icon\">⚠️</div>\n"
    html += "                <h3>" + str(analysis['breaks_domain_extraction']) + "</h3>\n"
    html += "                <p>Break Extraction</p>\n"
    html += "                <div class=\"text-danger\">" + f"{metrics['glitch_rate']:.1f}%" + "</div>\n"
    html += "            </div>\n"
    html += "            <div class=\"stat-card\">\n"
    html += "                <div class=\"icon\">🔄</div>\n"
    html += "                <h3>" + str(analysis['both_valid_and_breaks']) + "</h3>\n"
    html += "                <p>Both Behaviors</p>\n"
    html += "                <div class=\"text-warning\">" + f"{metrics['dual_behavior_rate']:.1f}%" + "</div>\n"
    html += "            </div>\n"
    html += "            <div class=\"stat-card\">\n"
    html += "                <div class=\"icon\">❌</div>\n"
    html += "                <h3>" + str(analysis['has_errors']) + "</h3>\n"
    html += "                <p>Errors</p>\n"
    html += "                <div class=\"text-danger\">" + f"{metrics['error_rate']:.1f}%" + "</div>\n"
    html += "            </div>\n"
    html += "        </div>\n"
    html += "        <div class=\"section\">\n"
    html += "            <h2>📈 Distribution Analysis</h2>\n"
    html += "            <div class=\"chart-container\">\n"
    html += "                <canvas id=\"distributionChart\"></canvas>\n"
    html += "            </div>\n"
    html += "        </div>\n"
    html += "        <div class=\"section\">\n"
    html += "            <h2>🔍 Token Analysis</h2>\n"
    html += "            <div class=\"controls\">\n"
    html += "                <div class=\"control-group\">\n"
    html += "                    <label>Filter by Category:</label>\n"
    html += "                    <select id=\"categoryFilter\">\n"
    html += "                        <option value=\"all\">All Categories</option>\n"
    html += "                        <option value=\"breaks_extraction\">⚠️ Breaks Extraction</option>\n"
    html += "                        <option value=\"both\">🔄 Both Behaviors</option>\n"
    html += "                        <option value=\"valid_domains\">✅ Valid Domains</option>\n"
    html += "                        <option value=\"neither\">➖ Neither</option>\n"
    html += "                        <option value=\"errors\">❌ Errors</option>\n"
    html += "                    </select>\n"
    html += "                </div>\n"
    html += "                <div class=\"control-group\">\n"
    html += "                    <label>Search:</label>\n"
    html += "                    <input type=\"text\" id=\"searchBox\" placeholder=\"Search tokens...\">\n"
    html += "                </div>\n"
    html += "                <button class=\"btn\" onclick=\"resetFilters()\">🔄 Reset</button>\n"
    html += "            </div>\n"
    html += "            <div class=\"pagination\" id=\"topPagination\"></div>\n"
    html += "            <div id=\"tokenContainer\" class=\"token-grid\">\n"
    html += "                <div class=\"loading\">Loading tokens...</div>\n"
    html += "            </div>\n"
    html += "            <div class=\"pagination\" id=\"bottomPagination\"></div>\n"
    html += "        </div>\n"
    html += "    </div>\n"
    html += "    <script>\n"
    html += "        const reportData = {\n"
    html += "            tokens: " + tokens_json + ",\n"
    html += "            itemsPerPage: " + str(items_per_page) + ",\n"
    html += "            currentPage: 1,\n"
    html += "            currentCategory: 'all',\n"
    html += "            currentSearch: '',\n"
    html += "            filteredTokens: [],\n"
    html += "            originalTokens: []\n"
    html += "        };\n"
    html += "        const chartData = " + chart_data_json + ";\n"
    html += "        function initializeChart() {\n"
    html += "            const ctx = document.getElementById('distributionChart').getContext('2d');\n"
    html += "            new Chart(ctx, {\n"
    html += "                type: 'doughnut',\n"
    html += "                data: {\n"
    html += "                    labels: chartData.distribution.labels,\n"
    html += "                    datasets: [{\n"
    html += "                        data: chartData.distribution.data,\n"
    html += "                        backgroundColor: chartData.distribution.colors,\n"
    html += "                        borderWidth: 3,\n"
    html += "                        borderColor: '#fff'\n"
    html += "                    }]\n"
    html += "                },\n"
    html += "                options: {\n"
    html += "                    responsive: true,\n"
    html += "                    maintainAspectRatio: false,\n"
    html += "                    plugins: {\n"
    html += "                        legend: { position: 'bottom' },\n"
    html += "                        tooltip: {\n"
    html += "                            callbacks: {\n"
    html += "                                label: function(context) {\n"
    html += "                                    const total = context.dataset.data.reduce((a, b) => a + b, 0);\n"
    html += "                                    const percentage = ((context.parsed * 100) / total).toFixed(1);\n"
    html += "                                    return context.label + ': ' + context.parsed.toLocaleString() + ' (' + percentage + '%)';\n"
    html += "                                }\n"
    html += "                            }\n"
    html += "                        }\n"
    html += "                    }\n"
    html += "                }\n"
    html += "            });\n"
    html += "        }\n"
    html += "        function initializeTokens() {\n"
    html += "            const allTokens = [];\n"
    html += "            Object.keys(reportData.tokens).forEach(category => {\n"
    html += "                allTokens.push(...reportData.tokens[category]);\n"
    html += "            });\n"
    html += "            reportData.originalTokens = allTokens;\n"
    html += "            reportData.filteredTokens = [...allTokens];\n"
    html += "            renderTokens();\n"
    html += "            updatePagination();\n"
    html += "        }\n"
    html += "        function filterTokens() {\n"
    html += "            const category = document.getElementById('categoryFilter').value;\n"
    html += "            const search = document.getElementById('searchBox').value.toLowerCase();\n"
    html += "            let filtered = [...reportData.originalTokens];\n"
    html += "            if (category !== 'all') {\n"
    html += "                filtered = filtered.filter(token => token.category === category);\n"
    html += "            }\n"
    html += "            if (search) {\n"
    html += "                filtered = filtered.filter(token => \n"
    html += "                    token.token.toLowerCase().includes(search) ||\n"
    html += "                    token.id.toString().includes(search)\n"
    html += "                );\n"
    html += "            }\n"
    html += "            reportData.filteredTokens = filtered;\n"
    html += "            reportData.currentPage = 1;\n"
    html += "            renderTokens();\n"
    html += "            updatePagination();\n"
    html += "        }\n"
    html += "        function renderTokens() {\n"
    html += "            const container = document.getElementById('tokenContainer');\n"
    html += "            const startIndex = (reportData.currentPage - 1) * reportData.itemsPerPage;\n"
    html += "            const endIndex = startIndex + reportData.itemsPerPage;\n"
    html += "            const pageTokens = reportData.filteredTokens.slice(startIndex, endIndex);\n"
    html += "            if (pageTokens.length === 0) {\n"
    html += "                container.innerHTML = '<div class=\"loading\">No tokens found.</div>';\n"
    html += "                return;\n"
    html += "            }\n"
    html += "            let html = '';\n"
    html += "            pageTokens.forEach(token => {\n"
    html += "                html += '<div class=\"token-card ' + token.category + '\">';\n"
    html += "                html += '<div class=\"token-header\">';\n"
    html += "                html += '<span class=\"token-id\">' + token.id + '</span>';\n"
    html += "                html += '</div>';\n"
    html += "                html += '<div class=\"token-display\">' + escapeHtml(token.token) + '</div>';\n"
    html += "                html += '<div class=\"stats-row\">';\n"
    html += "                html += '<span>Length:</span><span>' + (token.response_length || 0) + '</span>';\n"
    html += "                html += '</div>';\n"
    html += "                html += '<div class=\"stats-row\">';\n"
    html += "                html += '<span>Issues:</span><span>' + (token.issue_count || 0) + '</span>';\n"
    html += "                html += '</div>';\n"
    html += "                if (token.error) {\n"
    html += "                    html += '<div class=\"stats-row text-danger\">';\n"
    html += "                    html += '<span>Error:</span><span>' + escapeHtml(token.error) + '</span>';\n"
    html += "                    html += '</div>';\n"
    html += "                }\n"
    html += "                html += '</div>';\n"
    html += "            });\n"
    html += "            container.innerHTML = html;\n"
    html += "        }\n"
    html += "        function updatePagination() {\n"
    html += "            const totalPages = Math.ceil(reportData.filteredTokens.length / reportData.itemsPerPage);\n"
    html += "            const current = reportData.currentPage;\n"
    html += "            let paginationHtml = '';\n"
    html += "            paginationHtml += '<button onclick=\"changePage(' + (current - 1) + ')\"';\n"
    html += "            if (current <= 1) paginationHtml += ' disabled';\n"
    html += "            paginationHtml += '>Previous</button>';\n"
    html += "            paginationHtml += '<span>Page ' + current + ' of ' + totalPages + '</span>';\n"
    html += "            paginationHtml += '<button onclick=\"changePage(' + (current + 1) + ')\"';\n"
    html += "            if (current >= totalPages) paginationHtml += ' disabled';\n"
    html += "            paginationHtml += '>Next</button>';\n"
    html += "            document.getElementById('topPagination').innerHTML = paginationHtml;\n"
    html += "            document.getElementById('bottomPagination').innerHTML = paginationHtml;\n"
    html += "        }\n"
    html += "        function changePage(page) {\n"
    html += "            const totalPages = Math.ceil(reportData.filteredTokens.length / reportData.itemsPerPage);\n"
    html += "            if (page >= 1 && page <= totalPages) {\n"
    html += "                reportData.currentPage = page;\n"
    html += "                renderTokens();\n"
    html += "                updatePagination();\n"
    html += "            }\n"
    html += "        }\n"
    html += "        function resetFilters() {\n"
    html += "            document.getElementById('categoryFilter').value = 'all';\n"
    html += "            document.getElementById('searchBox').value = '';\n"
    html += "            filterTokens();\n"
    html += "        }\n"
    html += "        function escapeHtml(text) {\n"
    html += "            const div = document.createElement('div');\n"
    html += "            div.textContent = text;\n"
    html += "            return div.innerHTML;\n"
    html += "        }\n"
    html += "        document.getElementById('categoryFilter').addEventListener('change', filterTokens);\n"
    html += "        document.getElementById('searchBox').addEventListener('input', filterTokens);\n"
    html += "        document.addEventListener('DOMContentLoaded', function() {\n"
    html += "            initializeChart();\n"
    html += "            initializeTokens();\n"
    html += "        });\n"
    html += "    </script>\n"
    html += "</body>\n"
    html += "</html>\n"

    return html


@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--open-browser', '-o', is_flag=True, help='Open the report in default browser')
@click.option('--items-per-page', default=50, help='Number of items per page (default: 50)')
def main(input_file: str, output_file: str, open_browser: bool, items_per_page: int):
    """
    Generate an enhanced HTML report from domain extraction test results.

    This version includes advanced features like export functionality, keyboard shortcuts,
    performance metrics, and comprehensive analytics.

    INPUT_FILE: Path to the JSON results file
    OUTPUT_FILE: Path for the generated HTML report
    """
    click.echo("🚀 Enhanced Domain Extraction Report Generator")
    click.echo("📂 Loading data from {}...".format(input_file))

    data = load_data(input_file)

    click.echo("🔍 Analyzing results with advanced metrics...")
    analysis = analyze_results(data)
    metrics = calculate_performance_metrics(analysis)

    click.echo("🎨 Generating enhanced HTML report...")
    html_content = generate_enhanced_html_report(data, analysis)

    # Write the HTML file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    file_size = len(html_content)
    click.echo("✅ Enhanced report generated: {}".format(output_path))
    click.echo("📊 File size: {:,} bytes ({:.1f} KB)".format(file_size, file_size/1024))
    click.echo("🎯 Glitch rate: {:.1f}%".format(metrics['glitch_rate']))
    click.echo("📈 Success rate: {:.1f}%".format(metrics['success_rate']))
    click.echo("🔍 Issue diversity: {} unique issue types".format(metrics['issue_diversity']))

    if open_browser:
        import webbrowser
        webbrowser.open('file://{}'.format(output_path.absolute()))
        click.echo("🌐 Report opened in browser")


if __name__ == '__main__':
    main()
