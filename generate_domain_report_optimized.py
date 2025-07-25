#!/usr/bin/env python3
"""
Optimized Domain Extraction Report Generator for Large Datasets

This version is designed to handle large datasets (1000+ tokens) efficiently
by implementing pagination, lazy loading, and performance optimizations.

Usage:
    python generate_domain_report_optimized.py input.json output.html
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple
import click
import math


def load_data(filepath: str) -> Dict[str, Any]:
    """Load and validate JSON data."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        click.echo(f"Error loading {filepath}: {e}", err=True)
        sys.exit(1)


def analyze_results(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the domain extraction results with memory-efficient processing."""
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
        'min_response_length': float('inf')
    }

    # Process in chunks to avoid memory issues
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
                    'error': result['error']
                })
                continue

            creates_valid = result.get('creates_valid_domain', False)
            breaks_extraction = result.get('breaks_domain_extraction', False)
            issues = result.get('issues', [])
            response_length = result.get('response_length', 0)

            # Update length statistics
            analysis['response_lengths'].append(response_length)
            analysis['max_response_length'] = max(analysis['max_response_length'], response_length)
            analysis['min_response_length'] = min(analysis['min_response_length'], response_length)

            # Categorize tokens with minimal data
            token_summary = {
                'id': token_id,
                'token': token_text[:50] + '...' if len(str(token_text)) > 50 else token_text,  # Truncate long tokens
                'response_length': response_length,
                'issue_count': len(issues),
                'top_issue': issues[0] if issues else None
            }

            if creates_valid and breaks_extraction:
                analysis['both_valid_and_breaks'] += 1
                analysis['tokens_by_category']['both'].append(token_summary)
            elif creates_valid:
                analysis['creates_valid_domains'] += 1
                analysis['tokens_by_category']['valid_domains'].append(token_summary)
            elif breaks_extraction:
                analysis['breaks_domain_extraction'] += 1
                analysis['tokens_by_category']['breaks_extraction'].append(token_summary)
            else:
                analysis['neither'] += 1
                analysis['tokens_by_category']['neither'].append(token_summary)

            # Count issues efficiently
            for issue in issues:
                analysis['issue_categories'][issue] += 1
                # Only store sample tokens for top issues
                if len(analysis['issue_details'][issue]) < 10:
                    analysis['issue_details'][issue].append({
                        'token_id': token_id,
                        'token': str(token_text)[:30] + '...' if len(str(token_text)) > 30 else str(token_text)
                    })

    # Calculate average
    if analysis['response_lengths']:
        analysis['avg_response_length'] = sum(analysis['response_lengths']) / len(analysis['response_lengths'])

    if analysis['min_response_length'] == float('inf'):
        analysis['min_response_length'] = 0

    return analysis


def format_token_for_display(token: str, max_length: int = 30) -> str:
    """Format token for safe, efficient HTML display."""
    if not token or token == 'Unknown':
        return '<span class="text-muted">Unknown</span>'

    # Truncate long tokens
    if len(token) > max_length:
        token = token[:max_length] + '...'

    # Simple HTML escape
    token = str(token).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    return f'<code class="token-display">{token}</code>'


def generate_chart_data(analysis: Dict[str, Any]) -> str:
    """Generate lightweight chart data."""
    categories = {
        'Valid Domains': analysis['creates_valid_domains'],
        'Breaks Extraction': analysis['breaks_domain_extraction'],
        'Both': analysis['both_valid_and_breaks'],
        'Neither': analysis['neither'],
        'Errors': analysis['has_errors']
    }

    return json.dumps({
        'labels': list(categories.keys()),
        'data': list(categories.values()),
        'colors': ['#28a745', '#dc3545', '#ffc107', '#6c757d', '#e83e8c']
    })


def generate_optimized_html_report(data: Dict[str, Any], analysis: Dict[str, Any]) -> str:
    """Generate optimized HTML report with pagination and lazy loading."""

    chart_data = generate_chart_data(analysis)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Pagination settings
    items_per_page = 50

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Domain Extraction Analysis Report (Optimized)</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.5;
            color: #333;
            background: #f8f9fa;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
            margin-bottom: 20px;
            border-radius: 8px;
        }}

        .header h1 {{
            font-size: 2rem;
            margin-bottom: 8px;
            font-weight: 300;
        }}

        .performance-note {{
            background: #e7f3ff;
            border: 1px solid #b3d9ff;
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 20px;
            font-size: 0.9rem;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}

        .stat-card {{
            background: white;
            border-radius: 6px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 3px solid #667eea;
        }}

        .stat-card h3 {{
            font-size: 1.5rem;
            color: #667eea;
            margin-bottom: 4px;
        }}

        .stat-card p {{
            color: #666;
            font-size: 0.85rem;
        }}

        .section {{
            background: white;
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .section h2 {{
            color: #333;
            margin-bottom: 15px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 8px;
            font-size: 1.3rem;
        }}

        .chart-container {{
            position: relative;
            height: 300px;
            margin-bottom: 20px;
        }}

        .controls {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
        }}

        .control-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .control-group label {{
            font-size: 0.9rem;
            font-weight: 500;
        }}

        .control-group select, .control-group input {{
            padding: 6px 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 0.9rem;
        }}

        .pagination {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            margin: 20px 0;
        }}

        .pagination button {{
            padding: 8px 12px;
            border: 1px solid #ddd;
            background: white;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9rem;
        }}

        .pagination button:hover:not(:disabled) {{
            background: #f8f9fa;
        }}

        .pagination button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}

        .pagination button.active {{
            background: #667eea;
            color: white;
            border-color: #667eea;
        }}

        .token-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 12px;
        }}

        .token-card {{
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 12px;
            background: #fafafa;
            font-size: 0.9rem;
        }}

        .token-card.valid-domain {{ border-left: 3px solid #28a745; }}
        .token-card.breaks-extraction {{ border-left: 3px solid #dc3545; }}
        .token-card.both {{ border-left: 3px solid #ffc107; }}
        .token-card.neither {{ border-left: 3px solid #6c757d; }}
        .token-card.error {{ border-left: 3px solid #e83e8c; }}

        .token-display {{
            background: #f1f3f4;
            padding: 3px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.8rem;
            border: 1px solid #dadce0;
            word-break: break-all;
        }}

        .issue-tag {{
            display: inline-block;
            background: #e9ecef;
            color: #495057;
            padding: 2px 6px;
            border-radius: 10px;
            font-size: 0.75rem;
            margin: 2px;
        }}

        .loading {{
            text-align: center;
            padding: 40px;
            color: #666;
        }}

        .stats-row {{
            display: flex;
            justify-content: space-between;
            margin: 4px 0;
            font-size: 0.85rem;
        }}

        .text-muted {{ color: #666; }}

        .search-box {{
            width: 200px;
            padding: 6px 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}

        .results-info {{
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 10px;
        }}

        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.85rem;
        }}

        @media (max-width: 768px) {{
            .container {{ padding: 10px; }}
            .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .controls {{ flex-direction: column; }}
            .control-group {{ width: 100%; }}
        }}

        @media print {{
            .controls, .pagination {{ display: none; }}
            .section {{ break-inside: avoid; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Domain Extraction Analysis Report</h1>
            <div style="font-size: 1rem; opacity: 0.9;">
                Model: {data.get('model_path', 'Unknown')} |
                Generated: {timestamp} |
                Optimized for Large Datasets
            </div>
        </div>

        <div class="performance-note">
            📊 <strong>Performance Mode:</strong> This report uses pagination and lazy loading to handle large datasets efficiently.
            Data is loaded in chunks of {items_per_page} items for optimal browser performance.
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <h3>{analysis['total_tokens']:,}</h3>
                <p>Total Tokens</p>
            </div>
            <div class="stat-card">
                <h3>{analysis['creates_valid_domains']:,}</h3>
                <p>Valid Domains</p>
            </div>
            <div class="stat-card">
                <h3>{analysis['breaks_domain_extraction']:,}</h3>
                <p>Break Extraction</p>
            </div>
            <div class="stat-card">
                <h3>{analysis['both_valid_and_breaks']:,}</h3>
                <p>Both Behaviors</p>
            </div>
            <div class="stat-card">
                <h3>{analysis['has_errors']:,}</h3>
                <p>Errors</p>
            </div>
        </div>

        <div class="section">
            <h2>Distribution Overview</h2>
            <div class="chart-container">
                <canvas id="distributionChart"></canvas>
            </div>
        </div>

        <div class="section">
            <h2>Response Statistics</h2>
            <div class="stats-row">
                <span>Average Response Length:</span>
                <span><strong>{analysis['avg_response_length']:.1f} characters</strong></span>
            </div>
            <div class="stats-row">
                <span>Maximum Response Length:</span>
                <span><strong>{analysis['max_response_length']:,} characters</strong></span>
            </div>
            <div class="stats-row">
                <span>Minimum Response Length:</span>
                <span><strong>{analysis['min_response_length']:,} characters</strong></span>
            </div>
        </div>

        <div class="section">
            <h2>Top Issues</h2>
            <div class="token-grid">"""

    # Add top issues (limited to prevent DOM bloat)
    for issue, count in list(analysis['issue_categories'].most_common(10)):
        sample_tokens = analysis['issue_details'][issue][:3]

        html += f"""
                <div class="token-card">
                    <strong>{issue}</strong>
                    <div class="stats-row">
                        <span>Occurrences:</span>
                        <span style="background: #fff3cd; padding: 2px 6px; border-radius: 3px; font-size: 0.8rem;">{count:,}</span>
                    </div>
                    <div style="margin-top: 8px; font-size: 0.8rem;">
                        <strong>Sample tokens:</strong><br>"""

        for token_info in sample_tokens:
            html += f"{format_token_for_display(token_info['token'], 20)} "

        if len(analysis['issue_details'][issue]) > 3:
            more_count = len(analysis['issue_details'][issue]) - 3
            html += f"<span class='text-muted'>+{more_count} more</span>"

        html += "</div></div>"

    html += """
            </div>
        </div>

        <div class="section">
            <h2>Token Analysis</h2>

            <div class="controls">
                <div class="control-group">
                    <label for="categoryFilter">Category:</label>
                    <select id="categoryFilter">
                        <option value="all">All Categories</option>
                        <option value="breaks_extraction">Breaks Extraction</option>
                        <option value="both">Both Behaviors</option>
                        <option value="valid_domains">Valid Domains Only</option>
                        <option value="neither">Neither</option>
                        <option value="errors">Errors</option>
                    </select>
                </div>

                <div class="control-group">
                    <label for="searchBox">Search:</label>
                    <input type="text" id="searchBox" class="search-box" placeholder="Search tokens...">
                </div>

                <div class="control-group">
                    <label for="sortBy">Sort by:</label>
                    <select id="sortBy">
                        <option value="id">Token ID</option>
                        <option value="response_length">Response Length</option>
                        <option value="issue_count">Issue Count</option>
                    </select>
                </div>
            </div>

            <div class="results-info" id="resultsInfo"></div>

            <div class="pagination" id="topPagination"></div>

            <div id="tokenContainer" class="token-grid">
                <div class="loading">Loading tokens...</div>
            </div>

            <div class="pagination" id="bottomPagination"></div>
        </div>

        <div class="footer">
            Report generated on {timestamp} by Glitcher Domain Extraction Analyzer (Optimized Version)
        </div>
    </div>

    <script>
        // Global data store
        const reportData = {{
            tokens: {json.dumps(analysis['tokens_by_category'])},
            itemsPerPage: {items_per_page},
            currentPage: 1,
            currentCategory: 'all',
            currentSearch: '',
            currentSort: 'id',
            filteredTokens: []
        }};

        // Initialize chart
        const ctx = document.getElementById('distributionChart').getContext('2d');
        const chartData = {chart_data};

        new Chart(ctx, {{
            type: 'doughnut',
            data: {{
                labels: chartData.labels,
                datasets: [{{
                    data: chartData.data,
                    backgroundColor: chartData.colors,
                    borderWidth: 2,
                    borderColor: '#fff'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'bottom',
                        labels: {{
                            padding: 15,
                            usePointStyle: true
                        }}
                    }}
                }}
            }}
        }});

        // Token management functions
        function getAllTokens() {{
            if (reportData.currentCategory === 'all') {{
                let allTokens = [];
                Object.keys(reportData.tokens).forEach(category => {{
                    reportData.tokens[category].forEach(token => {{
                        allTokens.push({{...token, category}});
                    }});
                }});
                return allTokens;
            }} else {{
                return reportData.tokens[reportData.currentCategory].map(token => {{
                    return {{...token, category: reportData.currentCategory}};
                }});
            }}
        }}

        function filterTokens() {{
            let tokens = getAllTokens();

            // Apply search filter
            if (reportData.currentSearch) {{
                const search = reportData.currentSearch.toLowerCase();
                tokens = tokens.filter(token =>
                    token.token.toLowerCase().includes(search) ||
                    token.id.toString().includes(search) ||
                    (token.top_issue && token.top_issue.toLowerCase().includes(search))
                );
            }}

            // Apply sorting
            tokens.sort((a, b) => {{
                switch (reportData.currentSort) {{
                    case 'response_length':
                        return b.response_length - a.response_length;
                    case 'issue_count':
                        return b.issue_count - a.issue_count;
                    default:
                        return a.id - b.id;
                }}
            }});

            reportData.filteredTokens = tokens;
            reportData.currentPage = 1;
            updateDisplay();
        }}

        function formatTokenDisplay(token, maxLength = 25) {{
            if (!token || token === 'Unknown') return '<span class="text-muted">Unknown</span>';

            let displayToken = token.length > maxLength ? token.substring(0, maxLength) + '...' : token;
            displayToken = displayToken.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

            return `<code class="token-display">${{displayToken}}</code>`;
        }}

        function getCategoryClass(category) {{
            const classes = {{
                'valid_domains': 'valid-domain',
                'breaks_extraction': 'breaks-extraction',
                'both': 'both',
                'neither': 'neither',
                'errors': 'error'
            }};
            return classes[category] || '';
        }}

        function renderTokens() {{
            const container = document.getElementById('tokenContainer');
            const startIndex = (reportData.currentPage - 1) * reportData.itemsPerPage;
            const endIndex = startIndex + reportData.itemsPerPage;
            const pageTokens = reportData.filteredTokens.slice(startIndex, endIndex);

            if (pageTokens.length === 0) {{
                container.innerHTML = '<div class="loading">No tokens found matching your criteria.</div>';
                return;
            }}

            let html = '';
            pageTokens.forEach(token => {{
                const categoryClass = getCategoryClass(token.category);

                html += `
                    <div class="token-card ${{categoryClass}}">
                        <div class="stats-row">
                            <span><strong>ID:</strong></span>
                            <span>${{token.id}}</span>
                        </div>
                        <div class="stats-row">
                            <span><strong>Token:</strong></span>
                            <span>${{formatTokenDisplay(token.token)}}</span>
                        </div>`;

                if (token.error) {{
                    html += `
                        <div class="stats-row">
                            <span><strong>Error:</strong></span>
                            <span class="text-muted">${{token.error}}</span>
                        </div>`;
                }} else {{
                    html += `
                        <div class="stats-row">
                            <span><strong>Response:</strong></span>
                            <span>${{token.response_length.toLocaleString()}} chars</span>
                        </div>
                        <div class="stats-row">
                            <span><strong>Issues:</strong></span>
                            <span>${{token.issue_count}}</span>
                        </div>`;

                    if (token.top_issue) {{
                        html += `
                            <div style="margin-top: 6px;">
                                <span class="issue-tag">${{token.top_issue}}</span>
                            </div>`;
                    }}
                }}

                html += '</div>';
            }});

            container.innerHTML = html;
        }}

        function renderPagination() {{
            const totalPages = Math.ceil(reportData.filteredTokens.length / reportData.itemsPerPage);
            const currentPage = reportData.currentPage;

            let paginationHTML = '';

            // Previous button
            paginationHTML += `<button onclick="changePage(${{currentPage - 1}})" ${{currentPage === 1 ? 'disabled' : ''}}>← Previous</button>`;

            // Page numbers (show max 5 pages)
            const startPage = Math.max(1, currentPage - 2);
            const endPage = Math.min(totalPages, startPage + 4);

            for (let i = startPage; i <= endPage; i++) {{
                const activeClass = i === currentPage ? 'active' : '';
                paginationHTML += `<button class="${{activeClass}}" onclick="changePage(${{i}})">${{i}}</button>`;
            }}

            // Next button
            paginationHTML += `<button onclick="changePage(${{currentPage + 1}})" ${{currentPage === totalPages ? 'disabled' : ''}}>Next →</button>`;

            // Page info
            const startItem = (currentPage - 1) * reportData.itemsPerPage + 1;
            const endItem = Math.min(currentPage * reportData.itemsPerPage, reportData.filteredTokens.length);
            paginationHTML += `<span style="margin-left: 20px; font-size: 0.9rem;">Page ${{currentPage}} of ${{totalPages}} (${{startItem}}-${{endItem}} of ${{reportData.filteredTokens.length}})</span>`;

            document.getElementById('topPagination').innerHTML = paginationHTML;
            document.getElementById('bottomPagination').innerHTML = paginationHTML;
        }}

        function updateResultsInfo() {{
            const info = document.getElementById('resultsInfo');
            const total = reportData.filteredTokens.length;
            const category = reportData.currentCategory === 'all' ? 'all categories' : reportData.currentCategory.replace('_', ' ');

            let infoText = `Showing ${{total.toLocaleString()}} tokens from ${{category}}`;
            if (reportData.currentSearch) {{
                infoText += ` matching "${{reportData.currentSearch}}"`;
            }}

            info.textContent = infoText;
        }}

        function updateDisplay() {{
            renderTokens();
            renderPagination();
            updateResultsInfo();
        }}

        function changePage(page) {{
            const totalPages = Math.ceil(reportData.filteredTokens.length / reportData.itemsPerPage);
            if (page >= 1 && page <= totalPages) {{
                reportData.currentPage = page;
                updateDisplay();
                // Scroll to top of token container
                document.getElementById('tokenContainer').scrollIntoView({{ behavior: 'smooth' }});
            }}
        }}

        // Event listeners
        document.getElementById('categoryFilter').addEventListener('change', function(e) {{
            reportData.currentCategory = e.target.value;
            filterTokens();
        }});

        document.getElementById('searchBox').addEventListener('input', function(e) {{
            reportData.currentSearch = e.target.value;
            // Debounce search
            clearTimeout(window.searchTimeout);
            window.searchTimeout = setTimeout(() => {{
                filterTokens();
            }}, 300);
        }});

        document.getElementById('sortBy').addEventListener('change', function(e) {{
            reportData.currentSort = e.target.value;
            filterTokens();
        }});

        // Initialize display
        document.addEventListener('DOMContentLoaded', function() {{
            filterTokens();
        }});
    </script>
</body>
</html>"""

    return html


@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--open-browser', '-o', is_flag=True, help='Open the report in default browser')
@click.option('--items-per-page', default=50, help='Number of items per page (default: 50)')
def main(input_file: str, output_file: str, open_browser: bool, items_per_page: int):
    """
    Generate an optimized HTML report from domain extraction test results.

    This version is optimized for large datasets with pagination and lazy loading.

    INPUT_FILE: Path to the JSON results file
    OUTPUT_FILE: Path for the generated HTML report
    """
    click.echo(f"🚀 Optimized Domain Extraction Report Generator")
    click.echo(f"📂 Loading data from {input_file}...")

    data = load_data(input_file)

    click.echo("🔍 Analyzing results (memory-optimized)...")
    analysis = analyze_results(data)

    click.echo("🎨 Generating optimized HTML report...")
    html_content = generate_optimized_html_report(data, analysis)

    # Write the HTML file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    file_size = len(html_content)
    click.echo(f"✅ Optimized report generated: {output_path}")
    click.echo(f"📊 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")

    if open_browser:
        import webbrowser
        webbrowser.open(f'file://{output_path.absolute()}')
        click.echo("🌐 Report opened in browser")


if __name__ == '__main__':
    main()
