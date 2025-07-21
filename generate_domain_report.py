#!/usr/bin/env python3
"""
Standalone HTML report generator for glitch token domain extraction results.

This script generates a comprehensive HTML report from domain extraction test results
that can be exported to PDF. It analyzes token behavior, categorizes issues, and
provides detailed statistics.

Usage:
    python generate_domain_report.py input.json output.html
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple
import click


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
    """Analyze the domain extraction results."""
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
        'issue_details': defaultdict(list)
    }

    for result in results:
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

        analysis['response_lengths'].append(response_length)

        # Categorize tokens
        if creates_valid and breaks_extraction:
            analysis['both_valid_and_breaks'] += 1
            analysis['tokens_by_category']['both'].append({
                'id': token_id,
                'token': token_text,
                'response_length': response_length,
                'issues': issues
            })
        elif creates_valid:
            analysis['creates_valid_domains'] += 1
            analysis['tokens_by_category']['valid_domains'].append({
                'id': token_id,
                'token': token_text,
                'response_length': response_length,
                'issues': issues
            })
        elif breaks_extraction:
            analysis['breaks_domain_extraction'] += 1
            analysis['tokens_by_category']['breaks_extraction'].append({
                'id': token_id,
                'token': token_text,
                'response_length': response_length,
                'issues': issues
            })
        else:
            analysis['neither'] += 1
            analysis['tokens_by_category']['neither'].append({
                'id': token_id,
                'token': token_text,
                'response_length': response_length,
                'issues': issues
            })

        # Count issues
        for issue in issues:
            analysis['issue_categories'][issue] += 1
            analysis['issue_details'][issue].append({
                'token_id': token_id,
                'token': token_text
            })

    return analysis


def generate_chart_data(analysis: Dict[str, Any]) -> str:
    """Generate JavaScript data for charts."""
    categories = {
        'Valid Domains Only': analysis['creates_valid_domains'],
        'Breaks Extraction Only': analysis['breaks_domain_extraction'],
        'Both Valid & Breaks': analysis['both_valid_and_breaks'],
        'Neither': analysis['neither'],
        'Errors': analysis['has_errors']
    }

    chart_data = {
        'labels': list(categories.keys()),
        'data': list(categories.values()),
        'colors': ['#28a745', '#dc3545', '#ffc107', '#6c757d', '#e83e8c']
    }

    return json.dumps(chart_data)


def format_token_for_display(token: str) -> str:
    """Format token for safe HTML display."""
    if not token or token == 'Unknown':
        return '<span class="text-muted">Unknown</span>'

    # Escape HTML and handle special characters
    token = token.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    # Highlight non-printable or unusual characters
    formatted = ""
    for char in token:
        if ord(char) < 32 or ord(char) > 126:
            formatted += f'<span class="special-char" title="Unicode: U+{ord(char):04X}">\\u{ord(char):04X}</span>'
        else:
            formatted += char

    return f'<code class="token-display">{formatted}</code>'


def generate_html_report(data: Dict[str, Any], analysis: Dict[str, Any]) -> str:
    """Generate the complete HTML report."""

    chart_data = generate_chart_data(analysis)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Calculate statistics
    avg_response_length = sum(analysis['response_lengths']) / len(analysis['response_lengths']) if analysis['response_lengths'] else 0
    max_response_length = max(analysis['response_lengths']) if analysis['response_lengths'] else 0
    min_response_length = min(analysis['response_lengths']) if analysis['response_lengths'] else 0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Domain Extraction Analysis Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
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
            padding: 40px 0;
            text-align: center;
            margin-bottom: 30px;
            border-radius: 10px;
        }}

        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
            font-weight: 300;
        }}

        .header .subtitle {{
            font-size: 1.1rem;
            opacity: 0.9;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .summary-card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
            border-left: 4px solid #667eea;
        }}

        .summary-card h3 {{
            font-size: 2rem;
            margin-bottom: 5px;
            color: #667eea;
        }}

        .summary-card p {{
            color: #666;
            font-size: 0.9rem;
        }}

        .chart-container {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        .chart-container h2 {{
            margin-bottom: 20px;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}

        .section {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        .section h2 {{
            color: #333;
            margin-bottom: 20px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}

        .token-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
        }}

        .token-card {{
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            background: #f9f9f9;
        }}

        .token-card.valid-domain {{
            border-left: 4px solid #28a745;
        }}

        .token-card.breaks-extraction {{
            border-left: 4px solid #dc3545;
        }}

        .token-card.both {{
            border-left: 4px solid #ffc107;
        }}

        .token-card.neither {{
            border-left: 4px solid #6c757d;
        }}

        .token-card.error {{
            border-left: 4px solid #e83e8c;
        }}

        .token-display {{
            background: #f8f9fa;
            padding: 4px 8px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            border: 1px solid #dee2e6;
        }}

        .special-char {{
            background: #fff3cd;
            color: #856404;
            padding: 2px 4px;
            border-radius: 3px;
            font-size: 0.8rem;
        }}

        .issues-list {{
            margin-top: 10px;
        }}

        .issue-tag {{
            display: inline-block;
            background: #e9ecef;
            color: #495057;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            margin: 2px;
        }}

        .stats-row {{
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
            font-size: 0.9rem;
        }}

        .text-muted {{
            color: #6c757d;
        }}

        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: bold;
        }}

        .badge-success {{
            background: #d4edda;
            color: #155724;
        }}

        .badge-danger {{
            background: #f8d7da;
            color: #721c24;
        }}

        .badge-warning {{
            background: #fff3cd;
            color: #856404;
        }}

        .badge-secondary {{
            background: #e2e3e5;
            color: #383d41;
        }}

        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9rem;
        }}

        @media print {{
            body {{
                background: white;
            }}
            .container {{
                max-width: none;
                padding: 0;
            }}
            .section {{
                break-inside: avoid;
                box-shadow: none;
                border: 1px solid #ddd;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Domain Extraction Analysis Report</h1>
            <div class="subtitle">
                Model: {data.get('model_path', 'Unknown')} |
                Test Type: {data.get('test_type', 'Unknown')} |
                Generated: {timestamp}
            </div>
        </div>

        <div class="summary-grid">
            <div class="summary-card">
                <h3>{analysis['total_tokens']}</h3>
                <p>Total Tokens Tested</p>
            </div>
            <div class="summary-card">
                <h3>{analysis['creates_valid_domains']}</h3>
                <p>Create Valid Domains</p>
            </div>
            <div class="summary-card">
                <h3>{analysis['breaks_domain_extraction']}</h3>
                <p>Break Domain Extraction</p>
            </div>
            <div class="summary-card">
                <h3>{analysis['both_valid_and_breaks']}</h3>
                <p>Both Valid & Breaks</p>
            </div>
            <div class="summary-card">
                <h3>{analysis['has_errors']}</h3>
                <p>Processing Errors</p>
            </div>
        </div>

        <div class="chart-container">
            <h2>Token Distribution</h2>
            <canvas id="distributionChart" width="400" height="200"></canvas>
        </div>

        <div class="section">
            <h2>Response Statistics</h2>
            <div class="stats-row">
                <span>Average Response Length:</span>
                <span><strong>{avg_response_length:.1f} characters</strong></span>
            </div>
            <div class="stats-row">
                <span>Maximum Response Length:</span>
                <span><strong>{max_response_length:,} characters</strong></span>
            </div>
            <div class="stats-row">
                <span>Minimum Response Length:</span>
                <span><strong>{min_response_length:,} characters</strong></span>
            </div>
        </div>

        <div class="section">
            <h2>Issue Categories</h2>
            <div class="token-grid">"""

    # Add issue category cards
    for issue, count in analysis['issue_categories'].most_common():
        tokens_with_issue = analysis['issue_details'][issue][:5]  # Show first 5
        more_count = len(analysis['issue_details'][issue]) - 5

        html += f"""
                <div class="token-card">
                    <strong>{issue}</strong>
                    <div class="stats-row">
                        <span>Occurrences:</span>
                        <span class="badge badge-warning">{count}</span>
                    </div>
                    <div style="margin-top: 10px; font-size: 0.8rem;">
                        <strong>Sample tokens:</strong><br>"""

        for token_info in tokens_with_issue:
            html += f"{format_token_for_display(token_info['token'])} "

        if more_count > 0:
            html += f"<span class='text-muted'>...and {more_count} more</span>"

        html += "</div></div>"

    html += """
            </div>
        </div>"""

    # Add detailed token sections
    sections = [
        ('Valid Domains Only', 'valid_domains', 'valid-domain', 'success'),
        ('Breaks Extraction Only', 'breaks_extraction', 'breaks-extraction', 'danger'),
        ('Both Valid & Breaks', 'both', 'both', 'warning'),
        ('Neither', 'neither', 'neither', 'secondary'),
        ('Processing Errors', 'errors', 'error', 'danger')
    ]

    for section_title, key, css_class, badge_class in sections:
        tokens = analysis['tokens_by_category'][key]
        if not tokens:
            continue

        html += f"""
        <div class="section">
            <h2>{section_title} <span class="badge badge-{badge_class}">{len(tokens)}</span></h2>
            <div class="token-grid">"""

        for token_info in tokens:
            token_id = token_info['id']
            token_text = token_info['token']

            html += f"""
                <div class="token-card {css_class}">
                    <div class="stats-row">
                        <span><strong>Token ID:</strong></span>
                        <span>{token_id}</span>
                    </div>
                    <div class="stats-row">
                        <span><strong>Token:</strong></span>
                        <span>{format_token_for_display(token_text)}</span>
                    </div>"""

            if 'error' in token_info:
                html += f"""
                    <div class="stats-row">
                        <span><strong>Error:</strong></span>
                        <span class="text-muted">{token_info['error']}</span>
                    </div>"""
            else:
                html += f"""
                    <div class="stats-row">
                        <span><strong>Response Length:</strong></span>
                        <span>{token_info.get('response_length', 0):,} chars</span>
                    </div>"""

                issues = token_info.get('issues', [])
                if issues:
                    html += """
                    <div class="issues-list">
                        <strong>Issues:</strong><br>"""
                    for issue in issues:
                        html += f'<span class="issue-tag">{issue}</span>'
                    html += "</div>"

            html += "</div>"

        html += """
            </div>
        </div>"""

    html += f"""
        <div class="footer">
            Report generated on {timestamp} by Glitcher Domain Extraction Analyzer
        </div>
    </div>

    <script>
        // Create the distribution chart
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
                            padding: 20,
                            usePointStyle: true
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""

    return html


@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--open-browser', '-o', is_flag=True, help='Open the report in default browser')
def main(input_file: str, output_file: str, open_browser: bool):
    """
    Generate an HTML report from domain extraction test results.

    INPUT_FILE: Path to the JSON results file
    OUTPUT_FILE: Path for the generated HTML report
    """
    click.echo(f"Loading data from {input_file}...")
    data = load_data(input_file)

    click.echo("Analyzing results...")
    analysis = analyze_results(data)

    click.echo("Generating HTML report...")
    html_content = generate_html_report(data, analysis)

    # Write the HTML file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    click.echo(f"Report generated: {output_path}")
    click.echo(f"Total tokens analyzed: {analysis['total_tokens']}")
    click.echo(f"Valid domains: {analysis['creates_valid_domains']}")
    click.echo(f"Breaks extraction: {analysis['breaks_domain_extraction']}")
    click.echo(f"Processing errors: {analysis['has_errors']}")

    if open_browser:
        import webbrowser
        webbrowser.open(f"file://{output_path.absolute()}")
        click.echo("Report opened in browser")


if __name__ == '__main__':
    main()
