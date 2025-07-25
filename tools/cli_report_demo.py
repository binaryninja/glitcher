#!/usr/bin/env python3
"""
CLI Report Demo - Integration demonstration for Domain Extraction Report Generator

This script demonstrates how the HTML report generator would integrate with the
main Glitcher CLI application, showing the command structure and output.

Usage:
    python cli_report_demo.py [options]
"""

import os
import sys
import json
import click
from pathlib import Path
from datetime import datetime


@click.group()
def cli():
    """Glitcher CLI with Domain Report Integration Demo"""
    pass


@cli.command()
@click.argument('model_path')
@click.option('--token-file', '-f', help='File containing token IDs to test')
@click.option('--token-ids', '-t', help='Comma-separated token IDs')
@click.option('--output-dir', '-o', default='./results', help='Output directory for results')
@click.option('--max-tokens', default=50, help='Maximum tokens per test')
@click.option('--generate-report', '-r', is_flag=True, help='Generate HTML report after testing')
@click.option('--open-report', is_flag=True, help='Open report in browser after generation')
@click.option('--report-format', type=click.Choice(['html', 'pdf']), default='html',
              help='Report format (html for browser, pdf for direct PDF)')
def domain(model_path, token_file, token_ids, output_dir, max_tokens, generate_report,
          open_report, report_format):
    """
    Test domain extraction behavior with glitch tokens.

    This command tests how glitch tokens affect domain extraction tasks and
    optionally generates comprehensive HTML reports of the results.
    """

    click.echo("🔍 Glitcher Domain Extraction Testing")
    click.echo("=" * 50)

    # Simulate domain extraction testing
    click.echo(f"📋 Model: {model_path}")
    click.echo(f"📂 Output Directory: {output_dir}")
    click.echo(f"🎯 Max Tokens: {max_tokens}")

    if token_file:
        click.echo(f"📄 Token File: {token_file}")
    if token_ids:
        click.echo(f"🏷️  Token IDs: {token_ids}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Simulate running domain extraction tests
    results_file = os.path.join(output_dir, f"domain_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    click.echo("\n🚀 Starting domain extraction tests...")

    # In real implementation, this would run the actual tests
    # For demo, we'll use the existing results file if available
    if Path("email_llams321_domain_extraction.json").exists():
        click.echo("📊 Using existing test results for demo...")
        results_file = "email_llams321_domain_extraction.json"

        with open(results_file, 'r') as f:
            data = json.load(f)

        total_tokens = len(data.get('results', []))
        click.echo(f"✅ Processed {total_tokens} tokens")

        # Count results
        valid_domains = 0
        breaks_extraction = 0
        both = 0
        errors = 0

        for result in data.get('results', []):
            if 'error' in result:
                errors += 1
                continue

            creates_valid = result.get('creates_valid_domain', False)
            breaks = result.get('breaks_domain_extraction', False)

            if creates_valid and breaks:
                both += 1
            elif creates_valid:
                valid_domains += 1
            elif breaks:
                breaks_extraction += 1

        click.echo(f"📈 Results Summary:")
        click.echo(f"   • Valid domains: {valid_domains}")
        click.echo(f"   • Breaks extraction: {breaks_extraction}")
        click.echo(f"   • Both behaviors: {both}")
        click.echo(f"   • Errors: {errors}")

    else:
        click.echo("⚠️  No test data available for demo")
        click.echo("   In real usage, domain extraction tests would run here")
        results_file = None

    # Generate report if requested
    if generate_report and results_file:
        click.echo(f"\n📊 Generating {report_format.upper()} report...")

        try:
            from generate_domain_report import load_data, analyze_results, generate_html_report

            # Load and analyze data
            data = load_data(results_file)
            analysis = analyze_results(data)

            # Generate report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = os.path.join(output_dir, f"domain_report_{timestamp}.html")

            html_content = generate_html_report(data, analysis)

            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(html_content)

            click.echo(f"✅ Report generated: {report_file}")
            click.echo(f"📁 Report size: {len(html_content):,} characters")

            # Show key statistics
            click.echo(f"\n📋 Report Statistics:")
            click.echo(f"   • Total tokens analyzed: {analysis['total_tokens']}")
            click.echo(f"   • Issue categories found: {len(analysis['issue_categories'])}")
            click.echo(f"   • Top issue: {analysis['issue_categories'].most_common(1)[0] if analysis['issue_categories'] else 'None'}")

            if open_report:
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(report_file)}")
                click.echo("🌐 Report opened in browser")

            # PDF generation instructions
            if report_format == 'pdf' or click.confirm('\n📄 Show PDF export instructions?'):
                click.echo("\n📄 To export report to PDF:")
                click.echo("   1. Open the HTML file in Chrome or Edge")
                click.echo("   2. Press Ctrl+P (Cmd+P on Mac)")
                click.echo("   3. Choose 'Save as PDF' as destination")
                click.echo("   4. Enable 'Background graphics' in More settings")
                click.echo("   5. Adjust margins if needed (recommend: Minimum)")

        except ImportError:
            click.echo("❌ Report generator not available")
            click.echo("   Install requirements: pip install -r report_requirements.txt")
        except Exception as e:
            click.echo(f"❌ Error generating report: {e}")

    elif generate_report:
        click.echo("⚠️  Cannot generate report: No test results available")

    click.echo(f"\n💾 Results saved to: {output_dir}")


@cli.command()
@click.argument('results_file', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output HTML file path')
@click.option('--open-browser', '-b', is_flag=True, help='Open report in browser')
@click.option('--format', 'output_format', type=click.Choice(['html', 'summary']),
              default='html', help='Output format')
def report(results_file, output, open_browser, output_format):
    """
    Generate HTML report from domain extraction results.

    Takes JSON results from domain extraction tests and creates a comprehensive
    HTML report with statistics, charts, and detailed token analysis.
    """

    click.echo("📊 Glitcher Report Generator")
    click.echo("=" * 40)

    try:
        from generate_domain_report import load_data, analyze_results, generate_html_report

        click.echo(f"📂 Loading results from: {results_file}")
        data = load_data(results_file)

        click.echo("🔍 Analyzing data...")
        analysis = analyze_results(data)

        if output_format == 'summary':
            # Show summary only
            click.echo(f"\n📋 Analysis Summary:")
            click.echo(f"   Model: {data.get('model_path', 'Unknown')}")
            click.echo(f"   Total Tokens: {analysis['total_tokens']}")
            click.echo(f"   Valid Domains: {analysis['creates_valid_domains']}")
            click.echo(f"   Breaks Extraction: {analysis['breaks_domain_extraction']}")
            click.echo(f"   Both Behaviors: {analysis['both_valid_and_breaks']}")
            click.echo(f"   Processing Errors: {analysis['has_errors']}")

            if analysis['issue_categories']:
                click.echo(f"\n🏷️  Top Issues:")
                for issue, count in analysis['issue_categories'].most_common(5):
                    click.echo(f"   • {issue}: {count}")

        else:
            # Generate HTML report
            if not output:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output = f"domain_report_{timestamp}.html"

            click.echo(f"🎨 Generating HTML report...")
            html_content = generate_html_report(data, analysis)

            click.echo(f"💾 Saving to: {output}")
            with open(output, 'w', encoding='utf-8') as f:
                f.write(html_content)

            file_size = len(html_content)
            click.echo(f"✅ Report generated successfully!")
            click.echo(f"   File size: {file_size:,} characters")
            click.echo(f"   Features: Interactive charts, token analysis, PDF-ready")

            if open_browser:
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(output)}")
                click.echo("🌐 Report opened in browser")
            else:
                click.echo(f"🌐 Open report: {output}")

    except ImportError:
        click.echo("❌ Report generator not available")
        click.echo("   Install: pip install -r report_requirements.txt")
    except Exception as e:
        click.echo(f"❌ Error: {e}")


@cli.command()
@click.argument('demo_file', type=click.Path(exists=True))
def demo(demo_file):
    """
    Run a complete demonstration of domain testing and report generation.

    DEMO_FILE: Path to the domain extraction results JSON file to use for demo.

    This demonstrates the full workflow from domain extraction testing
    to comprehensive HTML report generation.
    """

    click.echo("🎬 Glitcher Domain Analysis Demo")
    click.echo("=" * 50)

    # Demo data provided as argument
    click.echo(f"📂 Using demo data: {demo_file}")

    click.echo("🎯 Demo Workflow:")
    click.echo("   1. Load domain extraction test results")
    click.echo("   2. Analyze token behaviors and issues")
    click.echo("   3. Generate comprehensive HTML report")
    click.echo("   4. Show integration with CLI commands")

    if click.confirm("\n▶️  Run demo?"):
        # Step 1: Show domain command demo
        click.echo("\n" + "="*60)
        click.echo("STEP 1: Domain Extraction Testing")
        click.echo("="*60)
        click.echo("Command: glitcher domain meta-llama/Llama-3.2-1B-Instruct --generate-report")

        ctx = click.Context(domain)
        ctx.invoke(domain,
                  model_path="meta-llama/Llama-3.2-1B-Instruct",
                  token_file=None,
                  token_ids=None,
                  output_dir="./demo_results",
                  max_tokens=50,
                  generate_report=True,
                  open_report=False,
                  report_format='html')

        # Step 2: Show report command demo
        click.echo("\n" + "="*60)
        click.echo("STEP 2: Standalone Report Generation")
        click.echo("="*60)
        click.echo(f"Command: glitcher report {demo_file} --format summary")

        ctx = click.Context(report)
        ctx.invoke(report,
                  results_file=demo_file,
                  output=None,
                  open_browser=False,
                  output_format='summary')

        click.echo("\n" + "="*60)
        click.echo("DEMO COMPLETE")
        click.echo("="*60)

        click.echo("🎉 Demo completed successfully!")
        click.echo("\n💡 Key Features Demonstrated:")
        click.echo("   • Integrated domain extraction testing")
        click.echo("   • Automatic report generation")
        click.echo("   • Multiple output formats (HTML, summary)")
        click.echo("   • Browser integration")
        click.echo("   • PDF export capability")

        click.echo("\n🔧 Integration Points:")
        click.echo("   • `glitcher domain --generate-report` - Test + Report")
        click.echo("   • `glitcher report results.json` - Standalone Report")
        click.echo("   • `--open-browser` - Auto-open in browser")
        click.echo("   • `--format summary` - Quick CLI summary")


@cli.command()
@click.option('--show-commands', is_flag=True, help='Show example commands')
def integration():
    """
    Show integration examples for the domain report generator.

    Displays example commands and workflows for integrating the HTML
    report generator with the main Glitcher CLI application.
    """

    click.echo("🔧 Integration Examples for Domain Report Generator")
    click.echo("=" * 60)

    click.echo("\n📋 Command Integration:")
    click.echo("   • Enhanced `domain` command with `--generate-report` option")
    click.echo("   • New `report` command for standalone report generation")
    click.echo("   • Automatic file management and output organization")

    if click.confirm("Show example commands?") or True:
        click.echo("\n💻 Example Commands:")

        examples = [
            ("Basic domain testing with report",
             "glitcher domain meta-llama/Llama-3.2-1B-Instruct --token-file tokens.json --generate-report"),

            ("Domain testing with auto-open report",
             "glitcher domain meta-llama/Llama-3.2-1B-Instruct --token-ids 1234,5678 --generate-report --open-report"),

            ("Generate standalone report",
             "glitcher report domain_results.json --output custom_report.html"),

            ("Quick summary from results",
             "glitcher report domain_results.json --format summary"),

            ("Full pipeline with browser open",
             "glitcher domain meta-llama/Llama-3.2-1B-Instruct --generate-report --open-report"),
        ]

        for desc, cmd in examples:
            click.echo(f"\n   {desc}:")
            click.echo(f"   $ {cmd}")

    click.echo("\n📊 Report Features:")
    click.echo("   • Interactive charts and visualizations")
    click.echo("   • Token categorization and analysis")
    click.echo("   • Issue frequency and distribution")
    click.echo("   • PDF-export ready styling")
    click.echo("   • Responsive design for all devices")

    click.echo("\n🎯 Use Cases:")
    click.echo("   • Research documentation")
    click.echo("   • Model behavior analysis")
    click.echo("   • Issue tracking and debugging")
    click.echo("   • Presentation materials")
    click.echo("   • Compliance reporting")


if __name__ == '__main__':
    click.echo("🚀 Glitcher CLI Demo - Domain Report Integration")
    click.echo("=" * 60)
    click.echo("Available commands:")
    click.echo("  • domain      - Test domain extraction (with optional report)")
    click.echo("  • report      - Generate HTML report from results")
    click.echo("  • demo        - Run complete demonstration")
    click.echo("  • integration - Show integration examples")
    click.echo()
    click.echo("Run 'python cli_report_demo.py COMMAND --help' for command details")
    click.echo("Run 'python cli_report_demo.py demo <demo_file.json>' for full demonstration")
    click.echo()

    cli()
