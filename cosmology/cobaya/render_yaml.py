#!/usr/bin/env python3
"""
render_yaml.py - Convert YAML templates to runnable Cobaya configuration files.

This script replaces placeholder tokens in YAML templates with actual paths,
making the configurations portable across different systems.

Usage:
    python render_yaml.py --class-path /path/to/class_public --output-dir ./chains
    python render_yaml.py -c /path/to/class_public -o ./chains --template edcl_lateonly_production.yaml.template
    
Environment Variables (alternative to command-line args):
    CLASS_PATH: Path to CLASS with EDCL patch
    OUTPUT_DIR: Directory for chain output (default: ./chains)
"""

import argparse
import os
import sys
from pathlib import Path


def render_template(template_path: str, class_path: str, output_dir: str) -> str:
    """
    Read a template file and replace placeholders with actual values.
    
    Args:
        template_path: Path to the .yaml.template file
        class_path: Path to CLASS installation
        output_dir: Directory for chain output
        
    Returns:
        Rendered YAML content as string
    """
    with open(template_path, 'r') as f:
        content = f.read()
    
    # Replace placeholders
    content = content.replace('__CLASS_PATH__', class_path)
    content = content.replace('__OUTPUT_PATH__', output_dir)
    
    return content


def main():
    parser = argparse.ArgumentParser(
        description='Render YAML templates for Cobaya MCMC runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Render all templates
    python render_yaml.py -c /path/to/class_public -o ./chains
    
    # Render a specific template
    python render_yaml.py -c /path/to/class_public -o ./chains -t edcl_lateonly_production.yaml.template
    
    # Use environment variables
    export CLASS_PATH=/path/to/class_public
    export OUTPUT_DIR=./chains
    python render_yaml.py
"""
    )
    
    parser.add_argument(
        '-c', '--class-path',
        default=os.environ.get('CLASS_PATH', ''),
        help='Path to CLASS installation with EDCL patch (or set CLASS_PATH env var)'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        default=os.environ.get('OUTPUT_DIR', './chains'),
        help='Directory for chain output (default: ./chains or OUTPUT_DIR env var)'
    )
    
    parser.add_argument(
        '-t', '--template',
        default=None,
        help='Specific template to render (default: all templates in templates/)'
    )
    
    parser.add_argument(
        '--templates-dir',
        default=None,
        help='Directory containing templates (default: same directory as this script)'
    )
    
    parser.add_argument(
        '--yaml-dir',
        default=None,
        help='Output directory for rendered YAML files (default: parent of templates dir)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate CLASS path
    if not args.class_path:
        print("ERROR: CLASS path not specified.", file=sys.stderr)
        print("Use --class-path or set CLASS_PATH environment variable.", file=sys.stderr)
        sys.exit(1)
    
    class_path = os.path.abspath(args.class_path)
    if not os.path.isdir(class_path):
        print(f"ERROR: CLASS path does not exist: {class_path}", file=sys.stderr)
        sys.exit(1)
    
    # Check for classy module
    classy_path = os.path.join(class_path, 'python', 'classy.cpython*.so')
    import glob
    if not glob.glob(os.path.join(class_path, 'python', 'classy*.so')):
        print(f"WARNING: classy module not found in {class_path}/python/", file=sys.stderr)
        print("Make sure CLASS is compiled with 'make' in the CLASS directory.", file=sys.stderr)
    
    # Determine template directory
    script_dir = Path(__file__).parent
    if args.templates_dir:
        templates_dir = Path(args.templates_dir)
    else:
        templates_dir = script_dir / 'templates'
        if not templates_dir.exists():
            templates_dir = script_dir  # Templates might be in same dir
    
    # Determine output directory for rendered YAML files
    if args.yaml_dir:
        yaml_dir = Path(args.yaml_dir)
    else:
        yaml_dir = templates_dir.parent  # One level up from templates/
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(yaml_dir, exist_ok=True)
    
    # Find templates to render
    if args.template:
        template_files = [templates_dir / args.template]
    else:
        template_files = list(templates_dir.glob('*.yaml.template'))
    
    if not template_files:
        print(f"ERROR: No template files found in {templates_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Render each template
    rendered_files = []
    for template_path in template_files:
        if not template_path.exists():
            print(f"ERROR: Template not found: {template_path}", file=sys.stderr)
            continue
        
        # Generate output filename (remove .template suffix)
        output_name = template_path.name.replace('.template', '')
        output_path = yaml_dir / output_name
        
        if args.verbose:
            print(f"Rendering: {template_path.name} -> {output_path}")
        
        # Render and write
        rendered = render_template(str(template_path), class_path, args.output_dir)
        
        with open(output_path, 'w') as f:
            f.write(rendered)
        
        rendered_files.append(output_path)
    
    # Summary
    print(f"\nRendered {len(rendered_files)} YAML file(s):")
    for f in rendered_files:
        print(f"  - {f}")
    
    print(f"\nConfiguration:")
    print(f"  CLASS path:  {class_path}")
    print(f"  Output dir:  {os.path.abspath(args.output_dir)}")
    print(f"\nTo run MCMC:")
    print(f"  export COBAYA_PACKAGES_PATH=/path/to/cobaya_packages")
    for f in rendered_files:
        print(f"  cobaya-run {f}")


if __name__ == '__main__':
    main()
