#!/usr/bin/env python3
"""Visualize the car analysis workflow graph structure"""

import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path

# Ensure the repository root is on the Python path for package imports
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from car_analysis.core.graph import build_single_car_graph


def visualize_workflow():
    """Generate and save workflow visualization"""

    print("🎨 Generating Car Analysis Workflow Visualization")
    print("=" * 60)

    try:
        # Build the graph
        print("🏗️  Building LangGraph workflow...")
        workflow = build_single_car_graph()

        # Generate mermaid diagram
        print("📊 Generating Mermaid diagram...")
        mermaid_code = workflow.get_graph().draw_mermaid()

        # Save mermaid code
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mermaid_filename = f"car_analysis_workflow_{timestamp}.mmd"

        with open(mermaid_filename, "w") as f:
            f.write(mermaid_code)

        print(f"✅ Mermaid diagram saved: {mermaid_filename}")

        # Try to generate PNG
        try:
            print("🖼️  Generating PNG visualization...")
            png_data = workflow.get_graph().draw_mermaid_png()

            png_filename = f"car_analysis_workflow_{timestamp}.png"
            with open(png_filename, "wb") as f:
                f.write(png_data)

            print(f"✅ PNG diagram saved: {png_filename}")

        except Exception as png_error:
            print(f"⚠️  PNG generation failed: {png_error}")
            print("💡 You can use the .mmd file with online Mermaid tools")

        # Print workflow structure
        print(f"\n📋 Workflow Structure Analysis:")
        print("=" * 40)

        # Display the mermaid code for inspection
        print("🔍 Mermaid Diagram Code:")
        print("-" * 30)
        lines = mermaid_code.split('\n')
        for i, line in enumerate(lines[:20], 1):  # Show first 20 lines
            print(f"{i:2d}: {line}")

        if len(lines) > 20:
            print(f"... ({len(lines) - 20} more lines)")

        print(f"\n✨ Workflow Features:")
        print(f"   • LangGraph state management")
        print(f"   • Conditional routing based on validation")
        print(f"   • Retry mechanisms with max limits")
        print(f"   • Error tracking and reporting")
        print(f"   • Dual scoring validation")

        return {
            "mermaid_file": mermaid_filename,
            "png_file": png_filename if 'png_filename' in locals() else None,
            "success": True
        }

    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    result = visualize_workflow()

    if result["success"]:
        print(f"\n🎉 Visualization Complete!")
        print(f"📁 Files generated:")
        print(f"   • {result['mermaid_file']} (Mermaid diagram)")
        if result.get('png_file'):
            print(f"   • {result['png_file']} (PNG image)")

        print(f"\n💡 To view the workflow:")
        print(f"   • Open the PNG file directly")
        print(f"   • Or copy the .mmd content to https://mermaid.live")
    else:
        print(f"\n❌ Visualization failed: {result.get('error')}")
