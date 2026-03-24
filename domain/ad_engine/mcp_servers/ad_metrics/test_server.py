#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test Ad Metrics MCP Server

Test the MCP Server directly without going through the full client.

Usage:
    python domain/ad_engine/mcp_servers/ad_metrics/test_server.py
"""
import asyncio
import json
import sys
from pathlib import Path

# Add project path - must be before imports
# test_server.py -> ad_metrics -> mcp_servers -> ad_engine -> domain -> project_root
project_root = Path(__file__).resolve().parents[4]
_src_path = project_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))
# Also add project root for domain imports
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


async def test_server_tools():
    """Test MCP Server tools by importing and calling directly"""
    print("=" * 60)
    print("Test Ad Metrics MCP Server (FastMCP)")
    print("=" * 60)

    # Import the FastMCP instance and tool functions
    from domain.ad_engine.mcp_servers.ad_metrics.ad_metrics_server import (
        mcp,
        query_metrics,
        list_available_metrics,
        get_metric_trend,
    )

    print(f"\n[OK] Server instance: {mcp.name}")

    # Test 1: List available metrics
    print("\n" + "-" * 40)
    print("Test 1: list_available_metrics")
    print("-" * 40)

    result = list_available_metrics()
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Test 2: Query metrics
    print("\n" + "-" * 40)
    print("Test 2: query_metrics")
    print("-" * 40)

    result = query_metrics(
        start_date="2024-03-01",
        end_date="2024-03-07",
        metrics=["ctr", "cvr", "spend"],
    )
    print(f"Query range: {result['query_info']['start_date']} ~ {result['query_info']['end_date']}")
    print(f"Days: {result['query_info']['days']}")
    print("\nMetrics summary:")
    for key, info in result["summary"].items():
        print(f"  {key}: {info['avg']} {info['unit']}")

    # Test 3: Get metric trend
    print("\n" + "-" * 40)
    print("Test 3: get_metric_trend")
    print("-" * 40)

    result = get_metric_trend(
        metric="ctr",
        start_date="2024-03-01",
        end_date="2024-03-05",
    )
    print(f"Metric: {result['metric']['name']} ({result['metric']['unit']})")
    print(f"Stats: avg {result['statistics']['average']}%, max {result['statistics']['max']}%, min {result['statistics']['min']}%")
    print("\nTrend data:")
    for item in result["trend"]:
        print(f"  {item['date']}: {item['value']}%")

    # Test 4: Error handling
    print("\n" + "-" * 40)
    print("Test 4: Error handling")
    print("-" * 40)

    try:
        result = query_metrics(
            start_date="2024-03-01",
            end_date="2024-03-07",
            metrics=["invalid_metric"],
        )
        print(f"Unexpected success: {result}")
    except ValueError as e:
        print(f"[OK] Caught expected error: {e}")

    print("\n" + "=" * 60)
    print("All tests completed [OK]")
    print("=" * 60)


if __name__ == "__main__":
    print("Starting MCP Server tests...\n")
    asyncio.run(test_server_tools())
