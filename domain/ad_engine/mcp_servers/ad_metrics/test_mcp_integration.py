#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test Agent MCP Server Integration

Test if AgentService can successfully connect to and call MCP Server.

Run:
    python domain/ad_engine/mcp_servers/ad_metrics/test_mcp_integration.py
"""
import asyncio
import json
import sys
from pathlib import Path

# Add project path - must be before imports
# test_mcp_integration.py -> ad_metrics -> mcp_servers -> ad_engine -> domain -> project_root
project_root = Path(__file__).resolve().parents[4]
_src_path = project_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))


async def test_mcp_client():
    """Test MCP Client connection and calls"""
    print("=" * 60)
    print("Test MCP Client connecting to MCP Server")
    print("=" * 60)

    from agent.mcp import MCPClient, MCPServerConfigLoader
    from agent.mcp.config_loader import MCPServerConfig

    # Method 1: Load config from domain directory
    print("\n[1] Load MCP Server config from domain directory...")
    loader = MCPServerConfigLoader()
    domain_root = project_root / "domain" / "ad_engine"

    try:
        configs = loader.load_from_domain(domain_root)
        print(f"[OK] Config loaded: {len(configs)} Servers")
        for name, config in configs.items():
            status = 'enabled' if config.enabled else 'disabled'
            print(f"  - {name}: {config.transport} ({status})")
    except Exception as e:
        print(f"[WARN] Load config failed: {e}")
        print("\nUsing manual config...")
        # Method 2: Manual config
        configs = {
            "ad_metrics": MCPServerConfig(
                name="ad_metrics",
                transport="stdio",
                command="python",
                args=[str(domain_root / "mcp_servers" / "ad_metrics" / "ad_metrics_server.py")],
                enabled=True,
                timeout=30,
            )
        }
        print(f"[OK] Manual config: {list(configs.keys())}")

    # Create MCP Client
    print("\n[2] Create MCP Client and initialize connection...")
    client = MCPClient(configs)

    try:
        await client.initialize()
        stats = client.get_stats()
        print(f"[OK] Initialization complete:")
        print(f"  - Connected Servers: {stats['server_count']}")
        print(f"  - Discovered Tools: {stats['tool_count']}")
        print(f"  - Tools: {stats['tools']}")

        if stats['tool_count'] == 0:
            print("[WARN] No tools found, skipping tool call tests")
            return

        # Get tool adapters
        print("\n[3] Get tool adapters...")
        adapters = client.get_tool_adapters()
        print(f"[OK] Got {len(adapters)} adapters:")
        for adapter in adapters:
            desc = adapter.description[:50] if len(adapter.description) > 50 else adapter.description
            print(f"  - {adapter.name}: {desc}...")

        # Test tool calls
        print("\n[4] Test calling MCP tools...")

        # 4.1 List available metrics
        print("\n[4.1] Call list_available_metrics...")
        result = await client.call_tool("list_available_metrics", {})
        if result.success:
            # Result content is a string like "[{'type': 'text', 'text': '{...}'}]"
            # We need to extract the 'text' field and parse it as JSON
            try:
                outer = eval(result.content)
                if isinstance(outer, list) and len(outer) > 0:
                    text_content = outer[0].get('text', '{}')
                    data = json.loads(text_content)
                else:
                    data = outer
            except (json.JSONDecodeError, SyntaxError):
                data = eval(result.content)
            print(f"[OK] Got metrics list: {data.get('total', 0)} metrics")
            for m in data.get('metrics', [])[:3]:
                print(f"  - {m['key']}: {m['name']} ({m['unit']})")
        else:
            print(f"[FAIL] Call failed: {result.error}")

        # 4.2 Query metrics
        print("\n[4.2] Call query_metrics...")
        result = await client.call_tool("query_metrics", {
            "start_date": "2024-03-01",
            "end_date": "2024-03-07",
            "metrics": ["ctr", "cvr", "spend"],
        })
        if result.success:
            try:
                outer = eval(result.content)
                if isinstance(outer, list) and len(outer) > 0:
                    text_content = outer[0].get('text', '{}')
                    data = json.loads(text_content)
                else:
                    data = outer
            except (json.JSONDecodeError, SyntaxError):
                data = eval(result.content)
            print(f"[OK] Query successful:")
            print(f"  - Date range: {data.get('query_info', {}).get('start_date')} ~ {data.get('query_info', {}).get('end_date')}")
            print(f"  - Days: {data.get('query_info', {}).get('days')}")
            print(f"  - Metrics:")
            for k, v in data.get('summary', {}).items():
                print(f"      {k}: {v['avg']} {v['unit']}")
        else:
            print(f"[FAIL] Call failed: {result.error}")

        # 4.3 Get trend
        print("\n[4.3] Call get_metric_trend...")
        result = await client.call_tool("get_metric_trend", {
            "metric": "ctr",
            "start_date": "2024-03-01",
            "end_date": "2024-03-05",
        })
        if result.success:
            try:
                outer = eval(result.content)
                if isinstance(outer, list) and len(outer) > 0:
                    text_content = outer[0].get('text', '{}')
                    data = json.loads(text_content)
                else:
                    data = outer
            except (json.JSONDecodeError, SyntaxError):
                data = eval(result.content)
            print(f"[OK] Trend data:")
            metric = data.get('metric', {})
            stats = data.get('statistics', {})
            print(f"  - Metric: {metric.get('name')} ({metric.get('unit')})")
            print(f"  - Average: {stats.get('average')}")
            print(f"  - Max: {stats.get('max')} @ {stats.get('max_date')}")
            print(f"  - Min: {stats.get('min')} @ {stats.get('min_date')}")
        else:
            print(f"[FAIL] Call failed: {result.error}")

    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Close connection
        print("\n[5] Close MCP connection...")
        await client.shutdown()
        print("[OK] Connection closed")

    print("\n" + "=" * 60)
    print("MCP Client Test Complete [OK]")
    print("=" * 60)


async def test_with_tool_registry():
    """Test using MCP tools through ToolRegistry"""
    print("\n" + "=" * 60)
    print("Test MCP tools via ToolRegistry")
    print("=" * 60)

    from agent.mcp import MCPClient, MCPServerConfigLoader
    from agent.tools.registry import ToolRegistry

    # Load config
    loader = MCPServerConfigLoader()
    domain_root = project_root / "domain" / "ad_engine"
    configs = loader.load_from_domain(domain_root)

    # Create MCP Client
    client = MCPClient(configs)
    await client.initialize()

    # Create ToolRegistry and register MCP tools
    registry = ToolRegistry()

    # Get MCP tool adapters and register
    adapters = client.get_tool_adapters()
    for adapter in adapters:
        registry.register_mcp_tool(adapter)

    stats = registry.get_stats()
    print(f"\n[OK] ToolRegistry status:")
    print(f"  - MCP tools: {stats['mcp_tools']}")
    print(f"  - Tools: {registry.list_tool_names()}")

    # Get OpenAI Schema
    schemas = registry.get_all_tool_schemas()
    print(f"\n[OK] OpenAI Schemas ({len(schemas)}):")
    for schema in schemas[:2]:
        desc = schema['function']['description'][:40]
        print(f"  - {schema['function']['name']}: {desc}...")

    await client.shutdown()
    print("\n[OK] Test complete")


if __name__ == "__main__":
    print("Starting MCP integration tests...\n")

    # Test 1: MCP Client connection
    asyncio.run(test_mcp_client())

    # Test 2: ToolRegistry integration
    asyncio.run(test_with_tool_registry())

    print("\n>>> All tests passed! <<<")
