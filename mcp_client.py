#!/usr/bin/env python3
"""
MCP协议客户端工具

功能：
- 通过MCP协议与服务器通信，测试和使用服务器的MCP接口
- 自动执行MCP协议流程：初始化连接、列出工具、调用意图识别工具
- 显示服务器的响应结果

使用方法：
python3 mcp_client.py --query "用户查询文本"
"""

from __future__ import annotations

import argparse
import json
import urllib.request
import urllib.error
import urllib.parse

def post_json(url: str, data: dict, headers: dict | None = None) -> dict:
    """发送 JSON POST 请求并返回响应"""
    if headers is None:
        headers = {}
    headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        url,
        data=json.dumps(data, ensure_ascii=False).encode("utf-8"),
        headers=headers,
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def main() -> None:
    """主函数"""
    ap = argparse.ArgumentParser(description="简单的 MCP HTTP 客户端，用于 /mcp")
    ap.add_argument("--url", default="http://localhost:8000/mcp")
    ap.add_argument("--app-id", default="", help="已发布的应用 ID")
    ap.add_argument("--environment", default="prod")
    ap.add_argument("--query", required=True)
    ap.add_argument("--visible", default="")
    ap.add_argument("--protocol", default="2025-11-25")
    args = ap.parse_args()

    # 初始化请求
    init = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": args.protocol, "capabilities": {}},
    }
    print("发送初始化请求...")
    init_res = post_json(args.url, init)
    print("initialize ->", json.dumps(init_res, ensure_ascii=False))

    # 列出工具
    headers = {"MCP-Protocol-Version": args.protocol}
    tools_list = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    print("\n发送工具列表请求...")
    tools_res = post_json(args.url, tools_list, headers=headers)
    print("tools/list ->", json.dumps(tools_res, ensure_ascii=False))

    # 调用工具
    visible = [s for s in args.visible.split(",") if s] if args.visible else None
    arguments = {"query": args.query}
    if args.app_id:
        arguments["app_id"] = args.app_id
    if args.environment:
        arguments["environment"] = args.environment
    if visible:
        arguments["visible_intent_ids"] = visible

    tools_call = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "route_intent", "arguments": arguments},
    }
    print("\n发送工具调用请求...")
    call_res = post_json(args.url, tools_call, headers=headers)
    print("tools/call ->", json.dumps(call_res, ensure_ascii=False))

if __name__ == "__main__":
    main()
