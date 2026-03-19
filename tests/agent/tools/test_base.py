# -*- coding: utf-8 -*-
"""测试 BaseTool 工具基类"""
import pytest

from agent.tools.base import BaseTool, FunctionTool


class TestFunctionTool:
    """测试 FunctionTool"""

    def test_create_from_function(self):
        """测试从函数创建工具"""
        def test_func(query: str) -> str:
            """测试函数"""
            return f"结果: {query}"

        tool = FunctionTool.from_function(
            func=test_func,
            name="test_tool",
            description="测试工具",
        )

        assert tool.name == "test_tool"
        assert tool.description == "测试工具"
        assert tool.run(query="测试") == "结果: 测试"

    def test_tool_with_parameters(self):
        """测试带参数的工具"""
        def search(query: str, top_k: int = 5) -> str:
            """搜索函数"""
            return f"找到 {top_k} 条: {query}"

        tool = FunctionTool.from_function(
            func=search,
            name="search",
            description="搜索工具",
        )

        assert tool.name == "search"
        assert tool.run(query="测试", top_k=3) == "找到 3 条: 测试"

    def test_to_openai_schema(self, sample_tool):
        """测试转换为 OpenAI Schema"""
        schema = sample_tool.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]

    def test_schema_has_correct_structure(self, sample_tool_with_params):
        """测试 Schema 结构正确"""
        schema = sample_tool_with_params.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search"

        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "query" in params["properties"]
        assert "top_k" in params["properties"]

    def test_required_parameters(self):
        """测试必需参数"""
        def func(required_param: str, optional_param: str = "default") -> str:
            return f"{required_param} - {optional_param}"

        tool = FunctionTool.from_function(
            func=func,
            name="test",
            description="测试",
        )

        schema = tool.to_openai_schema()
        required = schema["function"]["parameters"].get("required", [])

        assert "required_param" in required
        assert "optional_param" not in required

    def test_async_run(self, sample_tool):
        """测试异步执行"""
        import asyncio

        async def run_async():
            result = await sample_tool.arun(query="异步测试")
            return result

        result = asyncio.run(run_async())
        assert "异步测试" in result

    def test_run_with_kwargs(self, sample_tool_with_params):
        """测试使用关键字参数"""
        result = sample_tool_with_params.run(query="测试", top_k=10)
        assert "10" in result

    def test_tool_without_parameters(self):
        """测试无参数工具"""
        def no_params() -> str:
            return "无参数结果"

        tool = FunctionTool.from_function(
            func=no_params,
            name="no_params",
            description="无参数工具",
        )

        assert tool.run() == "无参数结果"
        schema = tool.to_openai_schema()
        assert schema["function"]["parameters"]["properties"] == {}


class TestBaseTool:
    """测试 BaseTool 抽象类"""

    def test_cannot_instantiate_directly(self):
        """测试不能直接实例化抽象类"""
        with pytest.raises(TypeError):
            BaseTool(
                name="test",
                description="测试",
                parameters={},
            )


class TestFunctionToolEdgeCases:
    """测试 FunctionTool 边界情况"""

    def test_function_with_complex_types(self):
        """测试复杂类型参数"""
        def complex_func(
            items: list[str],
            mapping: dict[str, int],
            flag: bool = False,
        ) -> str:
            return f"items={len(items)}, mapping={len(mapping)}, flag={flag}"

        tool = FunctionTool.from_function(
            func=complex_func,
            name="complex",
            description="复杂类型工具",
        )

        result = tool.run(
            items=["a", "b"],
            mapping={"x": 1},
            flag=True,
        )
        assert "items=2" in result

    def test_function_with_no_docstring(self):
        """测试无文档字符串的函数"""
        def no_doc(x: str) -> str:
            return x

        tool = FunctionTool.from_function(
            func=no_doc,
            name="no_doc",
            description="手动描述",
        )

        assert tool.description == "手动描述"

    def test_function_error_handling(self):
        """测试函数执行错误"""
        def error_func(should_error: bool = True) -> str:
            if should_error:
                raise ValueError("测试错误")
            return "成功"

        tool = FunctionTool.from_function(
            func=error_func,
            name="error",
            description="可能出错的工具",
        )

        with pytest.raises(ValueError):
            tool.run(should_error=True)

        assert tool.run(should_error=False) == "成功"
