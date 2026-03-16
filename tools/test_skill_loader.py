"""测试技能加载器

验证 Markdown Skill 加载器是否正常工作。
此脚本独立运行，不依赖 workflow 模块的其他部分。
"""
import importlib.util
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)


def load_module_from_file(name: str, file_path: Path):
    """从文件路径加载 Python 模块"""
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# 加载 skill_loader 模块（绕过 workflow/__init__.py）
base_module = load_module_from_file(
    "src.workflow.skill_loader.base",
    project_root / "src" / "workflow" / "skill_loader" / "base.py"
)
loader_module = load_module_from_file(
    "src.workflow.skill_loader",
    project_root / "src" / "workflow" / "skill_loader" / "__init__.py"
)

MarkdownSkill = base_module.MarkdownSkill
StandardSkill = base_module.StandardSkill
UnifiedSkillLoader = loader_module.UnifiedSkillLoader


def test_load_skill():
    """测试加载技能"""
    print("=" * 60)
    print("测试技能加载器")
    print("=" * 60)

    # 创建加载器
    domain_root = project_root / "domain" / "ad_engine"
    loader = UnifiedSkillLoader(domain_root)

    # 加载所有技能
    skills = loader.load_all()

    print(f"\n加载了 {len(skills)} 个技能:")
    for skill_id, skill in skills.items():
        print(f"  - {skill_id} ({type(skill).__name__})")

    # 获取 query_metrics 技能
    skill = loader.get_skill("query_metrics")
    if skill is None:
        print("\n[ERROR] 未找到 query_metrics 技能")
        return False

    print(f"\n[OK] 成功加载技能: {skill.skill_id}")
    print(f"   显示名称: {skill.display_name}")
    print(f"   描述: {skill.description}")
    print(f"   版本: {skill.version}")
    print(f"   作者: {skill.author}")
    print(f"   标签: {skill.tags}")
    print(f"   启用状态: {skill.enabled}")

    # 检查是否是 MarkdownSkill
    if isinstance(skill, MarkdownSkill):
        print(f"   脚本目录: {skill.scripts_path}")
        print(f"   参考文档目录: {skill.references_path}")
        print(f"   参考文档数量: {len(skill.references)}")

        # 显示参考文档列表
        if skill.references:
            print("   参考文档:")
            for ref in skill.references:
                print(f"     - {ref.name}")

    # 显示工具列表
    print(f"\n   工具数量: {len(skill.tools)}")
    for tool in skill.tools:
        print(f"     - {tool.name}: {tool.description[:50]}...")
        print(f"       参数: {list(tool.parameters.keys())}")
        print(f"       外部系统: {tool.handler_config.get('external_system', 'N/A')}")

    # 显示触发配置
    print(f"\n   触发配置:")
    trigger = skill.trigger
    if trigger:
        keywords = trigger.get("keywords", [])
        patterns = trigger.get("patterns", [])
        print(f"     关键词: {keywords}")
        print(f"     模式: {patterns}")

    # 显示提示词模板（前 200 字符）
    print(f"\n   提示词模板（前 200 字符）:")
    print(f"     {skill.prompt_template[:200]}...")

    # 测试 OpenAI Schema 生成
    print("\n" + "=" * 60)
    print("测试 OpenAI Schema 生成")
    print("=" * 60)

    schemas = loader.get_openai_tools_schema()
    print(f"\n生成了 {len(schemas)} 个工具 Schema:")
    for schema in schemas:
        func = schema.get("function", {})
        print(f"  - {func.get('name')}: {func.get('description', '')[:50]}...")

    # 测试关键词匹配
    print("\n" + "=" * 60)
    print("测试关键词匹配")
    print("=" * 60)

    test_queries = [
        "查一下最近一小时 CTR",
        "监控数据怎么样",
        "今天天气怎么样",  # 不应匹配
    ]

    for query in test_queries:
        matched = loader.get_skill_by_keyword(query)
        if matched:
            print(f"  '{query}' -> 匹配技能: {matched.skill_id}")
        else:
            print(f"  '{query}' -> 未匹配")

    # 测试模式匹配
    print("\n" + "=" * 60)
    print("测试模式匹配")
    print("=" * 60)

    test_texts = [
        "帮我查一下今天的指标",
        "看看最近一天的表现",
        "这个监控数据有问题",
        "今天吃什么",  # 不应匹配
    ]

    for text in test_texts:
        matched = loader.get_skill_by_pattern(text)
        if matched:
            print(f"  '{text}' -> 匹配技能: {matched.skill_id}")
        else:
            print(f"  '{text}' -> 未匹配")

    print("\n" + "=" * 60)
    print("[OK] 测试完成")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_load_skill()
    sys.exit(0 if success else 1)
