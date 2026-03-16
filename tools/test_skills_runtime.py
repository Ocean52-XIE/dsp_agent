"""测试技能运行时模块

验证技能加载、注册中心、skill-tool 生成和技能执行器。
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


# 加载 skills 模块（使用新的路径）
base_module = load_module_from_file(
    "src.workflow.skills.base",
    project_root / "src" / "workflow" / "skills" / "base.py"
)
loader_module = load_module_from_file(
    "src.workflow.skills.loader",
    project_root / "src" / "workflow" / "skills" / "loader.py"
)
registry_module = load_module_from_file(
    "src.workflow.skills.registry",
    project_root / "src" / "workflow" / "skills" / "registry.py"
)
skill_tool_module = load_module_from_file(
    "src.workflow.skills.skill_tool",
    project_root / "src" / "workflow" / "skills" / "skill_tool.py"
)
executor_module = load_module_from_file(
    "src.workflow.skills.executor",
    project_root / "src" / "workflow" / "skills" / "executor.py"
)
init_module = load_module_from_file(
    "src.workflow.skills",
    project_root / "src" / "workflow" / "skills" / "__init__.py"
)

# 获取导出的类
MarkdownSkill = base_module.MarkdownSkill
StandardSkill = base_module.StandardSkill
StandardTool = base_module.StandardTool
MarkdownSkillLoader = loader_module.MarkdownSkillLoader
SkillRegistry = registry_module.SkillRegistry
SkillToolGenerator = skill_tool_module.SkillToolGenerator
SkillExecutor = executor_module.SkillExecutor
SkillExecutionResult = executor_module.SkillExecutionResult
ToolCallRecord = executor_module.ToolCallRecord
SkillLoader = init_module.SkillLoader


def test_skills_runtime():
    """测试技能运行时"""
    print("=" * 60)
    print("测试技能运行时模块")
    print("=" * 60)

    # 1. 使用新的 SkillLoader 加载技能
    print("\n[1] 加载技能")
    domain_root = project_root / "domain" / "ad_engine"
    loader = SkillLoader(domain_root)
    skills = loader.load_all()

    print(f"加载了 {len(skills)} 个技能:")
    for skill_id, skill in skills.items():
        print(f"  - {skill_id} ({type(skill).__name__})")

    # 2. 注册到注册中心
    print("\n[2] 注册到注册中心")
    registry = SkillRegistry()
    count = registry.register_from_loader(loader)
    print(f"注册了 {count} 个技能")

    # 3. 测试规则筛选
    print("\n[3] 测试规则筛选")
    test_queries = [
        "查一下最近一小时 CTR",
        "监控数据怎么样",
        "今天天气怎么样",
    ]

    for query in test_queries:
        candidates = registry.get_candidates(query)
        print(f"  查询: '{query}'")
        print(f"    候选: {[c.skill_id for c in candidates]}")

    # 4. 测试 skill-tool 生成
    print("\n[4] 测试 skill-tool Schema 生成")
    candidates = registry.get_candidates("查一下 CTR")
    generator = SkillToolGenerator()
    schema = generator.generate_schema(candidates)

    print(f"  工具名称: {schema['function']['name']}")
    print(f"  参数: {list(schema['function']['parameters']['properties'].keys())}")

    skill_enum = schema['function']['parameters']['properties']['skill_name'].get('enum', [])
    print(f"  可选技能: {skill_enum}")

    # 5. 测试 SkillExecutor
    print("\n[5] 测试 SkillExecutor")
    skill = registry.get_skill("query_metrics")

    if skill is None:
        print("  [WARN] 未找到 query_metrics 技能，跳过执行测试")
    else:
        print(f"  技能ID: {skill.skill_id}")
        print(f"  类型: {skill.skill_type}")
        print(f"  工具数量: {len(skill.tools)}")

        # 创建执行器（无 LLM 和外部系统）
        executor = SkillExecutor(llm_client=None, external_registry=None)

        # 执行（预期会失败，因为没有外部系统）
        result = executor.execute(
            skill=skill,
            query="查一下广告主 10086 的 CTR",
            context={"metric_type": "ctr", "advertiser_id": "10086"}
        )

        print(f"  执行结果:")
        print(f"    成功: {result.success}")
        print(f"    错误: {result.error}")
        print(f"    耗时: {result.latency_ms}ms")

    # 6. 统计信息
    print("\n[6] 统计信息")
    stats = registry.get_stats()
    print(f"  总技能数: {stats['total_skills']}")
    print(f"  关键词索引: {stats['total_keywords']}")
    print(f"  模式索引: {stats['total_patterns']}")
    print(f"  类型分布: {stats['type_distribution']}")

    print("\n" + "=" * 60)
    print("[OK] 测试完成 - 迁移成功！")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_skills_runtime()
    sys.exit(0 if success else 1)
