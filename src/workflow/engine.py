from __future__ import annotations

"""鍩轰簬 LangGraph 鐨勫杞樁娈靛紡鏅鸿兘缂栨帓宸ヤ綔娴併€?

杩欎竴鐗堝伐浣滄祦鐩歌緝浜庢渶鍒濈殑鈥滃崟杞垎绫烩€濆疄鐜帮紝鏈変袱涓叧閿彉鍖栵細

1. `intent` 浠嶇劧鎸夆€滄瘡涓€杞敤鎴疯緭鍏モ€濋噸鏂板垽鏂紝閬垮厤鎶婃暣鍦轰細璇濆浐瀹氭鍦ㄦ煇涓€绉嶇被鍨嬩笂銆?
2. `task_stage` 浣滀负鈥滆法杞换鍔＄姸鎬佲€濅繚鐣欎笅鏉ワ紝鐢ㄤ簬琛ㄨ揪褰撳墠浼氳瘽姝ｅ湪缁忓巻
   `knowledge_qa -> issue_analysis -> confirm_code -> code_generation`
   杩欐牱鐨勯樁娈靛崌绾ц繃绋嬨€?

鑺傜偣鍐呴儴褰撳墠渚濇棫鏄?mock 鏁版嵁瀹炵幇锛屼絾鐘舵€佸瓧娈点€佽浆鍦洪€昏緫銆佹潯浠跺垎鏀拰鏈€缁堣緭鍑烘牸寮?
閮藉凡缁忔寜鐓х湡瀹炲彲鎵╁睍鐨勫伐浣滄祦鏂瑰紡缁勭粐锛屽悗缁彧闇€瑕侀€愭鏇挎崲鑺傜偣鍐呴儴鑳藉姏鍗冲彲銆?
"""

import os
from time import perf_counter
from pathlib import Path
from typing import Any, Callable, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from workflow.nodes.code_generation import run as code_generation_node
from workflow.nodes.conversation_transition import run as conversation_transition_node
from workflow.nodes.decline_code_generation_response import run as decline_code_generation_response_node
from workflow.nodes.domain_gate import run as domain_gate_node
from workflow.nodes.entry_router import run as entry_router_node
from workflow.nodes.finalize_response import run as finalize_response_node
from workflow.nodes.fix_plan import run as fix_plan_node
from workflow.nodes.intent_classifier import run as intent_classifier_node
from workflow.nodes.issue_localizer import run as issue_localizer_node
from workflow.nodes.knowledge_answer import run as knowledge_answer_node
from workflow.nodes.knowledge_answer.llm_qa import KnowledgeQALLMClient
from workflow.nodes.load_code_context import run as load_code_context_node
from workflow.nodes.load_context import run as load_context_node
from workflow.nodes.merge_evidence import run as merge_evidence_node
from workflow.nodes.out_of_scope_response import run as out_of_scope_response_node
from workflow.nodes.query_rewriter import run as query_rewriter_node
from workflow.nodes.retrieve_cases import run as retrieve_cases_node
from workflow.nodes.retrieve_code import run as retrieve_code_node
from workflow.nodes.retrieve_code.code_retriever import LocalCodeRetriever, parse_code_dirs_from_env
from workflow.nodes.retrieve_code_context import run as retrieve_code_context_node
from workflow.nodes.retrieve_wiki import run as retrieve_wiki_node
from workflow.nodes.retrieve_wiki.wiki_retriever import MarkdownWikiRetriever
from workflow.nodes.root_cause_analysis import run as root_cause_analysis_node
from workflow.domain_profile import DomainProfile, load_domain_profile
from workflow.runtime_logging import get_file_logger


def _env_bool(name: str, default: bool = False) -> bool:
    """璇诲彇 bool 鐜鍙橀噺骞跺仛鍏滃簳銆?"""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default

# 褰撳墠宸ョ▼宸茬粡鍒囨崲鎴愮湡瀹?LangGraph锛岃繖閲岀粺涓€鏍囪瘑鍚庣绫诲瀷銆?
BACKEND_NAME = "langgraph"

# 褰撳墠宸ョ▼宸茬粡鍒囨崲鎴愮湡瀹?langgraph锛屽洜姝よ繖閲屾槑纭爣璇嗗悗绔被鍨嬶紝
# 鍓嶇璋冭瘯闈㈡澘涔熶細鐩存帴灞曠ず杩欎釜鍊笺€?
BACKEND_NAME = "langgraph"


BACKEND_NAME = "langgraph"


class WorkflowState(TypedDict, total=False):
    """LangGraph 鍦ㄨ妭鐐归棿娴佽浆鐨勫叡浜姸鎬併€?

    杩欎唤鐘舵€佸悓鏃舵壙鎷呬笁绫昏亴璐ｏ細

    1. 琛ㄨ揪鈥滄湰杞姹傗€濈殑鍒嗙被鍜屾墽琛屼俊鎭€?
    2. 琛ㄨ揪鈥滆法杞細璇濃€濈殑浠诲姟闃舵鍜屼笂涓嬫枃璁板繂銆?
    3. 琛ㄨ揪鈥滄渶缁堝搷搴斺€濈殑缁撴瀯鍖栬緭鍑猴紝渚夸簬 API 灞傚拰鍓嶇鐩存帴娑堣垂銆?
    """

    # 褰撳墠鎵ц妯″紡锛?
    # - message锛氭櫘閫氱敤鎴锋秷鎭紝闇€閲嶆柊鍋氶鍩熷垽瀹氥€佹剰鍥捐瘑鍒拰杞満鍒ゆ柇銆?
    # - code_generation锛氶€氳繃涓撻棬纭鎺ュ彛鎭㈠锛岀洿鎺ヨ繘鍏ヤ唬鐮佺敓鎴愰摼璺€?
    mode: str
    # 鍗曡疆璋冪敤閾捐矾 ID锛屼究浜庤皟璇曘€佹棩蹇椾覆鑱斿拰寮曠敤杩借釜銆?
    trace_id: str
    # 褰撳墠浼氳瘽 ID锛岀敱 API 灞備紶鍏ャ€?
    session_id: str
    # 鏈疆鐢ㄦ埛杈撳叆鍘熸枃銆?
    user_query: str
    # 褰撳墠浼氳瘽鍘嗗彶娑堟伅锛屼緵澶氳疆涓婁笅鏂囨仮澶嶄娇鐢ㄣ€?
    history: list[dict[str, Any]]
    # 鍦ㄢ€滅‘璁ょ敓鎴愪唬鐮佲€濇帴鍙ｈ矾寰勪腑锛屼笂涓€鏉￠棶棰樺垎鏋愭秷鎭細琚洿鎺ュ甫鍏ュ浘銆?
    source_message: dict[str, Any]

    # 鏈疆閲嶆柊鍒ゆ柇鍑虹殑鍩虹鎰忓浘銆?
    route: str
    # 鏈疆鐪熸鎵ц鐨勫浘璺緞锛氭绱€佷唬鐮佺敓鎴愭垨鎺у埗鍝嶅簲銆?
    execution_path: str
    # 褰撳墠杞浉瀵逛笂涓€杞殑闃舵杞満绫诲瀷銆?
    transition_type: str
    # 鏈疆鎵ц瀹屾垚鍚庯紝浼氳瘽搴斿浜庡摢涓换鍔￠樁娈点€?
    task_stage: str

    # 褰撳墠鍝嶅簲鐘舵€佷笌娑堟伅鍏冧俊鎭€?
    status: str
    response_kind: str
    next_action: str

    # 棰嗗煙鐩稿叧鎬у垎鏁颁笌棰嗗煙鍒ゆ柇缁撴灉銆?
    domain_relevance: float
    is_domain_related: bool

    # 褰撳墠杞帹鏂嚭鐨勬ā鍧椾笌璇存槑銆?
    module_name: str
    module_hint: str

    # 璺ㄨ疆鎭㈠鍑虹殑娲诲姩涓婚涓庝换鍔′笂涓嬫枃銆?
    active_topic: str
    active_topic_source: str
    active_task_stage: str
    active_module_name: str
    active_module_hint: str
    active_qa_context: dict[str, Any] | None
    active_issue_context: dict[str, Any] | None
    last_analysis_result: dict[str, Any] | None
    last_analysis_citations: list[dict[str, Any]]
    pending_action: str

    # 鍘嗗彶鎽樿涓庢绱腑闂寸粨鏋溿€?
    history_summary: str
    retrieval_queries: list[str]
    retrieval_plan: dict[str, Any]
    wiki_hits: list[dict[str, Any]]
    wiki_retrieval_grade: str
    wiki_retrieval_profile: dict[str, Any]
    case_hits: list[dict[str, Any]]
    case_retrieval_grade: str
    case_retrieval_profile: dict[str, Any]
    code_hits: list[dict[str, Any]]
    code_retrieval_grade: str
    code_retrieval_profile: dict[str, Any]
    citations: list[dict[str, Any]]
    evidence_fusion_profile: dict[str, Any]

    # 缁熶竴鍒嗘瀽瀵硅薄涓庢渶缁堣緭鍑恒€?
    analysis: dict[str, Any] | None
    answer: str
    node_trace: list[dict[str, str]]
    assistant_message: dict[str, Any]


class WorkflowService:
    """澶氳疆闃舵寮?LangGraph 宸ヤ綔娴佹湇鍔°€?

    璁捐鐩爣涓嶆槸鎶婃瘡涓€杞秷鎭兘褰撴垚褰兼鐙珛鐨勮姹傦紝鑰屾槸锛?

    - 姣忚疆閮介噸鏂板垽鏂?`intent`锛岀‘淇濆綋鍓嶈緭鍏ヨ姝ｇ‘鐞嗚В锛?
    - 鍚屾椂浠庡巻鍙叉秷鎭仮澶?`task_stage`锛屾敮鎸佽瘽棰樺欢缁€侀樁娈靛崌绾у拰鍒囨崲涓婚锛?
    - 骞朵繚璇佷唬鐮佺敓鎴愬缁堜緷璧栧墠缃垎鏋愮粨鏋滐紝鑰屼笉鏄粎闈犱竴鍙モ€滅粰鎴戜唬鐮佲€濈洿鎺ヨ繘鍏ャ€?
    """

    def __init__(self) -> None:
        self.backend_name = BACKEND_NAME
        # 绯荤粺璋冭瘯寮€鍏筹紙榛樿鍏抽棴锛夛細
        # - false锛氫繚鎸佸綋鍓嶇簿绠€ debug 杈撳嚭锛?
        # - true锛氬湪鏈€缁堝搷搴斾腑闄勫姞 debug.verbose 鎵╁睍璋冭瘯淇℃伅銆?
        self.debug_verbose_enabled = _env_bool("WORKFLOW_DEBUG_VERBOSE", default=False)
        project_root = Path(__file__).resolve().parents[2]
        self._file_logger = get_file_logger(project_root=project_root)
        self.domain_profile: DomainProfile = load_domain_profile(project_root=project_root)

        # 鐪熷疄 Wiki 妫€绱㈠櫒锛氱洿鎺ヨ鍙栦粨搴撳唴鐨?Markdown 鏂囨。浣滀负璇枡銆?
        # 杩欓噷鍦ㄦ湇鍔″垵濮嬪寲鏃舵瀯寤鸿交閲忕储寮曪紝鍚庣画姣忔璇锋眰鐩存帴妫€绱紝閬垮厤閲嶅鎵洏銆?
        wiki_dir = self.domain_profile.resolve_wiki_dir(project_root)
        self._wiki_retriever = MarkdownWikiRetriever(
            wiki_dir=wiki_dir,
            project_root=project_root,
            default_top_k=4,
            module_doc_hints=self.domain_profile.module_doc_hints(),
        )

        # 鐪熷疄浠ｇ爜妫€绱㈠櫒锛?
        # - 榛樿绱㈠紩浠撳簱鏍圭洰褰?`codes/`锛堝彲閫氳繃鐜鍙橀噺 WORKFLOW_CODE_RETRIEVER_DIRS 瑕嗙洊锛夛紱
        # - 涓嶅啀榛樿鎵弿鏁翠釜浠撳簱锛岄伩鍏嶆妸宸ュ叿宸ョ▼鏂囦欢璇撼鍏ユ绱㈣寖鍥达紱
        # - 閲囩敤 Parent/Child 娣峰悎鍙洖锛岃繑鍥炲彲瀹氫綅鐨勪唬鐮佽瘉鎹€?
        env_code_dirs = os.getenv("WORKFLOW_CODE_RETRIEVER_DIRS", "").strip()
        if env_code_dirs:
            code_dirs = parse_code_dirs_from_env(project_root=project_root)
        else:
            code_dirs = self.domain_profile.resolve_code_roots(project_root)
        self._code_retriever = LocalCodeRetriever(
            project_root=project_root,
            code_dirs=code_dirs,
            default_top_k=4,
        )

        # knowledge_answer 鑺傜偣浣跨敤鐨?LLM 瀹㈡埛绔細
        # - 鏈厤缃?API Key 鏃朵笉浼氫腑鏂祦绋嬶紱
        # - 鑺傜偣鍐呴儴浼氳嚜鍔ㄩ檷绾у埌瑙勫垯鍥炵瓟銆?
        self._knowledge_qa_llm = KnowledgeQALLMClient.from_env(domain_profile=self.domain_profile)
        self._checkpointer = MemorySaver()
        self._graph = self._build_graph()
        self._file_logger.info(
            "workflow.service.initialized",
            backend=self.backend_name,
            domain_profile=self.domain_profile.profile_id,
            domain_display_name=self.domain_profile.display_name,
            debug_verbose_enabled=self.debug_verbose_enabled,
            logger_status=self._file_logger.status(),
            wiki_dir=str(wiki_dir),
            code_dirs=[str(path) for path in getattr(self._code_retriever, "code_dirs", [])],
            checkpointer_type=type(self._checkpointer).__name__,
        )

    def run_user_message(
        self,
        *,
        session_id: str,
        trace_id: str,
        user_query: str,
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """澶勭悊鏅€氱敤鎴锋秷鎭€?

        杩欐潯鍏ュ彛瀵瑰簲鍓嶇杈撳叆妗嗙洿鎺ュ彂鍑虹殑娑堟伅銆傛瘡娆¤繘鍏ヨ繖閲岋紝閮戒唬琛ㄨ閲嶆柊鍋氫竴杞細

        1. 浼氳瘽涓婁笅鏂囨仮澶嶏紱
        2. 棰嗗煙鍒ゅ畾锛?
        3. 鎰忓浘璇嗗埆锛?
        4. 闃舵杞満鍒ゆ柇锛?
        5. 杩涘叆鐭ヨ瘑闂瓟銆侀棶棰樺垎鏋愭垨浠ｇ爜鐢熸垚閾捐矾銆?
        """
        state: WorkflowState = {
            "mode": "message",
            "trace_id": trace_id,
            "session_id": session_id,
            "user_query": user_query.strip(),
            "history": history,
            "node_trace": [],
        }
        return self._invoke(state)

    def run_code_generation(
        self,
        *,
        session_id: str,
        trace_id: str,
        source_message: dict[str, Any] | None,
        history: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """浠庨棶棰樺垎鏋愮粨鏋滄仮澶嶏紝鐩存帴杩涘叆浠ｇ爜鐢熸垚閾捐矾銆?

        杩欐潯鍏ュ彛淇濈暀缁欌€滄寜閽‘璁ょ户缁敓鎴愪唬鐮佲€濈殑鍦烘櫙銆傚畠涓嶄細閲嶆柊鍋氶鍩熷垽瀹氬拰妫€绱㈤棶绛旓紝
        鑰屾槸鎶婁笂涓€鏉￠棶棰樺垎鏋愮粨璁哄綋浣滃彲淇″墠缃姸鎬侊紝鐩存帴缁х画鍚戝悗鎵ц銆?
        """
        resolved_source_message = dict(source_message or {})
        resolved_history = list(history or [])
        if not resolved_source_message or not resolved_history:
            checkpoint_state = self._load_checkpoint_state(session_id=session_id, mode="code_generation")
            if checkpoint_state:
                if not resolved_source_message:
                    latest_assistant = checkpoint_state.get("assistant_message") or {}
                    resolved_source_message = dict(latest_assistant)
                if not resolved_history:
                    resolved_history = list(checkpoint_state.get("history", []) or [])

        state: WorkflowState = {
            "mode": "code_generation",
            "trace_id": trace_id,
            "session_id": session_id,
            "user_query": resolved_source_message.get("content", ""),
            "source_message": resolved_source_message,
            "history": resolved_history,
            "node_trace": [],
        }
        return self._invoke(state)

    def _invoke(self, state: WorkflowState) -> dict[str, Any]:
        """鎵ц鍥惧苟琛ラ綈缁熶竴璋冭瘯淇℃伅銆?"""
        started_at = perf_counter()
        self._file_logger.info(
            "workflow.invoke.start",
            trace_id=state.get("trace_id", ""),
            session_id=state.get("session_id", ""),
            mode=state.get("mode", "message"),
            history_size=len(state.get("history", []) or []),
            user_query_preview=self._preview_text(state.get("user_query", ""), max_chars=120),
        )
        try:
            result = self._graph.invoke(state, config=self._invoke_config(state))
        except Exception as exc:
            latency_ms = int((perf_counter() - started_at) * 1000)
            self._file_logger.exception(
                "workflow.invoke.exception",
                trace_id=state.get("trace_id", ""),
                session_id=state.get("session_id", ""),
                mode=state.get("mode", "message"),
                latency_ms=latency_ms,
                error_type=type(exc).__name__,
            )
            raise
        assistant_message = dict(result["assistant_message"])
        latency_ms = int((perf_counter() - started_at) * 1000)

        # 娉ㄦ剰锛歞ebug 瀛楁鐜板湪鐢扁€滅郴缁熻皟璇曞紑鍏斥€濇帶鍒讹紝榛樿鍙兘涓嶅瓨鍦ㄣ€?
        # 鍥犳杩欓噷鍙湪 debug 宸插瓨鍦ㄤ笖鏄?dict 鐨勬儏鍐典笅琛?latency锛岄伩鍏?KeyError銆?
        debug_payload = assistant_message.get("debug")
        if isinstance(debug_payload, dict):
            debug_payload["latency_ms"] = latency_ms
        self._file_logger.info(
            "workflow.invoke.complete",
            trace_id=state.get("trace_id", ""),
            session_id=state.get("session_id", ""),
            mode=state.get("mode", "message"),
            latency_ms=latency_ms,
            response_kind=assistant_message.get("kind", "unknown"),
            response_status=assistant_message.get("status", "unknown"),
            next_action=(assistant_message.get("actions") or []),
            citation_count=len(assistant_message.get("citations", []) or []),
            node_trace_count=len(result.get("node_trace", []) or []),
        )
        return assistant_message

    def _invoke_config(self, state: WorkflowState) -> dict[str, Any]:
        session_id = str(state.get("session_id", "") or "").strip() or "default_session"
        return {
            "configurable": {
                "thread_id": session_id,
            }
        }

    def _load_checkpoint_state(self, *, session_id: str, mode: str) -> dict[str, Any] | None:
        try:
            snapshot = self._graph.get_state(
                {
                    "configurable": {
                        "thread_id": session_id,
                    }
                }
            )
        except Exception:
            return None
        values = getattr(snapshot, "values", None)
        return dict(values) if isinstance(values, dict) else None

    def _build_graph(self) -> Any:
        """瀹氫箟澶氳疆闃舵寮忓浘缁撴瀯銆?

        鍥炬湁涓ゆ潯澶ц矾寰勶細

        1. `message`锛氭櫘閫氭秷鎭矾寰勶紝鍏堟仮澶嶄笂涓嬫枃锛屽啀鍒ゆ柇杩欒疆鏄欢缁€佸崌绾ц繕鏄垏鎹富棰樸€?
        2. `code_generation`锛氫粠纭鑺傜偣鎭㈠鐨勫揩鎹疯矾寰勶紝鐩存帴瑁呴厤鍒嗘瀽涓婁笅鏂囧苟鐢熸垚浠ｇ爜寤鸿銆?
        """
        graph = StateGraph(WorkflowState)

        # 鍏ュ彛涓庝笂涓嬫枃鎭㈠鑺傜偣銆?
        graph.add_node("entry_router", self._entry_router)
        graph.add_node("load_context", self._load_context)
        graph.add_node("domain_gate", self._domain_gate)
        graph.add_node("intent_classifier", self._intent_classifier)
        graph.add_node("conversation_transition", self._conversation_transition)

        # 閫氱敤妫€绱笌璇佹嵁鑱氬悎鑺傜偣銆?
        graph.add_node("query_rewriter", self._query_rewriter)
        graph.add_node("retrieve_wiki", self._retrieve_wiki)
        graph.add_node("retrieve_cases", self._retrieve_cases)
        graph.add_node("retrieve_code", self._retrieve_code)
        graph.add_node("merge_evidence", self._merge_evidence)

        # 鐭ヨ瘑闂瓟涓庨棶棰樺垎鏋愯妭鐐广€?
        graph.add_node("knowledge_answer", self._knowledge_answer)
        graph.add_node("issue_localizer", self._issue_localizer)
        graph.add_node("root_cause_analysis", self._root_cause_analysis)
        graph.add_node("fix_plan", self._fix_plan)

        # 鐗规畩鍝嶅簲鑺傜偣銆?
        graph.add_node("decline_code_generation_response", self._decline_code_generation_response)
        graph.add_node("out_of_scope_response", self._out_of_scope_response)

        # 浠ｇ爜鐢熸垚閾捐矾鑺傜偣銆?
        graph.add_node("load_code_context", self._load_code_context)
        graph.add_node("retrieve_code_context", self._retrieve_code_context)
        graph.add_node("code_generation", self._code_generation)

        # 鏀舵暃杈撳嚭鑺傜偣銆?
        graph.add_node("finalize_response", self._finalize_response)

        # 鍥惧叆鍙ｏ細鍏堢湅鏈疆鏄櫘閫氭秷鎭繕鏄‘璁ゅ悗鐨勪唬鐮佺敓鎴愭仮澶嶃€?
        graph.add_edge(START, "entry_router")
        graph.add_conditional_edges(
            "entry_router",
            self._route_by_mode,
            {
                "message": "load_context",
                "code_generation": "load_code_context",
            },
        )

        # 鏅€氭秷鎭矾寰勶細
        # 鍏堟仮澶嶄細璇濅笂涓嬫枃锛屽啀鍋氶鍩熷垽瀹氾紱鍙湁杩涘叆涓氬姟鍩燂紝鎵嶇户缁仛鎰忓浘璇嗗埆鍜岄樁娈佃浆鍦恒€?
        graph.add_edge("load_context", "domain_gate")
        graph.add_conditional_edges(
            "domain_gate",
            self._route_by_domain_gate,
            {
                "in_scope": "intent_classifier",
                "out_of_scope": "out_of_scope_response",
            },
        )
        graph.add_edge("intent_classifier", "conversation_transition")
        graph.add_conditional_edges(
            "conversation_transition",
            self._route_by_execution_path,
            {
                "retrieval_flow": "query_rewriter",
                "code_generation_flow": "load_code_context",
                "decline_code_flow": "decline_code_generation_response",
            },
        )

        # 閫氱敤妫€绱㈤摼璺€傜煡璇嗛棶绛斿拰闂鍒嗘瀽鍏辩敤妫€绱紝鍙湪璇佹嵁铻嶅悎涔嬪悗鍐嶅垎鍙夈€?
        graph.add_edge("query_rewriter", "retrieve_wiki")
        graph.add_edge("retrieve_wiki", "retrieve_cases")
        graph.add_edge("retrieve_cases", "retrieve_code")
        graph.add_edge("retrieve_code", "merge_evidence")
        graph.add_conditional_edges(
            "merge_evidence",
            self._route_by_intent,
            {
                "knowledge_qa": "knowledge_answer",
                "issue_analysis": "issue_localizer",
            },
        )

        # 闂瓟鍜岄棶棰樺垎鏋愮殑鍚庡崐娈点€?
        graph.add_edge("knowledge_answer", "finalize_response")
        graph.add_edge("issue_localizer", "root_cause_analysis")
        graph.add_edge("root_cause_analysis", "fix_plan")
        graph.add_edge("fix_plan", "finalize_response")
        graph.add_edge("decline_code_generation_response", "finalize_response")
        graph.add_edge("out_of_scope_response", "finalize_response")

        # 浠ｇ爜鐢熸垚璺緞杈冪煭锛氳閰嶅垎鏋愪笂涓嬫枃 -> 妫€绱唬鐮佷笂涓嬫枃 -> 鐢熸垚 -> 鏀舵暃銆?
        graph.add_edge("load_code_context", "retrieve_code_context")
        graph.add_edge("retrieve_code_context", "code_generation")
        graph.add_edge("code_generation", "finalize_response")
        graph.add_edge("finalize_response", END)

        return graph.compile(checkpointer=self._checkpointer)

    def _entry_router(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("entry_router", entry_router_node, state)

    def _load_context(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("load_context", load_context_node, state)

    def _domain_gate(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("domain_gate", domain_gate_node, state)

    def _intent_classifier(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("intent_classifier", intent_classifier_node, state)

    def _conversation_transition(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("conversation_transition", conversation_transition_node, state)

    def _query_rewriter(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("query_rewriter", query_rewriter_node, state)

    def _retrieve_wiki(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("retrieve_wiki", retrieve_wiki_node, state)

    def _retrieve_cases(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("retrieve_cases", retrieve_cases_node, state)

    def _retrieve_code(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("retrieve_code", retrieve_code_node, state)

    def _merge_evidence(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("merge_evidence", merge_evidence_node, state)

    def _knowledge_answer(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("knowledge_answer", knowledge_answer_node, state)

    def _issue_localizer(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("issue_localizer", issue_localizer_node, state)

    def _root_cause_analysis(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("root_cause_analysis", root_cause_analysis_node, state)

    def _fix_plan(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("fix_plan", fix_plan_node, state)

    def _decline_code_generation_response(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("decline_code_generation_response", decline_code_generation_response_node, state)

    def _out_of_scope_response(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("out_of_scope_response", out_of_scope_response_node, state)

    def _load_code_context(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("load_code_context", load_code_context_node, state)

    def _retrieve_code_context(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("retrieve_code_context", retrieve_code_context_node, state)

    def _code_generation(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("code_generation", code_generation_node, state)

    def _finalize_response(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("finalize_response", finalize_response_node, state)

    def _route_by_mode(self, state: WorkflowState) -> str:
        """entry_router 鐨勬潯浠跺垎鏀嚱鏁般€?"""
        route = state.get("mode", "message")
        self._file_logger.debug(
            "workflow.route.mode",
            trace_id=state.get("trace_id", ""),
            session_id=state.get("session_id", ""),
            route=route,
        )
        return route

    def _route_by_domain_gate(self, state: WorkflowState) -> str:
        """domain_gate 鐨勬潯浠跺垎鏀嚱鏁般€?"""
        route = "in_scope" if state.get("is_domain_related") else "out_of_scope"
        self._file_logger.debug(
            "workflow.route.domain_gate",
            trace_id=state.get("trace_id", ""),
            session_id=state.get("session_id", ""),
            route=route,
            domain_relevance=state.get("domain_relevance", 0.0),
        )
        return route

    def _route_by_execution_path(self, state: WorkflowState) -> str:
        """鏍规嵁杞満鑺傜偣鐨勫喅瀹氾紝杩涘叆妫€绱€佷唬鐮佺敓鎴愭垨鎺у埗鍝嶅簲璺緞銆?"""
        route = state.get("execution_path", "retrieval_flow")
        self._file_logger.debug(
            "workflow.route.execution_path",
            trace_id=state.get("trace_id", ""),
            session_id=state.get("session_id", ""),
            route=route,
            transition_type=state.get("transition_type", "unknown"),
        )
        return route

    def _route_by_intent(self, state: WorkflowState) -> str:
        """merge_evidence 涔嬪悗鎸夋剰鍥鹃€夋嫨鏈€缁堝垎鏀€?"""
        route = state["route"]
        self._file_logger.debug(
            "workflow.route.intent",
            trace_id=state.get("trace_id", ""),
            session_id=state.get("session_id", ""),
            route=route,
            citation_count=len(state.get("citations", []) or []),
        )
        return route

    def _preview_text(self, value: Any, *, max_chars: int = 80) -> str:
        text = str(value or "").strip()
        if len(text) <= max_chars:
            return text
        return f"{text[:max_chars]}..."

    def _summarize_node_updates(self, updates: dict[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        scalar_keys = (
            "route",
            "execution_path",
            "transition_type",
            "task_stage",
            "status",
            "next_action",
            "response_kind",
            "module_name",
            "module_hint",
            "active_task_stage",
            "active_topic_source",
            "is_domain_related",
            "domain_relevance",
            "wiki_retrieval_grade",
            "case_retrieval_grade",
            "code_retrieval_grade",
        )
        for key in scalar_keys:
            if key not in updates:
                continue
            value = updates.get(key)
            if isinstance(value, str):
                summary[key] = self._preview_text(value, max_chars=120)
            else:
                summary[key] = value

        if "retrieval_queries" in updates:
            summary["retrieval_query_count"] = len(updates.get("retrieval_queries", []) or [])
        if "wiki_hits" in updates:
            summary["wiki_hit_count"] = len(updates.get("wiki_hits", []) or [])
        if "case_hits" in updates:
            summary["case_hit_count"] = len(updates.get("case_hits", []) or [])
        if "code_hits" in updates:
            summary["code_hit_count"] = len(updates.get("code_hits", []) or [])
        if "citations" in updates:
            summary["citation_count"] = len(updates.get("citations", []) or [])
        if "node_trace" in updates:
            summary["node_trace_count"] = len(updates.get("node_trace", []) or [])
        if "assistant_message" in updates:
            assistant_message = dict(updates.get("assistant_message") or {})
            summary["assistant_kind"] = assistant_message.get("kind", "unknown")
            summary["assistant_status"] = assistant_message.get("status", "unknown")
            summary["assistant_action_count"] = len(assistant_message.get("actions", []) or [])
            summary["assistant_citation_count"] = len(assistant_message.get("citations", []) or [])
        return summary

    def _run_node(
        self,
        node_name: str,
        node_runner: Callable[[Any, dict[str, Any]], dict[str, Any]],
        state: WorkflowState,
    ) -> dict[str, Any]:
        started_at = perf_counter()
        self._file_logger.debug(
            "workflow.node.start",
            trace_id=state.get("trace_id", ""),
            session_id=state.get("session_id", ""),
            node=node_name,
            route=state.get("route", ""),
            task_stage=state.get("task_stage", ""),
        )
        try:
            updates = node_runner(self, state)
        except Exception as exc:
            latency_ms = int((perf_counter() - started_at) * 1000)
            self._file_logger.exception(
                "workflow.node.exception",
                trace_id=state.get("trace_id", ""),
                session_id=state.get("session_id", ""),
                node=node_name,
                latency_ms=latency_ms,
                error_type=type(exc).__name__,
            )
            raise
        latency_ms = int((perf_counter() - started_at) * 1000)
        self._file_logger.info(
            "workflow.node.complete",
            trace_id=state.get("trace_id", ""),
            session_id=state.get("session_id", ""),
            node=node_name,
            latency_ms=latency_ms,
            updates=self._summarize_node_updates(updates),
        )
        return updates

    def runtime_log_status(self) -> dict[str, Any]:
        return self._file_logger.status()

    def _trace(self, state: WorkflowState, node: str, summary: str) -> list[dict[str, str]]:
        """鍦ㄧ姸鎬佷腑杩藉姞鑺傜偣杞ㄨ抗锛屼緵鍓嶇 debug 闈㈡澘娓叉煋銆?"""
        return [*state.get("node_trace", []), {"node": node, "summary": summary}]

    def _build_history_summary(self, history: list[dict[str, Any]]) -> str:
        """鏋勯€犺交閲忓巻鍙叉憳瑕併€?"""
        recent_questions = [
            message.get("content", "")[:24]
            for message in history
            if message.get("role") == "user"
        ][-4:]
        return " / ".join(recent_questions) if recent_questions else "暂无历史用户提问"

    def _latest_message_by(
        self,
        history: list[dict[str, Any]],
        predicate: Any,
    ) -> dict[str, Any] | None:
        """鍊掑簭鏌ユ壘婊¤冻鏉′欢鐨勬渶杩戜竴鏉℃秷鎭€?"""
        for message in reversed(history):
            if predicate(message):
                return message
        return None

    def _derive_task_stage(self, message: dict[str, Any] | None) -> str:
        """鏍规嵁鏈€杩戜竴鏉″叧閿姪鎵嬫秷鎭仮澶嶄細璇濆綋鍓嶉樁娈点€?"""
        if not message:
            return "idle"
        if message.get("status") == "confirm_code":
            return "confirm_code"
        if message.get("kind") == "code_generation":
            return "code_generation"
        if message.get("intent") == "issue_analysis":
            return "issue_analysis"
        if message.get("intent") == "knowledge_qa":
            return "knowledge_qa"
        return "idle"

    def _extract_module_from_message(self, message: dict[str, Any] | None) -> tuple[str, str]:
        """浠庡巻鍙插垎鏋愭垨闂瓟娑堟伅涓仮澶嶆椿鍔ㄦā鍧椼€?"""
        if not message:
            return "", ""
        analysis = message.get("analysis") or {}
        module_name = analysis.get("module", "")
        module_hint = self._infer_module(module_name)[1] if module_name else ""
        return module_name, module_hint

    def _is_pronoun_followup(self, text: str) -> bool:
        """识别“它/这个/那块”这类依赖上下文的追问。"""
        pronouns = ("它", "这个", "这个问题", "那个", "那这个", "这块", "这里", "上面这个")
        return any(token in text for token in pronouns)

    def _looks_like_code_generation_request(self, text: str) -> bool:
        """识别用户是否在请求进入代码实现阶段。"""
        terms = (
            "给我代码",
            "给出代码",
            "直接给代码",
            "代码实现",
            "实现一个",
            "写一下代码",
            "补丁",
            "patch",
            "改代码",
            "修改代码",
            "直接修",
        )
        lowered = text.lower()
        return any(term in text or term in lowered for term in terms)

    def _looks_like_decline_code_request(self, text: str) -> bool:
        """识别用户是否明确表示暂不进入代码实现。"""
        terms = ("不用代码", "先不用", "暂不需要", "不用了", "不需要代码", "先不要代码")
        return any(term in text for term in terms)

    def _is_same_topic(self, current_module: str, active_module: str | None, user_query: str) -> bool:
        """鍒ゆ柇褰撳墠杞槸鍚︿粛鍦ㄥ欢缁笂涓€涓富棰樸€?"""
        if active_module and current_module == active_module:
            return True
        if active_module and self._is_pronoun_followup(user_query):
            return True
        return False

    def _infer_module(self, text: str) -> tuple[str, str]:
        """Infer module from the active domain profile."""
        default_module = self.domain_profile.default_module
        default_hint = self.domain_profile.module_hint(default_module)
        if not text:
            return default_module, default_hint

        lowered = text.lower()
        modules = sorted(self.domain_profile.modules, key=lambda item: item.route_priority)

        # Symbol-level routing takes precedence for code-location style queries.
        for module in modules:
            if module.symbol_keywords and any(token.lower() in lowered for token in module.symbol_keywords):
                return module.name, module.hint

        best_module_name = default_module
        best_module_hint = default_hint
        best_score = 0
        best_priority = 10**9
        for module in modules:
            keyword_score = sum(1 for token in module.keywords if token and token.lower() in lowered)
            alias_score = sum(1 for token in module.aliases if token and token.lower() in lowered)
            score = keyword_score + alias_score
            if score <= 0:
                continue
            if score > best_score or (score == best_score and module.route_priority < best_priority):
                best_score = score
                best_priority = module.route_priority
                best_module_name = module.name
                best_module_hint = module.hint

        return best_module_name, best_module_hint
