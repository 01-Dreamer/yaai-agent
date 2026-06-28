from __future__ import annotations

import asyncio
import re
from typing import Any

from neo4j import GraphDatabase

from src.config import settings
from src.core.context import RuntimeContext
from src.tools.base import ToolResult, ToolSpec


WRITE_KEYWORDS = {
    "CREATE",
    "MERGE",
    "SET",
    "DELETE",
    "DETACH",
    "REMOVE",
    "DROP",
    "LOAD",
    "CALL DBMS",
    "CALL APOC",
}


FIELD_WORDS = (
    "人工智能",
    "大模型",
    "智能体",
    "知识图谱",
    "自然语言处理",
    "计算机视觉",
    "软件工程",
    "智慧教育",
    "智慧农业",
    "智慧文旅",
    "工业智能",
    "云计算",
    "边缘计算",
    "物联网",
    "嵌入式",
    "网络安全",
    "信息安全",
    "数据科学",
    "大数据",
)

ORG_WORDS = (
    "云南大学软件学院&人工智能学院",
    "云南大学软件学院",
    "人工智能学院",
    "云南人工智能协会",
    "滇池智能科技有限公司",
    "云南数字农业科技有限公司",
    "云南大学",
)


class GraphQueryTool:
    name = "graph.query_tool"
    description = (
        "只读查询 Neo4j 知识图谱。参数：查询关键词必填；查询意图可选；返回条数可选，默认 10 条，最多 50 条。"
        "支持意图：查看图谱结构、专家详情、按方向查专家、按方向查项目、按方向查政策、按方向查活动、组织关联、全局搜索。"
        "返回：查询模式、关键词、记录列表或结构信息；结果摘要会给出中文描述。"
        "示例：{\"查询关键词\":\"人工智能方向有哪些专家\",\"查询意图\":\"按方向查专家\",\"返回条数\":8}。"
        "限制：只能执行只读查询，禁止任何写入或删除操作。"
    )
    spec = ToolSpec(
        name=name,
        description=description,
        namespace="graph",

        capabilities=("query_graph", "expert_search", "relationship_search"),
    )

    def __init__(self) -> None:
        self._driver = None

    async def run(self, context: RuntimeContext, **kwargs: Any) -> ToolResult:
        try:
            data = await asyncio.to_thread(self._run_sync, kwargs)
            summary = self._summarize(data)
            return ToolResult(True, data=data, summary=summary)
        except Exception as exc:
            return ToolResult(False, error=str(exc))

    def _run_sync(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if not settings.neo4j_uri or not settings.neo4j_username:
            raise RuntimeError("Neo4j is not configured")

        cypher = str(kwargs.get("cypher") or "").strip()
        params = kwargs.get("params") if isinstance(kwargs.get("params"), dict) else {}
        limit = self._limit(kwargs.get("limit") or params.get("limit") or 10)
        if cypher:
            return self._run_cypher(cypher, {**params, "limit": limit}, limit)

        query = str(kwargs.get("query") or kwargs.get("keyword") or "").strip()
        intent = str(kwargs.get("intent") or "").strip()
        keyword = str(kwargs.get("keyword") or self._extract_keyword(query) or query).strip()
        if not query and not keyword and intent != "schema":
            raise ValueError("missing graph query")

        intent = intent or self._infer_intent(query)
        if intent == "schema":
            return self._schema()
        if intent == "expert_detail":
            return self._expert_detail(keyword, limit)
        if intent == "experts_by_field":
            return self._experts_by_field(keyword, limit)
        if intent == "projects_by_field":
            return self._projects_by_field(keyword, limit)
        if intent == "policies_by_field":
            return self._policies_by_field(keyword, limit)
        if intent == "activities_by_field":
            return self._activities_by_field(keyword, limit)
        if intent == "organization_related":
            return self._organization_related(keyword, limit)
        return self._global_search(keyword, limit)

    def _driver_instance(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password),
            )
        return self._driver

    def _read(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        driver = self._driver_instance()
        with driver.session(database=settings.neo4j_database) as session:
            result = session.run(cypher, params or {})
            return [self._sanitize(dict(record)) for record in result]

    def _run_cypher(self, cypher: str, params: dict[str, Any], limit: int) -> dict[str, Any]:
        self._validate_readonly_cypher(cypher)
        if " limit " not in f" {cypher.lower()} ":
            cypher = f"{cypher.rstrip(';')} LIMIT $limit"
        rows = self._read(cypher, {**params, "limit": limit})
        return {"mode": "cypher", "cypher": cypher, "params": self._sanitize(params), "records": rows}

    def _schema(self) -> dict[str, Any]:
        labels = self._read("CALL db.labels() YIELD label RETURN label ORDER BY label")
        rels = self._read("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType")
        counts = self._read("MATCH (n) RETURN labels(n) AS labels, count(*) AS count ORDER BY count DESC")
        return {"mode": "schema", "labels": labels, "relationshipTypes": rels, "counts": counts}

    def _experts_by_field(self, keyword: str, limit: int) -> dict[str, Any]:
        rows = self._read(
            """
            MATCH (e:Expert)-[r:研究方向]->(f:Field)
            WHERE f.name CONTAINS $keyword OR f.description CONTAINS $keyword
            OPTIONAL MATCH (e)-[:任职于]->(o:Organization)
            RETURN e.id AS id, e.name AS name, e.title AS title, e.mentor_type AS mentorType,
                   coalesce(o.name, e.organization) AS organization, e.source_page AS sourcePage,
                   f.name AS field, r.weight AS weight
            ORDER BY coalesce(r.weight, 0) DESC, name
            LIMIT $limit
            """,
            {"keyword": keyword, "limit": limit},
        )
        return {"mode": "experts_by_field", "keyword": keyword, "records": rows}

    def _expert_detail(self, keyword: str, limit: int) -> dict[str, Any]:
        rows = self._read(
            """
            MATCH (e:Expert)
            WHERE e.name CONTAINS $keyword
               OR e.profile CONTAINS $keyword
               OR e.organization CONTAINS $keyword
            OPTIONAL MATCH (e)-[r]-(n)
            RETURN e.id AS id, e.name AS name, e.title AS title, e.mentor_type AS mentorType,
                   e.organization AS organization, e.profile AS profile, e.source_page AS sourcePage,
                   collect({rel: type(r), direction: CASE WHEN startNode(r)=e THEN 'out' ELSE 'in' END,
                            labels: labels(n), props: properties(n), relProps: properties(r)})[..20] AS relations
            LIMIT $limit
            """,
            {"keyword": keyword, "limit": limit},
        )
        return {"mode": "expert_detail", "keyword": keyword, "records": rows}

    def _projects_by_field(self, keyword: str, limit: int) -> dict[str, Any]:
        rows = self._read(
            """
            MATCH (p:Project)-[:项目方向]->(f:Field)
            WHERE f.name CONTAINS $keyword OR f.description CONTAINS $keyword
            OPTIONAL MATCH (p)-[:牵头单位]->(o:Organization)
            OPTIONAL MATCH (p)-[:匹配政策]->(policy:Policy)
            OPTIONAL MATCH (p)-[:参与专家]->(expert:Expert)
            RETURN p.id AS id, p.name AS name, p.project_type AS projectType, p.status AS status,
                   p.year AS year, p.description AS description, collect(DISTINCT f.name) AS fields,
                   collect(DISTINCT o.name) AS organizations, collect(DISTINCT policy.name) AS policies,
                   collect(DISTINCT expert.name)[..10] AS experts
            ORDER BY name
            LIMIT $limit
            """,
            {"keyword": keyword, "limit": limit},
        )
        return {"mode": "projects_by_field", "keyword": keyword, "records": rows}

    def _policies_by_field(self, keyword: str, limit: int) -> dict[str, Any]:
        rows = self._read(
            """
            MATCH (p:Policy)-[:支持方向]->(f:Field)
            WHERE f.name CONTAINS $keyword OR f.description CONTAINS $keyword
            OPTIONAL MATCH (p)-[:适用于]->(o:Organization)
            OPTIONAL MATCH (project:Project)-[:匹配政策]->(p)
            RETURN p.id AS id, p.name AS name, p.level AS level, p.year AS year, p.status AS status,
                   p.description AS description, collect(DISTINCT f.name) AS fields,
                   collect(DISTINCT o.name) AS organizations, collect(DISTINCT project.name) AS projects
            ORDER BY name
            LIMIT $limit
            """,
            {"keyword": keyword, "limit": limit},
        )
        return {"mode": "policies_by_field", "keyword": keyword, "records": rows}

    def _activities_by_field(self, keyword: str, limit: int) -> dict[str, Any]:
        rows = self._read(
            """
            MATCH (a:Activity)-[:活动主题]->(f:Field)
            WHERE f.name CONTAINS $keyword OR f.description CONTAINS $keyword
            OPTIONAL MATCH (a)-[:主办方]->(o:Organization)
            OPTIONAL MATCH (expert:Expert)-[:演讲于]->(a)
            RETURN a.id AS id, a.name AS name, a.activity_type AS activityType, a.date AS date,
                   a.city AS city, a.description AS description, collect(DISTINCT f.name) AS fields,
                   collect(DISTINCT o.name) AS organizers, collect(DISTINCT expert.name) AS speakers
            ORDER BY date DESC
            LIMIT $limit
            """,
            {"keyword": keyword, "limit": limit},
        )
        return {"mode": "activities_by_field", "keyword": keyword, "records": rows}

    def _organization_related(self, keyword: str, limit: int) -> dict[str, Any]:
        rows = self._read(
            """
            MATCH (o:Organization)
            WHERE o.name CONTAINS $keyword OR o.short_name CONTAINS $keyword OR o.description CONTAINS $keyword
            OPTIONAL MATCH (expert:Expert)-[:任职于]->(o)
            OPTIONAL MATCH (project:Project)-[:牵头单位]->(o)
            OPTIONAL MATCH (activity:Activity)-[:主办方]->(o)
            OPTIONAL MATCH (policy:Policy)-[:适用于]->(o)
            RETURN o.id AS id, o.name AS name, o.short_name AS shortName, o.org_type AS orgType,
                   o.city AS city, o.description AS description,
                   collect(DISTINCT expert.name)[..20] AS experts,
                   collect(DISTINCT project.name) AS projects,
                   collect(DISTINCT activity.name) AS activities,
                   collect(DISTINCT policy.name) AS policies
            LIMIT $limit
            """,
            {"keyword": keyword, "limit": limit},
        )
        return {"mode": "organization_related", "keyword": keyword, "records": rows}

    def _global_search(self, keyword: str, limit: int) -> dict[str, Any]:
        rows = self._read(
            """
            MATCH (n)
            WHERE any(value IN [n.name, n.description, n.profile, n.organization, n.title, n.short_name]
                      WHERE value IS NOT NULL AND toString(value) CONTAINS $keyword)
            OPTIONAL MATCH (n)-[r]-(m)
            RETURN labels(n) AS labels, properties(n) AS props,
                   collect({rel: type(r), labels: labels(m), props: properties(m)})[..8] AS relations
            LIMIT $limit
            """,
            {"keyword": keyword, "limit": limit},
        )
        return {"mode": "global_search", "keyword": keyword, "records": rows}

    def _infer_intent(self, query: str) -> str:
        has_field = any(word in query for word in FIELD_WORDS)
        has_org = any(word in query for word in ORG_WORDS) or any(
            word in query for word in ["组织", "单位", "学院", "企业", "协会"]
        )
        if any(word in query for word in ["图谱结构", "schema", "标签", "关系类型"]):
            return "schema"
        if has_org and not has_field:
            return "organization_related"
        if any(word in query for word in ["专家详情", "这个专家", "老师详情", "是谁", "介绍"]):
            if has_field:
                return "experts_by_field"
            return "expert_detail"
        if any(word in query for word in ["专家", "老师", "教授", "导师"]):
            return "experts_by_field" if has_field else "global_search"
        if "项目" in query:
            return "projects_by_field"
        if "政策" in query:
            return "policies_by_field"
        if any(word in query for word in ["活动", "会议", "论坛", "研讨会"]):
            return "activities_by_field"
        if any(word in query for word in ["组织", "单位", "学院", "企业", "协会", "云南大学"]):
            return "organization_related"
        return "global_search"

    def _extract_keyword(self, query: str) -> str:
        for word in FIELD_WORDS:
            if word in query:
                return word
        for word in ORG_WORDS:
            if word in query:
                return word
        quoted = re.findall(r"[“\"']([^“\"']+)[”\"']", query)
        if quoted:
            return quoted[0].strip()
        cleaned = re.sub(
            r"(哪些|有关|相关|推荐|查询|搜索|查找|一下|的|有|请|帮我|给我|列出|是谁|介绍|详情|老师|教授|导师|专家)",
            " ",
            query,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned[:40]

    def _validate_readonly_cypher(self, cypher: str) -> None:
        normalized = re.sub(r"\s+", " ", cypher.strip()).upper()
        if not (normalized.startswith("MATCH ") or normalized.startswith("OPTIONAL MATCH ") or normalized.startswith("CALL DB.")):
            raise ValueError("only read-only MATCH / CALL db.* cypher is allowed")
        for keyword in WRITE_KEYWORDS:
            if keyword in normalized:
                raise ValueError(f"write cypher is not allowed: {keyword}")

    def _limit(self, value: Any) -> int:
        try:
            return max(1, min(int(value), 50))
        except Exception:
            return 10

    def _sanitize(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._sanitize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._sanitize(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _summarize(self, data: dict[str, Any]) -> str:
        records = data.get("records") or []
        if not records and data.get("mode") == "schema":
            return f"图谱包含 labels={data.get('labels')}，relationships={data.get('relationshipTypes')}。"
        lines = [f"知识图谱查询结果：mode={data.get('mode')}，keyword={data.get('keyword', '')}"]
        for index, record in enumerate(records[:8], start=1):
            lines.append(f"{index}. {record}")
        return "\n".join(lines)
