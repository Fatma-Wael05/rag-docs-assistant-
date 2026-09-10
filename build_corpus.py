"""
build_corpus.py
Builds the RAG document corpus by cloning the actual doc repos locally
(fast, no bot-blocking) and cleaning MDX/MD noise (imports, JSX component
placeholders, frontmatter) down to clean prose+headers markdown.

Setup:
    No extra pip installs needed -- just Python's standard library + git.

Run (from inside your project folder, venv activated or not -- git is
all this needs):
    python build_corpus.py

This will:
  1. Shallow-clone the doc repos into ./_doc_repos/ (deleted-safe, not part
     of your submission -- add _doc_repos/ to .gitignore)
  2. Copy + clean a curated set of pages into ./data/raw/
"""

import os
import re
import subprocess
import shutil

REPOS_DIR = "_doc_repos"
OUT_DIR = os.path.join("data", "raw")

# (repo_url, local_name, sparse_checkout_path)
REPOS = [
    ("https://github.com/langchain-ai/docs.git", "lc_docs", "src"),
    ("https://github.com/crewAIInc/crewAI.git", "crewai", "docs/v1.15.4/en"),
    ("https://github.com/run-llama/llama_index.git", "llama_index", "docs"),
]

# (repo_local_name, relative_path_in_repo, output_filename)
PAGES = [
    # --- LangChain (core) ---
    ("lc_docs", "src/oss/langchain/agents.mdx", "langchain_agents.md"),
    ("lc_docs", "src/oss/langchain/tools.mdx", "langchain_tools.md"),
    ("lc_docs", "src/oss/langchain/models.mdx", "langchain_models.md"),
    ("lc_docs", "src/oss/langchain/messages.mdx", "langchain_messages.md"),
    ("lc_docs", "src/oss/langchain/retrieval.mdx", "langchain_retrieval.md"),
    ("lc_docs", "src/oss/langchain/short-term-memory.mdx", "langchain_short_term_memory.md"),
    ("lc_docs", "src/oss/langchain/long-term-memory.mdx", "langchain_long_term_memory.md"),
    ("lc_docs", "src/oss/langchain/knowledge-base.mdx", "langchain_knowledge_base.md"),
    ("lc_docs", "src/oss/langchain/multi-agent/index.mdx", "langchain_multi_agent.md"),
    # --- LangGraph ---
    ("lc_docs", "src/oss/langgraph/overview.mdx", "langgraph_overview.md"),
    ("lc_docs", "src/oss/langgraph/graph-api.mdx", "langgraph_graph_api.md"),
    ("lc_docs", "src/oss/langgraph/persistence.mdx", "langgraph_persistence.md"),
    ("lc_docs", "src/oss/langgraph/workflows-agents.mdx", "langgraph_workflows_agents.md"),
    ("lc_docs", "src/oss/langgraph/agentic-rag.mdx", "langgraph_agentic_rag.md"),
    # --- CrewAI ---
    ("crewai", "docs/v1.15.4/en/concepts/agents.mdx", "crewai_agents.md"),
    ("crewai", "docs/v1.15.4/en/concepts/tasks.mdx", "crewai_tasks.md"),
    ("crewai", "docs/v1.15.4/en/concepts/crews.mdx", "crewai_crews.md"),
    ("crewai", "docs/v1.15.4/en/concepts/tools.mdx", "crewai_tools.md"),
    ("crewai", "docs/v1.15.4/en/concepts/memory.mdx", "crewai_memory.md"),
    ("crewai", "docs/v1.15.4/en/concepts/flows.mdx", "crewai_flows.md"),
    # --- LlamaIndex ---
    ("llama_index", "docs/src/content/docs/framework/module_guides/indexing/index.md", "llamaindex_indexing.md"),
    ("llama_index", "docs/src/content/docs/framework/module_guides/querying/index.md", "llamaindex_querying.md"),
    ("llama_index", "docs/src/content/docs/framework/optimizing/production_rag.md", "llamaindex_production_rag.md"),
    ("llama_index", "docs/src/content/docs/framework/optimizing/building_rag_from_scratch.md", "llamaindex_building_rag.md"),
]


def clone_repos():
    os.makedirs(REPOS_DIR, exist_ok=True)
    for url, name, sparse_path in REPOS:
        target = os.path.join(REPOS_DIR, name)
        if os.path.isdir(target):
            print(f"[skip] {name} already cloned")
            continue
        print(f"Cloning {name} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", url, target],
            check=True,
        )
        subprocess.run(["git", "sparse-checkout", "set", sparse_path], cwd=target, check=True)


def clean_mdx(text: str) -> str:
    # Strip YAML frontmatter (--- ... ---)
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)
    # Remove import lines
    text = re.sub(r"^import .*$", "", text, flags=re.MULTILINE)
    # Remove :::python / :::js block markers
    text = re.sub(r":::(python|js)\s*", "", text)
    text = re.sub(r":::", "", text)
    # Remove self-closing JSX component placeholders, e.g. <AgentsIntroPy />
    text = re.sub(r"<[A-Z][A-Za-z0-9]*\s*/>", "", text)
    # Remove <img ... /> blocks
    text = re.sub(r"<img[^>]*/>", "", text, flags=re.DOTALL)
    # Unwrap simple tags like <Note>...</Note>, <Tip>...</Tip> but keep inner text
    text = re.sub(r"</?(Note|Tip|Warning|Info|Card|CardGroup|Columns|Column|Accordion|AccordionGroup)[^>]*>", "", text)
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main():
    clone_repos()
    os.makedirs(OUT_DIR, exist_ok=True)

    ok, missing = 0, []
    for repo_name, rel_path, out_filename in PAGES:
        src_path = os.path.join(REPOS_DIR, repo_name, rel_path)
        if not os.path.isfile(src_path):
            print(f"[MISSING] {src_path}")
            missing.append(src_path)
            continue

        with open(src_path, "r", encoding="utf-8") as f:
            raw = f.read()

        cleaned = clean_mdx(raw)
        if len(cleaned) < 200:
            print(f"[TOO SHORT after cleaning] {src_path} ({len(cleaned)} chars) -- skipping")
            missing.append(src_path)
            continue

        out_path = os.path.join(OUT_DIR, out_filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned)
        print(f"[OK] {out_filename} ({len(cleaned)} chars)")
        ok += 1

    print(f"\nDone. {ok} saved, {len(missing)} missing/skipped.")
    if missing:
        print("Check these paths manually (repo structure may have shifted):")
        for m in missing:
            print(f"  - {m}")


if __name__ == "__main__":
    main()
