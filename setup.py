from setuptools import setup, find_packages
from pathlib import Path

this_dir = Path(__file__).parent
readme_path = this_dir / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="agent-graph-rag",
    version="0.1.0",
    description="Agent Graph RAG project",
    long_description=long_description,
    long_description_content_type="text/markdown",

    # 关键：整个项目里找包（包括 src、models 等）
    packages=find_packages(),

    python_requires=">=3.8",
    install_requires=[
        "oss2",
    ],
)