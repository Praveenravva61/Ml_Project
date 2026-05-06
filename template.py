import os
import sys
import logging
from pathlib import Path


project_name = "DataSciencePro"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/entity_config.py",
    f"src/{project_name}/constants/__init__.py"
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    'main.py',
    "DockerFile",
    "setup.py",
    "research/research.ipynb",
    "templates/index.html"
]



for file_path in list_of_files:
    file_path = Path(file_path)
    file_dir, file_name = os.path.split(file_path)

    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Creating directory: {file_dir}")

    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            pass
        logging.info(f"Creating file: {file_path}")
    else:
        logging.info(f"File already exists: {file_path}") 