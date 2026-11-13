import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import importlib
import nbformat as nbf

MODULES = [
    'scripts.sec_01_05',
    'scripts.sec_06_11',
    'scripts.sec_12_17',
    'scripts.sec_18_26',
    'scripts.sec_27_33',
    'scripts.sec_34_41',
    'scripts.sec_42_47',
    'scripts.sec_48_54',
]

def build_notebook(output_path: Path):
    nb = nbf.v4.new_notebook()
    total_sections = 0
    
    for mod_name in MODULES:
        mod = importlib.import_module(mod_name)
        importlib.reload(mod)
        for sec in mod.SECTIONS:
            total_sections += 1
            md_cell = nbf.v4.new_markdown_cell(sec["markdown"])
            code_cell = nbf.v4.new_code_cell(sec["code"])
            code_cell.execution_count = None
            code_cell.outputs = []
            nb.cells.append(md_cell)
            nb.cells.append(code_cell)
            
    _, nb = nbf.validator.normalize(nb)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, output_path)
    print(f"Successfully generated {output_path}:")
    print(f"  Total Sections : {total_sections}")
    print(f"  Total Cells    : {len(nb.cells)} ({len(nb.cells)//2} Markdown, {len(nb.cells)//2} Code)")
    return nb

if __name__ == '__main__':
    nb_path = project_root / 'notebooks' / '08_fewshot_pseudolabeling_efficientnet_b2.ipynb'
    build_notebook(nb_path)
