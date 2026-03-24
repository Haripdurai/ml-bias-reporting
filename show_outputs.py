import json
with open('churn_explainability_executed.ipynb') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and cell.get('outputs'):
        for out in cell['outputs']:
            if out.get('output_type') in ('stream', 'execute_result'):
                text = ''.join(out.get('text', out.get('data', {}).get('text/plain', [])))
                if text.strip():
                    print(f'--- Cell {i+1} ---')
                    print(text[:800])
                    print()
