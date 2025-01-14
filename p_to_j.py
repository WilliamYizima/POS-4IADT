# Gerando o arquivo Jupyter Notebook baseado no script fornecido

from nbformat import v4 as nbf

# Criando um novo notebook
notebook = nbf.new_notebook()

# Dividindo o script em células baseadas em seções e comentários
cells = []

with open('./testes.py', "r", encoding="utf-8") as file:
    script_content = file.read()


# Adicionando cada célula com comentários explicativos
for section in script_content.split("\n\n"):
    # Adicionar comentário de explicação se a célula começa com um comentário
    if section.startswith("#"):
        cells.append(nbf.new_markdown_cell(section))
    else:
        cells.append(nbf.new_code_cell(section))

# Adicionando as células ao notebook
notebook.cells.extend(cells)

# Salvando o notebook como um arquivo .ipynb
notebook_path = "./testes_notebook.ipynb"
with open(notebook_path, "w", encoding="utf-8") as f:
    f.write(nbf.writes(notebook))

notebook_path

