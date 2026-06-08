# Imagem oficial do Python
FROM python:3.12-slim

# Instala o gerenciador uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Define o diretório padrão do container
WORKDIR /app

# Copia os arquivos de especificação e de dependências
COPY pyproject.toml uv.lock ./

# Sincroniza e instala as dependências
RUN uv sync --frozen --no-cache

# Copia as pastas de código-fonte e modelos para dentro da estrutura da imagem
COPY src/ ./src/
COPY models/ ./models/

# Declara a porta que será exposta
EXPOSE 8000

# Executar o servidor Uvicorn com o ambiente virtual do uv
CMD ["uv", "run", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
