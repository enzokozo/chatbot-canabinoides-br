"""
src/core/config.py
──────────────────
Configurações centrais do sistema RAG de canabinóides.

Carregamento de segredos:
  1. Lê o arquivo `.env` na raiz do repositório (via `env_file`).
  2. Variáveis de ambiente do SO têm precedência sobre o `.env`
     (comportamento padrão do pydantic-settings — seguro para produção/CI).

NUNCA insira valores de credenciais diretamente neste arquivo.
Use sempre `.env` (desenvolvimento) ou um cofre de segredos (produção).

Uso:
    from core.config import get_settings
    settings = get_settings()
    print(settings.QDRANT_URL)
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Enums auxiliares
# ---------------------------------------------------------------------------

class AppEnv(str, Enum):
    DEVELOPMENT = "development"
    STAGING     = "staging"
    PRODUCTION  = "production"


class LogLevel(str, Enum):
    DEBUG   = "DEBUG"
    INFO    = "INFO"
    WARNING = "WARNING"
    ERROR   = "ERROR"


# ---------------------------------------------------------------------------
# Padrões de ruído identificados nos PDFs reais
# Cada entrada é uma substring ou regex aplicada na limpeza pós-extração.
# Centralizados aqui para que o chunker importe diretamente de config.
# ---------------------------------------------------------------------------

PDF_NOISE_PATTERNS: dict[str, list[str]] = {
    # Rodapé repetitivo do documento CFM (endereço físico da entidade)
    "CFM_RESOLUCAO_2324_2022": [
        r"SGAS\s+915\s+Lote\s+72[^\n]*",
        r"Brasília[- ]DF[^\n]*CEP[^\n]*",
        r"www\.cfm\.org\.br[^\n]*",
        r"Página\s+\d+\s+de\s+\d+",            # "Página 3 de 18"
    ],
    # Carimbos e links da RDC 660 (Imprensa Nacional / DOU)
    "RDC_660_2022": [
        r"Imprensa\s+Nacional[^\n]*",
        r"Este\s+documento\s+pode\s+ser\s+verificado[^\n]*",
        r"Acessivel\.com[^\n]*",
        r"https?://\S+",                         # qualquer URL
        r"Autenticidade\s+pode\s+ser\s+confirmada[^\n]*",
        r"Documento\s+assinado\s+digitalmente[^\n]*",
    ],
    # Cabeçalho fixo de advertência da RDC 327
    "RDC_327_2019": [
        r"ADVERTÊNCIA[:\s]*Este\s+documento[^\n]*",
        r"Nota:\s+Esta\s+versão[^\n]*",
        r"NÃO\s+OFICIAL[^\n]*",
        r"D\.O\.U\.[^\n]*Seção\s+1[^\n]*",      # referências ao DOU no cabeçalho
    ],
    # Padrões globais presentes em todos os documentos
    "_ALL_": [
        r"^\s*\d+\s*$",                          # linhas que só têm número (nº de página)
        r"^\s*[-–—]{3,}\s*$",                    # linhas só com traços (separadores)
        r"Powered\s+by[^\n]*",
    ],
}

# Fontes regulatórias válidas (allowlist — nunca aceite strings arbitrárias)
REGULATORY_SOURCES: list[str] = list(PDF_NOISE_PATTERNS.keys() - {"_ALL_"})


# ---------------------------------------------------------------------------
# Settings principal
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Todas as configurações do sistema.

    Ordem de precedência (maior → menor):
      1. Variáveis de ambiente do SO  (ex.: export QDRANT_URL=...)
      2. Arquivo .env na raiz do repositório
      3. Valores default definidos abaixo
    """

    # ── Identidade ──────────────────────────────────────────────────────────
    APP_NAME: str    = "cannabinoid-rag"
    APP_ENV: AppEnv  = AppEnv.DEVELOPMENT
    LOG_LEVEL: LogLevel = LogLevel.INFO

    # ── Caminhos (relativos à raiz do repositório) ──────────────────────────
    # O .env pode sobrescrever estes caminhos para ambientes diferentes.
    RAW_DATA_DIR: Path       = Path("data/raw")
    PROCESSED_DATA_DIR: Path = Path("data/processed")
    LOG_DIR: Path            = Path("logs")

    # ── LLM — Anthropic ─────────────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = Field(
        ...,
        description="Chave da API Anthropic. Obtenha em console.anthropic.com",
    )
    LLM_MODEL: str       = "claude-sonnet-4-20250514"
    LLM_MAX_TOKENS: int  = Field(default=2048, ge=256, le=8192)
    LLM_TEMPERATURE: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    # Temperatura 0.0 = respostas determinísticas e factuais, sem "criatividade"

    # ── Embeddings — OpenAI ─────────────────────────────────────────────────
    OPENAI_API_KEY: str = Field(
        ...,
        description="Chave da API OpenAI para geração de embeddings.",
    )
    EMBEDDING_MODEL: str      = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = Field(default=1536, ge=256, le=3072)

    # ── Banco vetorial — Qdrant ─────────────────────────────────────────────
    QDRANT_URL: str = Field(
        default="http://localhost:6333",
        description="URL do servidor Qdrant. Para Qdrant Cloud use a URL fornecida no console.",
    )
    QDRANT_API_KEY: str | None = Field(
        default=None,
        description="API key do Qdrant Cloud. Deixe None para instâncias locais sem autenticação.",
    )
    QDRANT_COLLECTION_NAME: str = "anvisa_cfm_regulations"
    QDRANT_TIMEOUT_SECONDS: int = Field(default=30, ge=5, le=120)

    # ── Chunking ────────────────────────────────────────────────────────────
    CHUNK_SIZE_TOKENS: int    = Field(default=512,  ge=128, le=2048)
    CHUNK_OVERLAP_TOKENS: int = Field(default=64,   ge=0,   le=256)
    MIN_CHUNK_CHARS: int      = Field(default=80,   ge=20)
    # Número mínimo de caracteres que uma página precisa ter para ser
    # considerada "texto digital" (abaixo disso, tenta OCR).
    MIN_CHARS_FOR_DIGITAL_PAGE: int = Field(default=50, ge=10)

    # ── API ─────────────────────────────────────────────────────────────────
    API_SECRET_KEY: str = Field(
        ...,
        description=(
            "Segredo para assinar tokens JWT. "
            "Gere com: python -c \"import secrets; print(secrets.token_hex(32))\""
        ),
    )
    API_ALLOWED_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8000"]
    RATE_LIMIT_PER_MINUTE: int     = Field(default=30, ge=1, le=300)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60, ge=5)

    # ── Segurança de ingestão ────────────────────────────────────────────────
    MAX_PDF_SIZE_MB: int  = Field(default=50, ge=1, le=200)
    MAX_PDF_PAGES: int    = Field(default=500, ge=1)

    # ── Pydantic-settings: lê o .env na raiz ────────────────────────────────
    model_config = SettingsConfigDict(
        # Busca o .env na raiz do repositório (dois níveis acima de src/core/).
        # Em produção, as variáveis de ambiente do SO têm precedência automática.
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        # "forbid" rejeita qualquer variável desconhecida no .env,
        # impedindo configurações silenciosamente ignoradas.
        extra="forbid",
        # Permite que campos com tipo Path recebam strings do .env
        arbitrary_types_allowed=True,
    )

    # ── Validadores ──────────────────────────────────────────────────────────

    @field_validator("RAW_DATA_DIR", "PROCESSED_DATA_DIR", "LOG_DIR", mode="before")
    @classmethod
    def _ensure_directory_exists(cls, v: str | Path) -> Path:
        """Cria o diretório automaticamente se não existir."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @model_validator(mode="after")
    def _validate_chunk_overlap(self) -> "Settings":
        """O overlap não pode ser maior que metade do chunk size."""
        if self.CHUNK_OVERLAP_TOKENS >= self.CHUNK_SIZE_TOKENS // 2:
            raise ValueError(
                f"CHUNK_OVERLAP_TOKENS ({self.CHUNK_OVERLAP_TOKENS}) deve ser menor que "
                f"metade de CHUNK_SIZE_TOKENS ({self.CHUNK_SIZE_TOKENS // 2}). "
                "Overlap muito grande desperdiça tokens e gera duplicatas no banco vetorial."
            )
        return self

    @model_validator(mode="after")
    def _warn_production_debug(self) -> "Settings":
        """Garante que DEBUG nunca vaze em produção."""
        if self.APP_ENV == AppEnv.PRODUCTION and self.LOG_LEVEL == LogLevel.DEBUG:
            # Não usamos logger aqui (ainda não está configurado).
            # Levantamos erro para forçar correção no deploy.
            raise ValueError(
                "LOG_LEVEL=DEBUG não é permitido em APP_ENV=production. "
                "Use INFO ou WARNING em produção para evitar vazamento de dados."
            )
        return self

    # ── Propriedades derivadas ───────────────────────────────────────────────

    @property
    def max_pdf_size_bytes(self) -> int:
        """Converte MAX_PDF_SIZE_MB para bytes (usado nas validações de arquivo)."""
        return self.MAX_PDF_SIZE_MB * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == AppEnv.PRODUCTION

    @property
    def is_development(self) -> bool:
        return self.APP_ENV == AppEnv.DEVELOPMENT


# ---------------------------------------------------------------------------
# Singleton com cache
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Retorna a instância singleton das configurações.

    @lru_cache garante que o arquivo .env é lido apenas uma vez durante
    o ciclo de vida da aplicação. Para recarregar (ex.: em testes),
    chame `get_settings.cache_clear()` antes.

    Exemplo:
        from core.config import get_settings
        settings = get_settings()
    """
    return Settings()
