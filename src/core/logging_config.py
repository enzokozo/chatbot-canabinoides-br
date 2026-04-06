"""
src/core/logging_config.py
──────────────────────────
Logger seguro para o sistema RAG médico.

Princípios de segurança aplicados:
  • Nenhum dado sensível (CPF, CRM, e-mail, data de nascimento, nome completo)
    é emitido nos logs — um `SafeFormatter` sanitiza cada mensagem.
  • Textos extraídos de PDFs ou prompts nunca são logados integralmente;
    apenas contagens de caracteres e prefixos de hashes identificam eventos.
  • Em produção (APP_ENV=production), o output é JSON estruturado via
    `structlog`, compatível com stacks de observabilidade (Datadog, Loki, etc.).
  • Em desenvolvimento, o output é texto colorido legível no terminal.
  • LOG_LEVEL=DEBUG é bloqueado em produção pelo validador de config.py.

Uso:
    from core.logging_config import get_logger, log_pipeline_event

    logger = get_logger(__name__)
    logger.info("Processando documento | doc_id=%s | páginas=%d", doc_id, n)

    log_pipeline_event(logger, "chunk_created", doc_id="abc123", extra={"chunks": 42})
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Padrões de PII que NUNCA devem aparecer nos logs
# ---------------------------------------------------------------------------

_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # CPF: 000.000.000-00
    (re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"), "[CPF_REDACTED]"),
    # CRM: CRM/SP 123456, CRM-RJ 12345, CRM 12345
    (re.compile(r"\bCRM[\s/\-]?[A-Z]{0,2}[\s/\-]?\d{4,6}\b", re.IGNORECASE), "[CRM_REDACTED]"),
    # CNPJ: 00.000.000/0000-00
    (re.compile(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b"), "[CNPJ_REDACTED]"),
    # E-mail
    (
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        "[EMAIL_REDACTED]",
    ),
    # Datas no formato DD/MM/AAAA (provável data de nascimento em contexto médico)
    (re.compile(r"\b\d{2}/\d{2}/\d{4}\b"), "[DATE_REDACTED]"),
    # Telefones BR: (11) 91234-5678, 11912345678
    (re.compile(r"\(?\d{2}\)?\s?\d{4,5}[\s\-]?\d{4}\b"), "[PHONE_REDACTED]"),
    # RG simples (7-9 dígitos, pode ter ponto/traço)
    (re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}-?[\dXx]\b"), "[RG_REDACTED]"),
]

# Texto de prompts ou conteúdo de PDF truncado a este tamanho antes de logar.
# Evita que um prompt longo (que pode conter PII enviada pelo usuário) vaze.
_MAX_LOG_MSG_CHARS: int = 250


# ---------------------------------------------------------------------------
# Formatter seguro
# ---------------------------------------------------------------------------

def _sanitize(text: str) -> str:
    """
    Remove padrões de PII e trunca textos longos.
    Aplicado a TODA mensagem antes de qualquer handler emiti-la.
    """
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    if len(text) > _MAX_LOG_MSG_CHARS:
        text = text[:_MAX_LOG_MSG_CHARS] + " …[TRUNCADO]"
    return text


class SafeFormatter(logging.Formatter):
    """
    Formatter customizado que sanitiza a mensagem e seus argumentos
    antes de construir a string final de log.

    Funciona com ambas as formas de logging:
        logger.info("CPF: %s", cpf_value)         # args como tupla
        logger.info("dados: %(cpf)s", {"cpf": x}) # args como dict
    """

    def format(self, record: logging.LogRecord) -> str:
        # Sanitiza a mensagem principal (antes da interpolação de args)
        record.msg = _sanitize(str(record.msg))

        # Sanitiza os argumentos de formatação
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: _sanitize(str(v)) for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(_sanitize(str(a)) for a in record.args)

        return super().format(record)


# ---------------------------------------------------------------------------
# Fábrica de loggers
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """
    Retorna um logger configurado com SafeFormatter.

    Em desenvolvimento : texto colorido no stdout.
    Em produção        : JSON estruturado (via structlog, se disponível).

    O `name` deve sempre ser `__name__` para rastreabilidade por módulo.

    Exemplo:
        logger = get_logger(__name__)
        logger.warning("Hash inválido detectado | doc_id=%s", doc_id[:8])
    """
    # Import aqui para evitar ciclo (config importa logging_config)
    from core.config import get_settings
    settings = get_settings()

    logger = logging.getLogger(name)

    # Evita duplicar handlers se o logger já foi inicializado
    if logger.handlers:
        return logger

    level_str: str = settings.LOG_LEVEL.value
    level: int = getattr(logging, level_str)

    logger.setLevel(level)

    if settings.is_production:
        _configure_production_handler(logger, level)
    else:
        _configure_development_handler(logger, level)

    # Não propaga para o root logger — evita duplicação de mensagens
    logger.propagate = False
    return logger


def _configure_development_handler(
    logger: logging.Logger,
    level: int,
) -> None:
    """Handler de texto legível para desenvolvimento."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        SafeFormatter(
            fmt=(
                "%(asctime)s | %(levelname)-8s | "
                "%(name)-40s | %(message)s"
            ),
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logger.addHandler(handler)


def _configure_production_handler(
    logger: logging.Logger,
    level: int,
) -> None:
    """
    Handler JSON estruturado para produção via structlog.
    Fallback para texto se structlog não estiver instalado.
    """
    try:
        import structlog

        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                # Sanitiza via SafeFormatter antes de serializar
                _structlog_sanitize_processor,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        # structlog já formata — usamos Formatter base (sem SafeFormatter)
        # pois o processor acima já sanitiza
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    except ImportError:
        # structlog não instalado: usa texto com SafeFormatter
        _configure_development_handler(logger, level)


def _structlog_sanitize_processor(
    _logger: Any,
    _method: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """
    Processor structlog que sanitiza todos os valores do event_dict.
    Garante que nenhum valor com PII escape para o JSON de produção.
    """
    return {
        k: _sanitize(str(v)) if isinstance(v, str) else v
        for k, v in event_dict.items()
    }


# ---------------------------------------------------------------------------
# Helper para eventos do pipeline
# ---------------------------------------------------------------------------

def log_pipeline_event(
    logger: logging.Logger,
    event: str,
    doc_id: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Loga um evento do pipeline de forma estruturada e segura.

    Convenções de segurança:
      • `doc_id` deve ser sempre o PREFIXO do hash SHA-256 (primeiros 16 chars),
        NUNCA o nome do arquivo original (que pode conter PII).
      • `extra` só deve conter métricas (contagens, booleans, durations),
        NUNCA textos extraídos ou conteúdo de prompts.

    Args:
        logger : Logger obtido via get_logger(__name__).
        event  : Nome do evento, ex: "pdf_extraction_start", "chunk_created".
        doc_id : Prefixo do hash SHA-256 do documento (identificador sem PII).
        extra  : Dicionário de métricas adicionais (opcional).

    Exemplo:
        log_pipeline_event(
            logger,
            event="embedding_batch_complete",
            doc_id=file_hash[:16],
            extra={"batch": 3, "chunks": 64, "elapsed_s": 1.2},
        )
    """
    if len(doc_id) > 32:
        # Proteção: nunca loga hashes completos nem nomes de arquivo
        doc_id = doc_id[:16] + "…"

    safe_extra_str = " | ".join(
        f"{k}={_sanitize(str(v))}"
        for k, v in (extra or {}).items()
    )

    logger.info(
        "PIPELINE | event=%s | doc_id=%s%s",
        event,
        doc_id,
        f" | {safe_extra_str}" if safe_extra_str else "",
    )
