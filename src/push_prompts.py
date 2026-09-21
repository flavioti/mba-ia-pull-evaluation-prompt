"""
Script para fazer push de prompts otimizados ao LangSmith Prompt Hub.

Este script:
1. Lê os prompts otimizados de prompts/bug_to_user_story_v2.yml
2. Valida os prompts
3. Faz push PÚBLICO para o LangSmith Hub
4. Adiciona metadados (tags, descrição, técnicas utilizadas)

SIMPLIFICADO: Código mais limpo e direto ao ponto.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate
from utils import load_yaml, check_env_vars, print_section_header, validate_prompt_structure

load_dotenv()

PROMPT_KEY = "bug_to_user_story_v2"
PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / f"{PROMPT_KEY}.yml"


def push_prompt_to_langsmith(prompt_name: str, prompt_data: dict) -> bool:
    """
    Faz push do prompt otimizado para o LangSmith Hub (PÚBLICO).

    Args:
        prompt_name: Nome do prompt (ex: "usuario/bug_to_user_story_v2")
        prompt_data: Dados do prompt (system_prompt, user_prompt, description,
                     techniques_applied, tags, ...)

    Returns:
        True se sucesso, False caso contrário
    """
    system_prompt = prompt_data["system_prompt"]
    user_prompt = prompt_data.get("user_prompt") or "{bug_report}"

    techniques = list(prompt_data.get("techniques_applied", []))
    tags = list(dict.fromkeys(list(prompt_data.get("tags", [])) + techniques))
    description = prompt_data.get("description", "")

    readme = (
        f"# {prompt_name}\n\n"
        f"{description}\n\n"
        f"**Versão:** {prompt_data.get('version', 'v2')}\n\n"
        "**Técnicas aplicadas:**\n"
        + "\n".join(f"- {t}" for t in techniques)
    )

    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_prompt),
        ])

        url = Client().push_prompt(
            prompt_name,
            object=prompt,
            is_public=True,
            description=description,
            readme=readme,
            tags=tags,
        )

        print(f"✅ Prompt publicado: {prompt_name}")
        print(f"   {url}")
        return True

    except Exception as e:
        print(f"❌ Erro ao fazer push de '{prompt_name}': {e}")
        return False


def validate_prompt(prompt_data: dict) -> tuple[bool, list]:
    """
    Valida estrutura básica de um prompt (versão simplificada).

    Args:
        prompt_data: Dados do prompt

    Returns:
        (is_valid, errors) - Tupla com status e lista de erros
    """
    if not isinstance(prompt_data, dict):
        return (False, ["O conteúdo do prompt não é um dicionário válido"])

    # Regras gerais: campos obrigatórios, system_prompt sem TODO, >= 2 técnicas
    _, errors = validate_prompt_structure(prompt_data)

    if not str(prompt_data.get("user_prompt", "")).strip():
        errors.append("user_prompt está vazio")

    return (len(errors) == 0, errors)


def main():
    """Função principal"""
    print_section_header("PUSH DE PROMPTS PARA O LANGSMITH")

    if not check_env_vars(["LANGSMITH_API_KEY", "USERNAME_LANGSMITH_HUB"]):
        return 1

    raw = load_yaml(str(PROMPT_FILE))
    if raw is None:
        print(f"   Verifique se o arquivo existe: {PROMPT_FILE}")
        return 1

    # O YAML tem uma chave raiz com o nome do prompt (como no v1); aceita também formato plano
    prompt_data = raw.get(PROMPT_KEY, raw) if isinstance(raw, dict) else raw

    is_valid, errors = validate_prompt(prompt_data)
    if not is_valid:
        print("❌ Prompt inválido:")
        for error in errors:
            print(f"   - {error}")
        return 1

    username = os.getenv("USERNAME_LANGSMITH_HUB")
    prompt_name = f"{username}/{PROMPT_KEY}"

    if not push_prompt_to_langsmith(prompt_name, prompt_data):
        return 1

    print("\n➡️  Próximo passo: python src/evaluate.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
