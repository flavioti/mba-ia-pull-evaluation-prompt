"""
Script para fazer pull de prompts do LangSmith Prompt Hub.

Este script:
1. Conecta ao LangSmith usando credenciais do .env
2. Faz pull dos prompts do Hub
3. Salva localmente em prompts/bug_to_user_story_v1.yml

SIMPLIFICADO: Usa serialização nativa do LangChain para extrair prompts.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client
from langchain_core.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from utils import save_yaml, check_env_vars, print_section_header


load_dotenv()

PROMPT_NAME = "leonanluppi/bug_to_user_story_v1"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "bug_to_user_story_v1.yml"


def pull_prompts_from_langsmith() -> bool:
    """
    Faz pull do prompt inicial (v1) do LangSmith Prompt Hub e salva em YAML.

    O prompt é baixado como ChatPromptTemplate; as mensagens de system e de
    human são extraídas e gravadas em prompts/bug_to_user_story_v1.yml, no
    mesmo formato do arquivo original (chave raiz com o nome do prompt).

    Returns:
        True se o pull e o salvamento foram bem-sucedidos, False caso contrário
    """
    if not check_env_vars(["LANGSMITH_API_KEY"]):
        return False

    print(f"⬇️  Fazendo pull do prompt: {PROMPT_NAME}")

    try:
        # langchain.hub foi removido no langchain 1.x; o Client do langsmith
        # funciona em qualquer versão e usa LANGSMITH_API_KEY do ambiente.
        prompt = Client().pull_prompt(
            PROMPT_NAME, 
            # Permite pull de prompts públicos sem autenticação
            dangerously_pull_public_prompt=True
        )
    except Exception as e:
        print(f"❌ Erro ao fazer pull de '{PROMPT_NAME}': {e}")
        return False

    system_prompt = ""
    user_prompt = ""

    for message in getattr(prompt, "messages", []):
        if isinstance(message, SystemMessagePromptTemplate):
            system_prompt = message.prompt.template
        elif isinstance(message, HumanMessagePromptTemplate):
            user_prompt = message.prompt.template

    # Fallback: prompt simples (PromptTemplate) sem mensagens separadas
    if not system_prompt and not user_prompt and hasattr(prompt, "template"):
        system_prompt = prompt.template

    if not system_prompt and not user_prompt:
        print("❌ O prompt retornado não contém mensagens de system/user.")
        return False

    key = PROMPT_NAME.split("/")[-1]
    data = {
        key: {
            "description": "Prompt para converter relatos de bugs em User Stories",
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "version": "v1",
            "source": PROMPT_NAME,
            "tags": ["bug-analysis", "user-story", "product-management"],
        }
    }

    if not save_yaml(data, str(OUTPUT_PATH)):
        return False

    print(f"✅ Prompt salvo em: {OUTPUT_PATH}")
    return True


def main():
    """Função principal"""
    ...
    pull_prompts_from_langsmith()


if __name__ == "__main__":
    sys.exit(main())
