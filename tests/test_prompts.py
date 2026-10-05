"""
Testes automatizados para validação de prompts.
"""
import re
import pytest
import yaml
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import validate_prompt_structure

PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"
PROMPT_KEY = "bug_to_user_story_v2"


def load_prompts(file_path: str):
    """Carrega prompts do arquivo YAML."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def prompt_data():
    data = load_prompts(PROMPT_FILE)
    assert PROMPT_KEY in data, f"Chave '{PROMPT_KEY}' não encontrada em {PROMPT_FILE.name}"
    return data[PROMPT_KEY]


@pytest.fixture(scope="module")
def system_prompt(prompt_data):
    return prompt_data.get("system_prompt", "")


class TestPrompts:
    def test_prompt_has_system_prompt(self, prompt_data, system_prompt):
        """Verifica se o campo 'system_prompt' existe e não está vazio."""
        assert "system_prompt" in prompt_data, "Campo 'system_prompt' ausente"
        assert isinstance(system_prompt, str), "'system_prompt' deve ser texto"
        assert system_prompt.strip(), "'system_prompt' está vazio"

    def test_prompt_has_role_definition(self, system_prompt):
        """Verifica se o prompt define uma persona (ex: "Você é um Product Manager")."""
        assert re.search(r"Você é (um|uma)\b", system_prompt, re.IGNORECASE), \
            "Prompt não define uma persona com 'Você é um/uma ...'"
        assert re.search(r"Product Manager", system_prompt, re.IGNORECASE), \
            "Persona deveria ser de Product Manager"

    def test_prompt_mentions_format(self, system_prompt):
        """Verifica se o prompt exige formato Markdown ou User Story padrão."""
        assert re.search(r"formato", system_prompt, re.IGNORECASE), \
            "Prompt não especifica um formato de saída"
        # Template padrão de User Story
        for trecho in ("Como um", "eu quero", "para que"):
            assert trecho in system_prompt, f"Template de User Story sem '{trecho}'"
        # Critérios de aceitação no padrão Dado/Quando/Então
        assert "Critérios de Aceitação" in system_prompt
        for trecho in ("Dado que", "Quando", "Então"):
            assert trecho in system_prompt, f"Critérios sem '{trecho}'"

    def test_prompt_has_few_shot_examples(self, system_prompt):
        """Verifica se o prompt contém exemplos de entrada/saída (técnica Few-shot)."""
        exemplos = re.findall(r"^#+\s*Exemplo\s+\d+", system_prompt, re.MULTILINE)
        assert len(exemplos) >= 2, f"Few-shot requer ao menos 2 exemplos, encontrados: {len(exemplos)}"
        # Cada exemplo deve ter entrada (Relato) e saída (Resposta)
        assert system_prompt.count("Relato:") >= len(exemplos), "Exemplo sem entrada ('Relato:')"
        assert system_prompt.count("Resposta:") >= len(exemplos), "Exemplo sem saída ('Resposta:')"

    def test_prompt_no_todos(self, prompt_data, system_prompt):
        """Garante que você não esqueceu nenhum `[TODO]` no texto."""
        # Case-sensitive e por palavra inteira para não confundir com "todos"
        assert "[TODO]" not in system_prompt
        assert not re.search(r"\bTODO\b", system_prompt), "system_prompt contém TODO"
        assert not re.search(r"\bTODO\b", str(prompt_data.get("user_prompt", ""))), "user_prompt contém TODO"

    def test_minimum_techniques(self, prompt_data):
        """Verifica (através dos metadados do yaml) se pelo menos 2 técnicas foram listadas."""
        techniques = prompt_data.get("techniques_applied")
        assert isinstance(techniques, list), "'techniques_applied' deve ser uma lista"
        assert len(techniques) >= 2, f"Mínimo de 2 técnicas, encontradas: {len(techniques)}"
        assert all(isinstance(t, str) and t.strip() for t in techniques)

        is_valid, errors = validate_prompt_structure(prompt_data)
        assert is_valid, f"Estrutura do prompt inválida: {errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
