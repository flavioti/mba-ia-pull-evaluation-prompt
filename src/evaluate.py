"""
Script COMPLETO para avaliar prompts otimizados.

Este script:
1. Carrega dataset de avaliação de arquivo .jsonl (datasets/bug_to_user_story.jsonl)
2. Cria/atualiza dataset no LangSmith
3. Puxa prompts otimizados do LangSmith Hub (fonte única de verdade)
4. Executa prompts contra o dataset
5. Calcula 5 métricas (Helpfulness, Correctness, F1-Score, Clarity, Precision)
6. Publica resultados no dashboard do LangSmith
7. Exibe resumo no terminal

Suporta múltiplos providers de LLM:
- OpenAI (gpt-4o, gpt-4o-mini)
- Google Gemini (gemini-2.5-flash)

Configure o provider no arquivo .env através da variável LLM_PROVIDER.
"""

import os
import sys
import json
import time
import logging
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate
from utils import check_env_vars, format_score, print_section_header, content_to_text, get_llm as get_configured_llm
from metrics import evaluate_f1_score, evaluate_clarity, evaluate_precision

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# Controle o nível via .env / ambiente: LOG_LEVEL=DEBUG|INFO|WARNING (padrão: INFO)
#   INFO  -> progresso, tempos, scores por exemplo
#   DEBUG -> + inputs, prévia das respostas, raciocínio do juiz, prompt template
# ---------------------------------------------------------------------------
logger = logging.getLogger("evaluate")
logger.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO))
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(_handler)
logger.propagate = False


def _preview(value: Any, limit: int = 300) -> str:
    """Resumo de uma string em uma linha, para não poluir o terminal."""
    text = str(value).replace("\n", "\\n")
    return text if len(text) <= limit else f"{text[:limit]}... (+{len(text) - limit} chars)"


def get_llm():
    llm = get_configured_llm()
    logger.info(
        "LLM principal instanciado: %s (model=%s)",
        type(llm).__name__,
        getattr(llm, "model", None) or getattr(llm, "model_name", "?"),
    )
    return llm


def load_dataset_from_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    examples = []
    logger.info("Lendo dataset local: %s", Path(jsonl_path).resolve())

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Ignorar linhas vazias
                    example = json.loads(line)
                    examples.append(example)
                    logger.debug("  linha %d: inputs=%s", line_no, _preview(example.get("inputs")))
                else:
                    logger.debug("  linha %d vazia, ignorada", line_no)

        logger.info("JSONL lido: %d exemplos", len(examples))
        return examples

    except FileNotFoundError:
        logger.error("Arquivo não encontrado: %s", jsonl_path)
        print(f"❌ Arquivo não encontrado: {jsonl_path}")
        print("\nCertifique-se de que o arquivo datasets/bug_to_user_story.jsonl existe.")
        return []
    except json.JSONDecodeError as e:
        logger.error("JSON inválido na linha %d de %s: %s", line_no, jsonl_path, e)
        print(f"❌ Erro ao parsear JSONL: {e}")
        return []
    except Exception as e:
        logger.exception("Erro inesperado ao carregar dataset")
        print(f"❌ Erro ao carregar dataset: {e}")
        return []


def create_evaluation_dataset(client: Client, dataset_name: str, jsonl_path: str) -> str:
    print(f"Criando dataset de avaliação: {dataset_name}...")

    examples = load_dataset_from_jsonl(jsonl_path)

    if not examples:
        print("❌ Nenhum exemplo carregado do arquivo .jsonl")
        return dataset_name

    print(f"   ✓ Carregados {len(examples)} exemplos do arquivo {jsonl_path}")

    try:
        logger.info("Consultando LangSmith por dataset '%s'...", dataset_name)
        datasets = client.list_datasets(dataset_name=dataset_name)
        existing_dataset = None

        for ds in datasets:
            if ds.name == dataset_name:
                existing_dataset = ds
                break

        if existing_dataset:
            logger.info("Dataset já existe (id=%s), reutilizando sem recriar exemplos", existing_dataset.id)
            print(f"   ✓ Dataset '{dataset_name}' já existe, usando existente")
            return dataset_name
        else:
            logger.info("Dataset não existe, criando '%s'...", dataset_name)
            dataset = client.create_dataset(dataset_name=dataset_name)
            logger.info("Dataset criado (id=%s). Enviando %d exemplos...", dataset.id, len(examples))

            for idx, example in enumerate(examples, 1):
                client.create_example(
                    dataset_id=dataset.id,
                    inputs=example["inputs"],
                    outputs=example["outputs"]
                )
                logger.debug("  exemplo %d/%d enviado", idx, len(examples))

            print(f"   ✓ Dataset criado com {len(examples)} exemplos")
            return dataset_name

    except Exception as e:
        logger.exception("Falha ao criar/consultar dataset no LangSmith")
        print(f"   ⚠️  Erro ao criar dataset: {e}")
        return dataset_name


def pull_prompt_from_langsmith(prompt_name: str) -> ChatPromptTemplate:
    try:
        print(f"   Puxando prompt do LangSmith Hub: {prompt_name}")
        logger.info("Pull do prompt '%s' (public=permitido)...", prompt_name)
        start = time.perf_counter()
        # langchain.hub foi removido no langchain 1.x; usa o Client do langsmith
        prompt = Client().pull_prompt(
            prompt_name,
            dangerously_pull_public_prompt=True
        )
        logger.info(
            "Prompt carregado em %.2fs | tipo=%s | variáveis=%s",
            time.perf_counter() - start,
            type(prompt).__name__,
            getattr(prompt, "input_variables", "?"),
        )
        for msg in getattr(prompt, "messages", []):
            template = getattr(getattr(msg, "prompt", None), "template", "")
            logger.debug("  [%s] %s", type(msg).__name__, _preview(template, 500))
        print(f"   ✓ Prompt carregado com sucesso")
        return prompt

    except Exception as e:
        error_msg = str(e).lower()
        logger.error("Falha no pull de '%s': %s: %s", prompt_name, type(e).__name__, e)

        print(f"\n{'=' * 70}")
        print(f"❌ ERRO: Não foi possível carregar o prompt '{prompt_name}'")
        print(f"{'=' * 70}\n")

        if "not found" in error_msg or "404" in error_msg:
            print("⚠️  O prompt não foi encontrado no LangSmith Hub.\n")
            print("AÇÕES NECESSÁRIAS:")
            print("1. Verifique se você já fez push do prompt otimizado:")
            print(f"   python src/push_prompts.py")
            print()
            print("2. Confirme se o prompt foi publicado com sucesso em:")
            print(f"   https://smith.langchain.com/prompts")
            print()
            print(f"3. Certifique-se de que o nome do prompt está correto: '{prompt_name}'")
            print()
            print("4. Se você alterou o prompt no YAML, refaça o push:")
            print(f"   python src/push_prompts.py")
        else:
            print(f"Erro técnico: {e}\n")
            print("Verifique:")
            print("- LANGSMITH_API_KEY está configurada corretamente no .env")
            print("- Você tem acesso ao workspace do LangSmith")
            print("- Sua conexão com a internet está funcionando")

        print(f"\n{'=' * 70}\n")
        raise


def evaluate_prompt_on_example(
    prompt_template: ChatPromptTemplate,
    example: Any,
    llm: Any
) -> Dict[str, Any]:
    try:
        inputs = example.inputs if hasattr(example, 'inputs') else {}
        outputs = example.outputs if hasattr(example, 'outputs') else {}

        logger.debug(
            "Inputs: chaves=%s | %s",
            list(inputs.keys()) if isinstance(inputs, dict) else type(inputs).__name__,
            _preview(inputs),
        )

        chain = prompt_template | llm

        start = time.perf_counter()
        response = chain.invoke(inputs)
        raw_type = type(response.content).__name__
        answer = content_to_text(response.content)
        logger.info(
            "Resposta do prompt gerada em %.2fs (%d chars, content=%s)",
            time.perf_counter() - start, len(answer), raw_type,
        )
        logger.debug("Resposta: %s", _preview(answer, 500))

        reference = outputs.get("reference", "") if isinstance(outputs, dict) else ""
        logger.debug("Referência: %s", _preview(reference, 500))

        if isinstance(inputs, dict):
            question = inputs.get("question", inputs.get("bug_report", inputs.get("pr_title", "N/A")))
        else:
            question = "N/A"

        return {
            "answer": answer,
            "reference": reference,
            "question": question
        }

    except Exception as e:
        logger.error("Erro ao gerar resposta do exemplo: %s: %s", type(e).__name__, e)
        print(f"      ⚠️  Erro ao avaliar exemplo: {e}")
        import traceback
        print(f"      Traceback: {traceback.format_exc()}")
        return {
            "answer": "",
            "reference": "",
            "question": ""
        }


def evaluate_prompt(
    prompt_name: str,
    dataset_name: str,
    client: Client
) -> Dict[str, float]:
    print(f"\n🔍 Avaliando: {prompt_name}")
    logger.info("===== Início da avaliação de '%s' =====", prompt_name)
    eval_start = time.perf_counter()

    try:
        prompt_template = pull_prompt_from_langsmith(prompt_name)

        logger.info("Listando exemplos do dataset '%s'...", dataset_name)
        examples = list(client.list_examples(dataset_name=dataset_name))
        print(f"   Dataset: {len(examples)} exemplos")
        logger.info("%d exemplos carregados do LangSmith", len(examples))

        llm = get_llm()
        logger.info("Modelo juiz (EVAL_MODEL): %s", os.getenv("EVAL_MODEL", "gpt-4o"))

        f1_scores = []
        clarity_scores = []
        precision_scores = []
        skipped = 0

        print("   Avaliando exemplos...")

        for i, example in enumerate(examples, 1):
            ex_start = time.perf_counter()
            logger.info("--- Exemplo %d/%d (id=%s) ---", i, len(examples), getattr(example, "id", "?"))

            result = evaluate_prompt_on_example(prompt_template, example, llm)

            if result["answer"]:
                logger.info("Chamando juiz: F1, Clarity, Precision...")
                judge_start = time.perf_counter()
                f1 = evaluate_f1_score(result["question"], result["answer"], result["reference"])
                clarity = evaluate_clarity(result["question"], result["answer"], result["reference"])
                precision = evaluate_precision(result["question"], result["answer"], result["reference"])
                logger.info("Juiz concluído em %.2fs", time.perf_counter() - judge_start)

                for metric_name, metric_result in (("F1", f1), ("Clarity", clarity), ("Precision", precision)):
                    logger.debug("  %s reasoning: %s", metric_name, _preview(metric_result.get("reasoning", ""), 500))
                    if metric_result.get("score", 0.0) == 0.0:
                        logger.warning(
                            "%s retornou score 0.0 no exemplo %d (possível falha do juiz): %s",
                            metric_name, i, _preview(metric_result.get("reasoning", ""), 200),
                        )

                f1_scores.append(f1["score"])
                clarity_scores.append(clarity["score"])
                precision_scores.append(precision["score"])

                print(f"      [{i}/{len(examples)}] F1:{f1['score']:.2f} Clarity:{clarity['score']:.2f} Precision:{precision['score']:.2f}")
                logger.info(
                    "Exemplo %d concluído em %.2fs | médias parciais: F1=%.2f Clarity=%.2f Precision=%.2f",
                    i, time.perf_counter() - ex_start,
                    sum(f1_scores) / len(f1_scores),
                    sum(clarity_scores) / len(clarity_scores),
                    sum(precision_scores) / len(precision_scores),
                )
            else:
                skipped += 1
                logger.warning("Exemplo %d ignorado: resposta vazia (veja o erro acima)", i)

        logger.info(
            "Loop concluído: %d avaliados, %d ignorados, %.1fs no total",
            len(f1_scores), skipped, time.perf_counter() - eval_start,
        )

        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        avg_clarity = sum(clarity_scores) / len(clarity_scores) if clarity_scores else 0.0
        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0

        avg_helpfulness = (avg_clarity + avg_precision) / 2
        avg_correctness = (avg_f1 + avg_precision) / 2

        logger.info(
            "Médias finais: F1=%.4f Clarity=%.4f Precision=%.4f | Helpfulness=%.4f Correctness=%.4f",
            avg_f1, avg_clarity, avg_precision, avg_helpfulness, avg_correctness,
        )

        return {
            "helpfulness": round(avg_helpfulness, 4),
            "correctness": round(avg_correctness, 4),
            "f1_score": round(avg_f1, 4),
            "clarity": round(avg_clarity, 4),
            "precision": round(avg_precision, 4)
        }

    except Exception as e:
        logger.exception("Erro na avaliação de '%s'; retornando scores zerados", prompt_name)
        print(f"   ❌ Erro na avaliação: {e}")
        return {
            "helpfulness": 0.0,
            "correctness": 0.0,
            "f1_score": 0.0,
            "clarity": 0.0,
            "precision": 0.0
        }


def display_results(prompt_name: str, scores: Dict[str, float]) -> bool:
    print("\n" + "=" * 50)
    print(f"Prompt: {prompt_name}")
    print("=" * 50)

    print("\nMétricas Derivadas:")
    print(f"  - Helpfulness: {format_score(scores['helpfulness'], threshold=0.8)}")
    print(f"  - Correctness: {format_score(scores['correctness'], threshold=0.8)}")

    print("\nMétricas Base:")
    print(f"  - F1-Score: {format_score(scores['f1_score'], threshold=0.8)}")
    print(f"  - Clarity: {format_score(scores['clarity'], threshold=0.8)}")
    print(f"  - Precision: {format_score(scores['precision'], threshold=0.8)}")

    average_score = sum(scores.values()) / len(scores)

    print("\n" + "-" * 50)
    print(f"📊 MÉDIA GERAL: {average_score:.4f}")
    print("-" * 50)

    all_above_threshold = all(score >= 0.8 for score in scores.values())
    passed = all_above_threshold and average_score >= 0.8

    logger.info(
        "Decisão: média=%.4f | todas>=0.8? %s | aprovado? %s | scores=%s",
        average_score, all_above_threshold, passed, scores,
    )

    if passed:
        print(f"\n✅ STATUS: APROVADO - Todas as métricas >= 0.8")
    else:
        print(f"\n❌ STATUS: REPROVADO")
        failed_metrics = [name for name, score in scores.items() if score < 0.8]
        if failed_metrics:
            print(f"⚠️  Métricas abaixo de 0.8: {', '.join(failed_metrics)}")
        print(f"⚠️  Média atual: {average_score:.4f} | Necessário: 0.8000")

    return passed


def main():
    print_section_header("AVALIAÇÃO DE PROMPTS OTIMIZADOS")
    main_start = time.perf_counter()

    provider = os.getenv("LLM_PROVIDER", "openai")
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    eval_model = os.getenv("EVAL_MODEL", "gpt-4o")

    print(f"Provider: {provider}")
    print(f"Modelo Principal: {llm_model}")
    print(f"Modelo de Avaliação: {eval_model}\n")

    logger.info(
        "Configuração: provider=%s | LLM_MODEL=%s | EVAL_MODEL=%s | LOG_LEVEL=%s | cwd=%s",
        provider, llm_model, eval_model, logging.getLevelName(logger.level), os.getcwd(),
    )

    required_vars = ["LANGSMITH_API_KEY", "LLM_PROVIDER"]
    if provider == "openai":
        required_vars.append("OPENAI_API_KEY")
    elif provider in ["google", "gemini"]:
        required_vars.append("GOOGLE_API_KEY")

    # Apenas presença (nunca o valor) das chaves
    logger.info(
        "Variáveis obrigatórias: %s",
        {var: ("ok" if os.getenv(var) else "FALTANDO") for var in required_vars},
    )

    if not check_env_vars(required_vars):
        logger.error("Encerrando: variáveis de ambiente faltando")
        return 1

    client = Client()
    project_name = os.getenv("LANGSMITH_PROJECT", "prompt-optimization-challenge-resolved")
    logger.info("Projeto LangSmith: %s", project_name)

    jsonl_path = "datasets/bug_to_user_story.jsonl"

    if not Path(jsonl_path).exists():
        logger.error("Dataset não encontrado: %s (cwd=%s)", jsonl_path, os.getcwd())
        print(f"❌ Arquivo de dataset não encontrado: {jsonl_path}")
        print("\nCertifique-se de que o arquivo existe antes de continuar.")
        return 1

    dataset_name = f"{project_name}-eval"
    create_evaluation_dataset(client, dataset_name, jsonl_path)

    print("\n" + "=" * 70)
    print("PROMPTS PARA AVALIAR")
    print("=" * 70)
    print("\nEste script irá puxar prompts do LangSmith Hub.")
    print("Certifique-se de ter feito push dos prompts antes de avaliar:")
    print("  python src/push_prompts.py\n")

    username = os.getenv("USERNAME_LANGSMITH_HUB", "")
    if not username:
        logger.error("USERNAME_LANGSMITH_HUB não configurada")
        print("❌ USERNAME_LANGSMITH_HUB não configurada no .env")
        print("   Configure seu username do LangSmith Hub antes de continuar.")
        return 1

    prompts_to_evaluate = [
        f"{username}/bug_to_user_story_v2",
    ]

    logger.info("Prompts a avaliar (%d): %s", len(prompts_to_evaluate), prompts_to_evaluate)

    all_passed = True
    evaluated_count = 0
    results_summary = []

    for prompt_name in prompts_to_evaluate:
        evaluated_count += 1
        logger.info("Prompt %d/%d: %s", evaluated_count, len(prompts_to_evaluate), prompt_name)

        try:
            scores = evaluate_prompt(prompt_name, dataset_name, client)

            passed = display_results(prompt_name, scores)
            all_passed = all_passed and passed

            results_summary.append({
                "prompt": prompt_name,
                "scores": scores,
                "passed": passed
            })

        except Exception as e:
            logger.exception("Falha ao avaliar '%s'", prompt_name)
            print(f"\n❌ Falha ao avaliar '{prompt_name}': {e}")
            all_passed = False

            results_summary.append({
                "prompt": prompt_name,
                "scores": {
                    "helpfulness": 0.0,
                    "correctness": 0.0,
                    "f1_score": 0.0,
                    "clarity": 0.0,
                    "precision": 0.0
                },
                "passed": False
            })

    print("\n" + "=" * 50)
    print("RESUMO FINAL")
    print("=" * 50 + "\n")

    if evaluated_count == 0:
        print("⚠️  Nenhum prompt foi avaliado")
        return 1

    print(f"Prompts avaliados: {evaluated_count}")
    print(f"Aprovados: {sum(1 for r in results_summary if r['passed'])}")
    print(f"Reprovados: {sum(1 for r in results_summary if not r['passed'])}\n")

    logger.info(
        "Resumo: avaliados=%d aprovados=%d reprovados=%d | tempo total=%.1fs | all_passed=%s",
        evaluated_count,
        sum(1 for r in results_summary if r['passed']),
        sum(1 for r in results_summary if not r['passed']),
        time.perf_counter() - main_start,
        all_passed,
    )

    if all_passed:
        print("✅ Todos os prompts atingiram todas as métricas >= 0.8!")
        print(f"\n✓ Confira os resultados em:")
        print(f"  https://smith.langchain.com/projects/{project_name}")
        print("\nPróximos passos:")
        print("1. Documente o processo no README.md")
        print("2. Capture screenshots das avaliações")
        print("3. Faça commit e push para o GitHub")
        return 0
    else:
        print("⚠️  Alguns prompts não atingiram todas as métricas >= 0.8")
        print("\nPróximos passos:")
        print("1. Refatore os prompts com score baixo")
        print("2. Faça push novamente: python src/push_prompts.py")
        print("3. Execute: python src/evaluate.py novamente")
        return 1

if __name__ == "__main__":
    main()
