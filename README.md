# Pull, Otimização e Avaliação de Prompts com LangChain e LangSmith

## Objetivo

Você deve entregar um software capaz de:

- Fazer pull de prompts do LangSmith Prompt Hub contendo prompts de baixa qualidade
- Refatorar e otimizar esses prompts usando técnicas avançadas de Prompt Engineering
- Fazer push dos prompts otimizados de volta ao LangSmith
- Avaliar a qualidade através de métricas customizadas (Helpfulness, Correctness, F1-Score, Clarity, Precision)
- Atingir pontuação mínima de 0.8 (80%) em todas as métricas de avaliação

## Exemplo no CLI

Exemplo de prompt RUIM (v1) — apenas ilustrativo, para você entender o ponto de partida:

```
==================================================
Prompt: {seu_username}/bug_to_user_story_v1
==================================================

Métricas Derivadas:
  - Helpfulness: 0.45 ✗
  - Correctness: 0.52 ✗

Métricas Base:
  - F1-Score: 0.48 ✗
  - Clarity: 0.50 ✗
  - Precision: 0.46 ✗

❌ STATUS: REPROVADO
⚠️  Métricas abaixo de 0.8: helpfulness, correctness, f1_score, clarity, precision
```

Exemplo de prompt OTIMIZADO (v2) — seu objetivo é chegar aqui:

```
# Após refatorar os prompts e fazer push
python src/push_prompts.py

# Executar avaliação
python src/evaluate.py

Executando avaliação dos prompts...
==================================================
Prompt: {seu_username}/bug_to_user_story_v2
==================================================

Métricas Derivadas:
  - Helpfulness: 0.94 ✓
  - Correctness: 0.96 ✓

Métricas Base:
  - F1-Score: 0.93 ✓
  - Clarity: 0.95 ✓
  - Precision: 0.92 ✓

✅ STATUS: APROVADO - Todas as métricas >= 0.8
```

## Tecnologias obrigatórias

- Linguagem: Python 3.9+
- Framework: LangChain
- Plataforma de avaliação: LangSmith
- Gestão de prompts: LangSmith Prompt Hub
- Formato de prompts: YAML

## Pacotes recomendados

```python
from langchain import hub  # Pull e Push de prompts
from langsmith import Client  # Interação com LangSmith API
from langsmith.evaluation import evaluate  # Avaliação de prompts
from langchain_openai import ChatOpenAI  # LLM OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI  # LLM Gemini
```

## OpenAI

- Crie uma API Key da OpenAI: https://platform.openai.com/api-keys
- Você vai precisar de um modelo de LLM para responder e de um modelo de LLM para avaliação. Consulte a documentação oficial da OpenAI para ver os modelos disponíveis.
- Custo estimado: ~$1-5 para completar o desafio

## Gemini (modelo free)

- Crie uma API Key da Google: https://aistudio.google.com/app/apikey
- Você vai precisar de um modelo de LLM para responder e de um modelo de LLM para avaliação. Consulte a documentação oficial do Google para ver os modelos disponíveis.
- Os limites de requisições gratuitas mudam com frequência. Consulte os limites atuais na documentação oficial do Google.

## Escolha dos modelos

Este desafio não fixa modelos. Nomes e versões mudam com frequência e alguns são descontinuados, então faz parte do desafio consultar a documentação oficial do provedor que você escolher, ver quais modelos estão disponíveis no momento e selecionar os que atendem ao objetivo. Você pode usar o mesmo modelo para responder e para avaliar, ou um modelo mais capaz na avaliação.

## Requisitos

### 1. Pull do Prompt inicial do LangSmith

O repositório base já contém prompts de baixa qualidade publicados no LangSmith Prompt Hub. Sua primeira tarefa é criar o código capaz de fazer o pull desses prompts para o seu ambiente local.

Tarefas:

- Configurar suas credenciais do LangSmith no arquivo .env (conforme o arquivo .env.example)
- Implementar o script src/pull_prompts.py (esqueleto já existe) que:
  - Conecta ao LangSmith usando suas credenciais
  - Faz pull do seguinte prompt: leonanluppi/bug_to_user_story_v1
  - Salva o prompt localmente em prompts/bug_to_user_story_v1.yml

### 2. Otimização do Prompt

Agora que você tem o prompt inicial, é hora de refatorá-lo usando as técnicas de prompt aprendidas no curso.

Tarefas:

- Analisar o prompt em prompts/bug_to_user_story_v1.yml
- Criar um novo arquivo prompts/bug_to_user_story_v2.yml com suas versões otimizadas
- Aplicar obrigatoriamente Few-shot Learning (exemplos claros de entrada/saída) e pelo menos uma das seguintes técnicas adicionais:
  - Chain of Thought (CoT): Instruir o modelo a "pensar passo a passo"
  - Tree of Thought: Explorar múltiplos caminhos de raciocínio
  - Skeleton of Thought: Estruturar a resposta em etapas claras
  - ReAct: Raciocínio + Ação para tarefas complexas
  - Role Prompting: Definir persona e contexto detalhado
- Documentar no README.md quais técnicas você escolheu e por quê

Requisitos do prompt otimizado:

- Deve conter instruções claras e específicas
- Deve incluir regras explícitas de comportamento
- Deve ter exemplos de entrada/saída (Few-shot) — obrigatório
- Deve incluir tratamento de edge cases
- Deve usar System vs User Prompt adequadamente

### 3. Push e Avaliação

Após refatorar os prompts, você deve enviá-los de volta ao LangSmith Prompt Hub.

Tarefas:

- Implementar o script src/push_prompts.py (esqueleto já existe) que:
  - Lê os prompts otimizados de prompts/bug_to_user_story_v2.yml
  - Faz push para o LangSmith com nomes versionados: {seu_username}/bug_to_user_story_v2
  - Adiciona metadados (tags, descrição, técnicas utilizadas)
- Executar o script e verificar no dashboard do LangSmith se os prompts foram publicados
- Deixá-lo público

### 4. Iteração

Espera-se 3-5 iterações.

- Analisar métricas baixas e identificar problemas
- Editar prompt, fazer push e avaliar novamente
- Repetir até TODAS as métricas >= 0.8

```
Critério de Aprovação:
- Helpfulness >= 0.8
- Correctness >= 0.8
- F1-Score >= 0.8
- Clarity >= 0.8
- Precision >= 0.8

MÉDIA das 5 métricas >= 0.8
```

IMPORTANTE: TODAS as 5 métricas devem estar >= 0.8, não apenas a média!

### 5. Testes de Validação

O que você deve fazer: Edite o arquivo tests/test_prompts.py e implemente, no mínimo, os 6 testes abaixo usando pytest:

- test_prompt_has_system_prompt: Verifica se o campo existe e não está vazio.
- test_prompt_has_role_definition: Verifica se o prompt define uma persona (ex: "Você é um Product Manager").
- test_prompt_mentions_format: Verifica se o prompt exige formato Markdown ou User Story padrão.
- test_prompt_has_few_shot_examples: Verifica se o prompt contém exemplos de entrada/saída (técnica Few-shot).
- test_prompt_no_todos: Garante que você não esqueceu nenhum [TODO] no texto.
- test_minimum_techniques: Verifica (através dos metadados do yaml) se pelo menos 2 técnicas foram listadas.

Como validar:

```
pytest tests/test_prompts.py
```

## Estrutura obrigatória do projeto

Faça um fork do repositório base: https://github.com/devfullcycle/mba-ia-pull-evaluation-prompt

```
mba-ia-pull-evaluation-prompt/
├── .env.example              # Template das variáveis de ambiente
├── requirements.txt          # Dependências Python
├── README.md                 # Sua documentação do processo
│
├── prompts/
│   ├── bug_to_user_story_v1.yml  # Prompt inicial (já incluso)
│   └── bug_to_user_story_v2.yml  # Seu prompt otimizado (criar)
│
├── datasets/
│   └── bug_to_user_story.jsonl   # 15 exemplos de bugs (já incluso)
│
├── src/
│   ├── pull_prompts.py       # Pull do LangSmith (implementar)
│   ├── push_prompts.py       # Push ao LangSmith (implementar)
│   ├── evaluate.py           # Avaliação automática (pronto)
│   ├── metrics.py            # 5 métricas implementadas (pronto)
│   └── utils.py              # Funções auxiliares (pronto)
│
├── tests/
│   └── test_prompts.py       # Testes de validação (implementar)
```

O que você deve implementar:

- prompts/bug_to_user_story_v2.yml — Criar do zero com seu prompt otimizado
- src/pull_prompts.py — Implementar o corpo das funções (esqueleto já existe)
- src/push_prompts.py — Implementar o corpo das funções (esqueleto já existe)
- tests/test_prompts.py — Implementar os 6 testes de validação (esqueleto já existe)
- README.md — Documentar seu processo de otimização

O que já vem pronto (não alterar):

- src/evaluate.py — Script de avaliação completo
- src/metrics.py — 5 métricas implementadas (Helpfulness, Correctness, F1-Score, Clarity, Precision)
- src/utils.py — Funções auxiliares
- datasets/bug_to_user_story.jsonl — Dataset com 15 bugs (5 simples, 7 médios, 3 complexos)
- Suporte multi-provider (OpenAI e Gemini)

## VirtualEnv para Python

Crie e ative um ambiente virtual antes de instalar dependências:

```
python3 -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Ordem de execução

1. Executar pull dos prompts ruins

```
python src/pull_prompts.py
```

2. Refatorar prompts

Edite manualmente o arquivo prompts/bug_to_user_story_v2.yml aplicando as técnicas aprendidas no curso.

3. Fazer push dos prompts otimizados

```
python src/push_prompts.py
```

4. Executar avaliação

```
python src/evaluate.py
```

## Entregável

1. Repositório público no GitHub (fork do repositório base) contendo:

- Todo o código-fonte implementado
- Arquivo prompts/bug_to_user_story_v2.yml 100% preenchido e funcional
- Arquivo README.md atualizado

2. README.md deve conter:

A) Seção "Técnicas Aplicadas (Fase 2)":

- Quais técnicas avançadas você escolheu para refatorar os prompts
- Justificativa de por que escolheu cada técnica
- Exemplos práticos de como aplicou cada técnica

B) Seção "Resultados Finais":

- Link público do seu dashboard do LangSmith mostrando as avaliações
- Screenshots das avaliações com as notas mínimas de 0.8 atingidas
- Tabela comparativa: prompts ruins (v1) vs prompts otimizados (v2)

C) Seção "Como Executar":

- Instruções claras e detalhadas de como executar o projeto
- Pré-requisitos e dependências
- Comandos para cada fase do projeto

3. Evidências no LangSmith:

- Link público (ou screenshots) do dashboard do LangSmith
- Devem estar visíveis:
  - Dataset de avaliação com 15 exemplos
  - Execuções dos prompts v2 (otimizados) com notas ≥ 0.8
  - Tracing detalhado de pelo menos 3 exemplos

## Dicas Finais

- Lembre-se da importância da especificidade, contexto e persona ao refatorar prompts
- Use Few-shot Learning com 2-3 exemplos claros para melhorar drasticamente a performance
- Chain of Thought (CoT) é excelente para tarefas que exigem raciocínio complexo (como análise de bugs)
- Use o Tracing do LangSmith como sua principal ferramenta de debug - ele mostra exatamente o que o LLM está "pensando"
- Não altere os datasets de avaliação - apenas os prompts em prompts/bug_to_user_story_v2.yml
- Itere, itere, itere - é normal precisar de 3-5 iterações para atingir 0.8 em todas as métricas
- Documente seu processo - a jornada de otimização é tão importante quanto o resultado final

---

# Documentação da Solução

## Técnicas Aplicadas (Fase 2)

O prompt otimizado está em [prompts/bug_to_user_story_v2.yml](prompts/bug_to_user_story_v2.yml). Os metadados do YAML listam as técnicas usadas:

```yaml
techniques_applied:
  - "role-prompting"
  - "few-shot-learning"
  - "chain-of-thought"
  - "skeleton-of-thought"
```

### Problemas do prompt v1

| Problema no v1 | Correção no v2 |
|---|---|
| Persona genérica ("assistente que ajuda...") | Role Prompting: Product Manager sênior com experiência em ágil e QA |
| Instrução vaga ("crie uma user story") | Tarefa, regras explícitas e formato de saída definido |
| Nenhum exemplo | Few-shot com 3 exemplos (simples, médio, complexo) |
| Nenhum tratamento de casos de borda | Seção dedicada a edge cases |
| `{bug_report}` duplicado no system e no user prompt | System contém só instruções; o relato entra apenas no `user_prompt` |
| Pedia o prefixo "User Story gerada:" na saída | Regra proibindo qualquer texto fora da User Story |

### 1. Role Prompting

**Por quê:** a métrica de Helpfulness e a qualidade do texto dependem do ponto de vista de quem escreve. Uma PM sênior pensa em valor para o usuário, critérios testáveis e prioridade, que é exatamente o que uma boa User Story precisa.

**Como foi aplicado:**

```text
# PERSONA
Você é uma Product Manager sênior, com 10 anos de experiência em produtos digitais,
metodologias ágeis e QA. Você transforma relatos de bugs (muitas vezes vagos, técnicos
ou desorganizados) em User Stories claras, testáveis e centradas no valor para o usuário.
```

### 2. Few-shot Learning (obrigatória)

**Por quê:** as métricas F1-Score e Precision comparam a resposta com a referência do dataset. Mostrar exemplos concretos é a forma mais confiável de alinhar estrutura, vocabulário (Dado/Quando/Então) e nível de detalhe ao esperado. Como o dataset tem bugs de três níveis de complexidade, há um exemplo para cada nível.

**Como foi aplicado:** três pares `Relato:` → `Resposta:` dentro do system prompt:

- **Exemplo 1 (simples):** botão "Salvar" que não responde no Firefox → user story curta + 5 critérios.
- **Exemplo 2 (médio):** exportação CSV com HTTP 500 → user story + critérios + seção `Contexto Técnico` preservando endpoint e log.
- **Exemplo 3 (complexo):** módulo de agendamento com 3 falhas → formato completo com `=== USER STORY PRINCIPAL ===`, critérios A/B/C, critérios técnicos, contexto do bug e tasks.

Os exemplos usam domínios diferentes dos do dataset (endereço, relatório financeiro, agendamento médico) para ensinar o **formato** sem "vazar" respostas.

### 3. Chain of Thought (CoT)

**Por quê:** os relatos variam muito (de uma linha a relatos longos com logs e impacto). Sem um raciocínio guiado, o modelo tende a errar a persona, inventar dados ou escolher o formato errado, o que derruba Correctness e Precision.

**Como foi aplicado:** uma seção `# RACIOCÍNIO` com 6 passos que o modelo executa **internamente** (a regra 1 proíbe expor o raciocínio na resposta):

```text
1. PERSONA: quem sente o problema? Use o papel mais específico possível...
2. FATOS: liste os dados concretos do relato (IDs, valores, endpoints, códigos HTTP, logs...)
3. COMPLEXIDADE: classifique o relato em SIMPLES, MÉDIO ou COMPLEXO.
4. FORMATO: escolha o template correspondente à complexidade.
5. BENEFÍCIO: o "para que" deve expressar valor real para a persona.
6. CRITÉRIOS: cada critério deve ser específico e testável...
```

### 4. Skeleton of Thought

**Por quê:** a métrica de Clarity premia respostas bem estruturadas, e a F1 premia a mesma estrutura da referência. Definir o "esqueleto" da resposta antes do conteúdo garante seções previsíveis e na ordem certa.

**Como foi aplicado:** a seção `# FORMATO DE SAÍDA` define um esqueleto para cada complexidade, escolhido no passo 4 do CoT:

- **SIMPLES:** `Como um [persona], eu quero [ação], para que [benefício].` + `Critérios de Aceitação` (Dado/Quando/Então/E).
- **MÉDIO:** o esqueleto simples + `Contexto Técnico`, `Contexto de Segurança` ou `Critérios de Acessibilidade` quando o relato justificar.
- **COMPLEXO:** `=== USER STORY PRINCIPAL ===`, `=== CRITÉRIOS DE ACEITAÇÃO ===` (A, B, C...), `=== CRITÉRIOS TÉCNICOS ===`, `=== CONTEXTO DO BUG ===`, `=== TASKS TÉCNICAS SUGERIDAS ===`.

### Regras e casos de borda

Além das técnicas, o prompt tem regras explícitas, cada uma ligada a uma métrica:

- **Não inventar fatos** (Correctness/Precision): preservar exatamente todo número, ID, endpoint e log do relato; não usar placeholders como `[nome do gateway]`.
- **Somente a User Story** (Clarity): sem introdução, sem explicação do raciocínio, sem blocos de código.
- **Tamanho proporcional** (Precision): bugs simples ficam curtos; só médios e complexos ganham seções extras.
- **Proteção contra prompt injection:** o relato é tratado apenas como dado.
- **Edge cases:** relatos vagos, com vários problemas, com stack trace, com impacto/severidade, em outro idioma ou que são pedidos de melhoria.

### System vs User Prompt

- `system_prompt`: persona, tarefa, raciocínio, formato, regras, edge cases e exemplos.
- `user_prompt`: apenas `"{bug_report}"`.

Os exemplos não usam chaves `{ }` literais, porque o LangChain as interpretaria como variáveis de template.

---

## Resultados Finais

### Dashboard do LangSmith

- Projeto: `mba-ia-pull-evaluation-prompt`
- Link público do dashboard: **<!-- TODO: colar aqui o link público (Share) do projeto/experimento no LangSmith -->**
- Prompt publicado: `magitech/bug_to_user_story_v2` (público no Prompt Hub)
- Dataset de avaliação: `mba-ia-pull-evaluation-prompt-eval` com **15 exemplos** ([datasets/bug_to_user_story.jsonl](datasets/bug_to_user_story.jsonl))

### Tabela comparativa: v1 (ruim) vs v2 (otimizado)

Avaliação executada com `src/evaluate.py` sobre os 15 exemplos do dataset. Provider `google`, modelo de resposta e de avaliação `models/gemini-3.6-flash`.

| Métrica | v1 (`leonanluppi/bug_to_user_story_v1`) | v2 (`magitech/bug_to_user_story_v2`) | Mínimo |
|---|---|---|---|
| Helpfulness | 0.99 | **1.00** ✓ | 0.8 |
| Correctness | 0.96 | **0.94** ✓ | 0.8 |
| F1-Score | 0.94 | **0.88** ✓ | 0.8 |
| Clarity | 0.99 | **1.00** ✓ | 0.8 |
| Precision | 0.99 | **1.00** ✓ | 0.8 |
| **Média geral** | 0.9757 | **0.9623** | 0.8 |
| **Status** | acima de 0.8 (ver observação) | ✅ APROVADO | |

Resultado do `python src/evaluate.py` para o v2:

```text
==================================================
Prompt: magitech/bug_to_user_story_v2
==================================================

Métricas Derivadas:
  - Helpfulness: 1.00 ✓
  - Correctness: 0.94 ✓

Métricas Base:
  - F1-Score: 0.88 ✓
  - Clarity: 1.00 ✓
  - Precision: 1.00 ✓

--------------------------------------------------
📊 MÉDIA GERAL: 0.9623
--------------------------------------------------

✅ STATUS: APROVADO - Todas as métricas >= 0.8
```

> **Observação sobre o v1:** com o `models/gemini-3.6-flash` como juiz, o v1 também ficou acima de 0.8 e ligeiramente acima do v2 na F1-Score (0.94 vs 0.88), métrica que compara a resposta com a referência do dataset. As notas mostram que esse modelo avaliador é pouco exigente e não diferencia bem os dois prompts. O ganho do v2 está na consistência do formato (template fixo por complexidade, critérios Dado/Quando/Então sempre presentes), nas regras contra dados inventados e na proteção contra prompt injection, conforme descrito em [Técnicas Aplicadas](#técnicas-aplicadas-fase-2).

### Screenshots

<!-- TODO: adicionar screenshots do LangSmith na pasta docs/ (a pasta screenshots/ está no .gitignore):
     1. Dataset com 15 exemplos
     2. Execução do prompt v2 com as notas >= 0.8
     3. Tracing detalhado de pelo menos 3 exemplos -->

---

## Como Executar

### Pré-requisitos

- Python **3.10** (testado com 3.10.18; os pacotes fixados em `requirements.txt` não instalam em Python ≥ 3.14)
- Conta no [LangSmith](https://smith.langchain.com) com API key e username do Prompt Hub
- API key do Google Gemini (<https://aistudio.google.com/app/apikey>) ou da OpenAI

### 1. Instalação

```bash
git clone https://github.com/<seu-usuario>/mba-ia-pull-evaluation-prompt.git
cd mba-ia-pull-evaluation-prompt

python3.10 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configuração

```bash
cp .env.example .env
```

Preencha no `.env`:

```env
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=mba-ia-pull-evaluation-prompt
USERNAME_LANGSMITH_HUB=seu_username

LLM_PROVIDER=google              # ou openai
GOOGLE_API_KEY=...               # ou OPENAI_API_KEY=...
LLM_MODEL=models/gemini-3.6-flash
EVAL_MODEL=models/gemini-3.6-flash
```

### 3. Fases do projeto

```bash
# Fase 1 - Pull do prompt ruim (salva em prompts/bug_to_user_story_v1.yml)
python src/pull_prompts.py

# Fase 2 - Refatoração: editar prompts/bug_to_user_story_v2.yml

# Testes de validação do prompt
pytest tests/test_prompts.py -v

# Fase 3 - Push do prompt otimizado para {USERNAME_LANGSMITH_HUB}/bug_to_user_story_v2
python src/push_prompts.py

# Fase 4 - Avaliação (cria o dataset no LangSmith e calcula as 5 métricas)
python src/evaluate.py
```

### Testes

[tests/test_prompts.py](tests/test_prompts.py) valida o `prompts/bug_to_user_story_v2.yml`:

| Teste | O que verifica |
|---|---|
| `test_prompt_has_system_prompt` | `system_prompt` existe e não está vazio |
| `test_prompt_has_role_definition` | Persona definida com "Você é um/uma ... Product Manager" |
| `test_prompt_mentions_format` | Exige formato e o template "Como um / eu quero / para que" + "Dado que / Quando / Então" |
| `test_prompt_has_few_shot_examples` | Pelo menos 2 exemplos, cada um com `Relato:` (entrada) e `Resposta:` (saída) |
| `test_prompt_no_todos` | Nenhum `TODO` / `[TODO]` no prompt |
| `test_minimum_techniques` | `techniques_applied` lista ao menos 2 técnicas e a estrutura passa em `validate_prompt_structure` |

```text
tests/test_prompts.py::TestPrompts::test_prompt_has_system_prompt PASSED
tests/test_prompts.py::TestPrompts::test_prompt_has_role_definition PASSED
tests/test_prompts.py::TestPrompts::test_prompt_mentions_format PASSED
tests/test_prompts.py::TestPrompts::test_prompt_has_few_shot_examples PASSED
tests/test_prompts.py::TestPrompts::test_prompt_no_todos PASSED
tests/test_prompts.py::TestPrompts::test_minimum_techniques PASSED
============================== 6 passed ==============================
```

### Observação sobre limites do Gemini

No plano gratuito do Gemini o limite diário de requisições (RPD) é baixo. Cada avaliação faz 1 chamada de geração + chamadas de avaliação por exemplo (15 exemplos), então pode ser necessário um plano pago (Tier 1) ou esperar a renovação da cota.
