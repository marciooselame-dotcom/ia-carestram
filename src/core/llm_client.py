import json
import logging
import os
import requests
from typing import Generator, List, Dict, Any, Optional
from dotenv import load_dotenv

# Load env variables
load_dotenv()

logger = logging.getLogger(__name__)

class LLMClient:
    """Handles communications with LLM services (Ollama local or NVIDIA cloud API).
    
    Provides interfaces to stream responses, format raw inputs
    using structured prompt templates, and degrade gracefully by falling back.
    """

    def __init__(self) -> None:
        """Initializes the LLMClient using environment variables or safe defaults."""
        # NVIDIA cloud API (faster, uses GPU in cloud)
        self.use_nvidia: bool = os.getenv("USE_NVIDIA", "false").lower() == "true"
        self.nvidia_api_key: str = os.getenv("NVIDIA_API_KEY", "")
        self.nvidia_model: str = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")
        # Ollama local
        self.base_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self.default_model: str = os.getenv("DEFAULT_MODEL", "gemma3:4b")
        self.fallback_model: str = os.getenv("FALLBACK_MODEL", "qwen2.5:0.5b")

    def get_available_models(self) -> List[str]:
        """Queries the local Ollama instance for installed models.

        Returns:
            List[str]: A list of installed model names (e.g. ['gemma3:4b', 'qwen2.5:0.5b']).

        Complexity:
            Time Complexity: O(M) where M is the number of models installed in Ollama.
            Space Complexity: O(M) to store the list of model names.
        """
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [model["name"] for model in data.get("models", [])]
                logger.info("Successfully fetched available Ollama models: %s", models)
                return models
            logger.warning("Failed to fetch Ollama models. HTTP Status: %d", response.status_code)
        except Exception as e:
            logger.error("Failed to connect to Ollama. Error: %s", str(e), exc_info=True)
        return []

    def get_active_model(self) -> str:
        """Determines the active model by checking availability in Ollama.

        If the default model is not installed, it falls back to the fallback model.
        If neither is installed, it uses the first available model or the default string.

        Returns:
            str: The name of the model to use.
        """
        available = self.get_available_models()
        if not available:
            logger.warning("No models found in Ollama. Defaulting to string: %s", self.default_model)
            return self.default_model

        # Check default model (exact match or prefix match)
        for model in available:
            if model.startswith(self.default_model) or self.default_model in model:
                return model

        # Check fallback model
        for model in available:
            if model.startswith(self.fallback_model) or self.fallback_model in model:
                logger.info("Default model '%s' not found. Using fallback: %s", self.default_model, model)
                return model

        # Use first available as last resort
        logger.warning("Neither default '%s' nor fallback '%s' found. Using first available: %s",
                       self.default_model, self.fallback_model, available[0])
        return available[0]

    def stream_chat(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        model_override: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Streams a chat response from Ollama or NVIDIA cloud API.

        Args:
            system_prompt (str): Instructions defining the LLM's persona and constraints.
            user_prompt (str): The specific input or query from the user.
            model_override (Optional[str]): Explicit model selection, bypassing auto-detection.

        Yields:
            Generator[str, None, None]: A generator yielding text fragments as they arrive.
        """
        if self.use_nvidia and self.nvidia_api_key:
            yield from self._stream_chat_nvidia(system_prompt, user_prompt, model_override)
        else:
            yield from self._stream_chat_ollama(system_prompt, user_prompt, model_override)

    def _stream_chat_ollama(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        model_override: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Streams a chat response from local Ollama."""
        model = model_override if model_override else self.get_active_model()
        url = f"{self.base_url}/api/chat"
        
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": True
        }

        logger.info("Sending request to Ollama using model: %s", model)
        
        try:
            response = requests.post(url, json=payload, stream=True, timeout=120)
            if response.status_code != 200:
                logger.error("Ollama returned HTTP status %d: %s", response.status_code, response.text)
                yield f"[Erro HTTP {response.status_code} na comunicação com Ollama]"
                return

            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    try:
                        chunk = json.loads(decoded_line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError as je:
                        logger.warning("Failed to decode JSON chunk: %s. Line: %s", str(je), decoded_line)

        except Exception as e:
            logger.error("Connection error while streaming from Ollama. Error: %s", str(e), exc_info=True)
            yield f"[Erro de Conexão com Ollama: {str(e)}]"

    def _stream_chat_nvidia(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        model_override: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Streams a chat response from NVIDIA cloud API (OpenAI-compatible)."""
        model = model_override if model_override else self.nvidia_model
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.nvidia_api_key}",
            "Content-Type": "application/json"
        }
        
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": True,
            "max_tokens": 2048,
            "temperature": 0.1
        }

        logger.info("Sending request to NVIDIA API using model: %s", model)
        
        try:
            response = requests.post(url, json=payload, headers=headers, stream=True, timeout=60)
            if response.status_code != 200:
                logger.error("NVIDIA API returned HTTP status %d: %s", response.status_code, response.text)
                yield f"[Erro HTTP {response.status_code} na NVIDIA API]"
                return

            for line in response.iter_lines():
                if line:
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data: "):
                        data_str = decoded[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            choices = chunk.get("choices", [])
                            if choices:
                                content = choices[0].get("delta", {}).get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            pass

        except Exception as e:
            logger.error("Connection error while streaming from NVIDIA. Error: %s", str(e), exc_info=True)
            yield f"[Erro de Conexão com NVIDIA: {str(e)}]"

    def build_format_prompts(
        self, 
        raw_text: str, 
        joint_type: str, 
        rag_context: Optional[str] = None, 
        user_preferences: Optional[str] = None
    ) -> tuple[str, str]:
        """Constructs system and user prompts for intelligently expanding doctor annotations.

        The Carestream workflow is:
        1. The editor already has a normal report template pre-loaded.
        2. The doctor writes brief annotations/alterations in the ANÁLISE section.
        3. The AI must expand those brief notes into formal radiological sentences.
        4. The AI must generate the IMPRESSÃO/CONCLUSÃO section.
        5. The AI must NOT rewrite or destroy the existing template structure.
        6. Lines that already describe normal findings must be preserved exactly as-is.

        Args:
            raw_text (str): The full text captured from the editor (template + annotations).
            joint_type (str): The joint type (e.g. joelho, ombro).
            rag_context (Optional[str]): Similar report examples from the vector store.
            user_preferences (Optional[str]): Doctor vocabulary preferences.

        Returns:
            tuple[str, str]: (system_prompt, user_prompt)
        """
        system_prompt = (
            "Você é um revisor de laudos de ressonância magnética. Faça APENAS o seguinte:\n\n"
            "1. IDENTIFIQUE as linhas com anotações do médico (achados patológicos, escritos pelo médico).\n"
            "2. CORRIJA erros de digitação/português nessas linhas (ex: 'edemas ubcondral' → 'edema subcondral', 'predominandoo' → 'predominando', 'derram articular' → 'derrame articular').\n"
            "3. PRESERVAÇÃO PARCIAL: quando um achado patológico afeta APENAS UMA PARTE de uma estrutura, NÃO remova a linha normal inteira.\n"
            "   Em vez disso, ajuste a linha para refletir que a parte não afetada está normal:\n"
            "   - Se a lesão for do MENISCO MEDIAL apenas, troque 'Meniscos de morfologia e sinal normais.' por 'Menisco lateral de morfologia e sinal normais.'\n"
            "   - Se a lesão for do MENISCO LATERAL apenas, troque por 'Menisco medial de morfologia e sinal normais.'\n"
            "   - Se houver erosões/irregularidades condrais apenas em um compartimento, troque 'Superfícies condrais ... regulares, sem erosões profundas.' por 'Demais superfícies condrais regulares, sem erosões profundas.'\n"
            "   - Se houver derrame articular, remova 'Não há derrame articular significativo.'\n"
            "4. PRESERVE todas as demais linhas do template exatamente como estão.\n"
            "5. A IMPRESSÃO deve conter APENAS os achados patológicos (listados separadamente).\n"
            "   Se a IMPRESSÃO original diz 'sem alterações significativas', substitua pelos achados reais.\n"
            "   Formato: cada achado em uma nova linha, com hífen no início.\n"
            "6. NÃO adicione palavras novas. NÃO invente descrições. NÃO adicione 'leve', 'discreta', etc.\n"
            "7. NÃO inclua saudações, explicações ou metadados — apenas o laudo completo revisado.\n"
        )

        if user_preferences:
            system_prompt += f"\n[PREFERÊNCIAS DE VOCABULÁRIO DO MÉDICO]:\n{user_preferences}\n"

        user_prompt = f"Exame: RM de {joint_type.upper()}\n"
        if rag_context:
            user_prompt += f"\n[EXEMPLOS DE LAUDOS SIMILARES PARA REFERÊNCIA DE ESTILO]:\n{rag_context}\n"

        user_prompt += (
            f"\n[LAUDO COMPLETO DO EDITOR (TEMPLATE + ANOTAÇÕES DO MÉDICO)]:\n"
            f"{raw_text}\n\n"
            "Retorne o laudo completo com as anotações expandidas e a IMPRESSÃO/CONCLUSÃO gerada. "
            "Preserve toda a estrutura original."
        )

        return system_prompt, user_prompt



    def build_review_prompts(
        self, 
        raw_text: str, 
        user_preferences: Optional[str] = None
    ) -> tuple[str, str]:
        """Constructs prompts for refining and improving an already written report (Modo Aperfeiçoamento).

        Args:
            raw_text (str): The existing complete report.
            user_preferences (Optional[str]): Doctor preferences.

        Returns:
            tuple[str, str]: (system_prompt, user_prompt)
        """
        system_prompt = (
            "Você é um revisor de laudos radiológicos de Ressonância Magnética.\n"
            "Sua tarefa é aperfeiçoar o texto do laudo:\n"
            "1. Corrija erros gramaticais, de ortografia e digitação.\n"
            "2. Padronize a nomenclatura médica para termos consagrados.\n"
            "3. Melhore a fluidez e coesão textual.\n"
            "4. NÃO altere o significado clínico nem os achados descritos.\n"
            "5. NÃO remova seções pré-existentes.\n"
        )
        
        if user_preferences:
            system_prompt += f"\n[PREFERÊNCIAS DE ESTILO DO MÉDICO]:\n{user_preferences}\n"

        user_prompt = (
            f"[LAUDO ORIGINAL]:\n{raw_text}\n\n"
            "Forneça o laudo revisado e aperfeiçoado, mantendo a estrutura exata, sem adicionar explicações adicionais."
        )
        return system_prompt, user_prompt

    def build_conclusion_prompts(self, raw_text: str) -> tuple[str, str]:
        """Constructs prompts to extract a clean, numbered list conclusion from a report description.

        Args:
            raw_text (str): The report description or findings.

        Returns:
            tuple[str, str]: (system_prompt, user_prompt)
        """
        system_prompt = (
            "Você é um assistente de radiologia. Sua tarefa é ler a descrição dos achados de um exame de Ressonância Magnética "
            "e gerar uma 'CONCLUSÃO' diagnóstica condensada.\n"
            "Regras:\n"
            "1. A conclusão deve ser uma lista numerada contendo apenas os achados significativos/patológicos descritos no texto.\n"
            "2. Não invente diagnósticos que não estejam explicitados ou diretamente sustentados pelos achados descritos.\n"
            "3. Se o exame for inteiramente normal, retorne '1. Exame dentro dos limites da normalidade.'\n"
            "4. Seja conciso e direto. Não adicione observações gerais ou recomendações de conduta médica.\n"
        )
        user_prompt = (
            f"[ACHADOS DO LAUDO]:\n{raw_text}\n\n"
            "Gere apenas a conclusão numerada:"
        )
        return system_prompt, user_prompt
