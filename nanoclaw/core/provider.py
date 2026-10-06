import logging
import os
import re
from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel
from dotenv import load_dotenv
'''
多模型适配(Factory)
'''
load_dotenv()

logger = logging.getLogger(__name__)

# 推理模型（o1/o3/o4、非 chat 的 gpt-5）只接受 temperature=1。
# 也覆盖 "openai/o3-mini" 这类带供应商前缀的型号。
_TEMPERATURE_LOCKED_MODEL = re.compile(r"^(o1|o3|o4)([\-._].*)?$")

# 运行中发现「只允许 temperature=1」后记住型号，避免后续请求再踩同一错误。
_FORCED_TEMPERATURE: dict[str, float] = {}

# 各大厂商官方的 OpenAI 兼容接口地址 (当用户未配置 BASE_URL 时作为兜底)
COMPATIBLE_BASE_URLS = {
    "aliyun": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "z.ai": "https://open.bigmodel.cn/api/paas/v4",
    "tencent": "https://api.hunyuan.cloud.tencent.com/v1"
}

def _model_leaf(model_name: str) -> str:
    return (model_name or "").strip().lower().split("/")[-1]


def model_locks_temperature_to_one(model_name: str) -> bool:
    """这些型号的接口拒绝 temperature!=1。"""
    name = _model_leaf(model_name)
    if _TEMPERATURE_LOCKED_MODEL.match(name):
        return True
    return name.startswith("gpt-5") and "chat" not in name


def temperature_required_by_error(exc: BaseException) -> float | None:
    """接口明确要求 temperature 只能为 1 时返回 1.0，否则返回 None。"""
    parts = [str(exc)]
    for attr in ("body", "message"):
        value = getattr(exc, attr, None)
        if value:
            parts.append(str(value))
    text = "\n".join(parts).lower()
    if "temperature" not in text:
        return None
    if any(phrase in text for phrase in (
        "only 1 is allowed",
        "only the default (1)",
        "supported values are: 1",
        "does not support 0 with this model",
    )):
        return 1.0
    return None


def model_temperature_was_forced(model_name: str) -> bool:
    return model_name in _FORCED_TEMPERATURE or model_locks_temperature_to_one(model_name)


def _effective_temperature(model_name: str, temperature: float) -> float:
    if model_name in _FORCED_TEMPERATURE:
        return _FORCED_TEMPERATURE[model_name]
    if model_locks_temperature_to_one(model_name):
        return 1.0
    return temperature


_ADAPTIVE_OPENAI_CLS = None


def _adaptive_openai_cls(chat_openai_cls):
    global _ADAPTIVE_OPENAI_CLS
    if _ADAPTIVE_OPENAI_CLS is not None:
        return _ADAPTIVE_OPENAI_CLS

    class TemperatureAdaptiveChatOpenAI(chat_openai_cls):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            try:
                return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as exc:
                if not self._lock_temperature_from_error(exc, kwargs):
                    raise
                return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            try:
                return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as exc:
                if not self._lock_temperature_from_error(exc, kwargs):
                    raise
                return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

        def _lock_temperature_from_error(self, exc: BaseException, kwargs: dict) -> bool:
            required = temperature_required_by_error(exc)
            if required is None:
                return False
            current = kwargs.get("temperature", self.temperature)
            if current == required:
                return False
            self.temperature = required
            kwargs["temperature"] = required
            _FORCED_TEMPERATURE[self.model_name] = required
            logger.warning("模型 %s 只允许 temperature=1，已自动改用 1 并重试", self.model_name)
            return True

    _ADAPTIVE_OPENAI_CLS = TemperatureAdaptiveChatOpenAI
    return _ADAPTIVE_OPENAI_CLS


def get_provider(
    provider_name: str = "openai", 
    model_name: str = "gpt-4o-mini", 
    temperature: float = 0.0,
    base_url: str | None = None,  # 允许外部传入
    api_key: str | None = None,   # 允许外部传入
    **kwargs: Any
) -> BaseChatModel:
    """
    模型适配器工厂
    """
    provider_name = provider_name.lower()
    
    if provider_name in ["openai", "aliyun", "dashscope", "z.ai", "tencent", "other"]:
        from langchain_openai import ChatOpenAI
        
        current_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not current_api_key:
            raise ValueError(f"未找到 API Key！请确保 .env 中配置了 OPENAI_API_KEY")
            

        final_base_url = base_url or os.environ.get("OPENAI_API_BASE")
        if not final_base_url:
            final_base_url = COMPATIBLE_BASE_URLS.get(provider_name) 

        return _adaptive_openai_cls(ChatOpenAI)(
            model=model_name, 
            temperature=_effective_temperature(model_name, temperature),
            api_key=current_api_key,
            base_url=final_base_url,
            **kwargs
        )

    elif provider_name == "anthropic":
        from langchain_anthropic import ChatAnthropic
        
        current_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not current_api_key:
            raise ValueError("未找到 ANTHROPIC_API_KEY 环境变量！")
            
        final_base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL")

        return ChatAnthropic(
            model_name=model_name, 
            temperature=temperature, 
            api_key=current_api_key,
            base_url=final_base_url,
            **kwargs
        )
        
    elif provider_name == "ollama":
        from langchain_community.chat_models import ChatOllama
        
        final_base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        
        return ChatOllama(
            model=model_name, 
            temperature=temperature, 
            base_url=final_base_url,
            **kwargs
        )
        
    else:
        raise ValueError(f"不支持的模型提供商: {provider_name}")

# 测试模型调用    
# LLM = get_provider(provider_name='aliyun', model_name='glm-5')
# res = LLM.invoke('你是谁')
# print(type(res))
# print(res)


