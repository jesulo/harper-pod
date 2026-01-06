"""
Harper Settings - Configuration management with environment variables
"""
from typing import Optional
import os
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class STTSettings(BaseModel):
    """Speech-to-Text configuration"""
    model: str = Field(default="whisper-local", description="STT model to use")
    groq_api_key: Optional[str] = Field(default=None, description="Groq API key for STT")
    device: Optional[str] = Field(default=None, description="Device for local STT models")


class TTSSettings(BaseModel):
    """Text-to-Speech configuration"""
    model: str = Field(default="chatterbox", description="TTS model to use")
    together_api_key: Optional[str] = Field(default=None, description="Together AI API key")
    voice: str = Field(default="tara", description="Default voice for TTS")
    sample_rate: int = Field(default=24000, description="Audio sample rate")
    
    # Resemble AI settings (for 'resemble' model)
    resemble_api_key: Optional[str] = Field(default=None, description="Resemble AI API key")
    resemble_voice_uuid: Optional[str] = Field(default=None, description="Resemble voice UUID")
    resemble_project_uuid: Optional[str] = Field(default=None, description="Resemble project UUID")
    resemble_precision: str = Field(default="PCM_16", description="Resemble audio precision")
    resemble_use_hd: bool = Field(default=False, description="Resemble HD mode")


class LLMSettings(BaseModel):
    """Large Language Model configuration"""
    model: str = Field(default="api", description="LLM model type (api/local)")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    model_name: str = Field(default="gpt-4o-mini", description="Specific model name")


class ServerSettings(BaseModel):
    """Server configuration"""
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=5000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")


class HarperSettings(BaseSettings):
    """Main Harper configuration"""
    
    # Model selection (top-level like realtime-phone-agents-course)
    stt_model: str = Field(default="whisper-local", description="STT model (whisper-local, groq)")
    tts_model: str = Field(default="chatterbox", description="TTS model (chatterbox, resemble, together)")
    
    # OpenAI API Key (for LLM)
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    
    # Component settings
    stt: STTSettings = Field(default_factory=STTSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    
    # Global settings
    language: str = Field(default="en", description="Default language")
    input_sample_rate: int = Field(default=24000, description="Input audio sample rate")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        case_sensitive = False
        # Allow missing .env file - use defaults
        env_ignore_empty = True
        extra = "allow"


# Global settings instance
settings = HarperSettings()


def get_env_or_constant(env_key: str, constant_name: str, default: str = "") -> str:
    """
    Get value from environment variable or fall back to constants.py
    
    Args:
        env_key: Environment variable name
        constant_name: Constant name in constants.py
        default: Default value if neither found
        
    Returns:
        Configuration value
    """
    # First try environment variable
    env_value = os.getenv(env_key)
    if env_value:
        return env_value
    
    # Fall back to constants.py
    try:
        import utils.constants as constants
        if hasattr(constants, constant_name):
            return getattr(constants, constant_name)
    except ImportError:
        pass
    
    return default