try:
    from observability.observability_wrapper import (
        trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
    )
except ImportError:  # observability module not available (e.g. isolated test env)
    from contextlib import contextmanager as _obs_cm, asynccontextmanager as _obs_acm
    def trace_agent(*_a, **_kw):  # type: ignore[misc]
        def _deco(fn): return fn
        return _deco
    class _ObsHandle:
        output_summary = None
        def capture(self, *a, **kw): pass
    @_obs_acm
    async def trace_step(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    @_obs_cm
    def trace_step_sync(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    def trace_model_call(*_a, **_kw): pass  # type: ignore[misc]
    def trace_tool_call(*_a, **_kw): pass  # type: ignore[misc]

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {'check_credentials_output': True,
 'check_jailbreak': True,
 'check_output': True,
 'check_pii_input': False,
 'check_toxic_code_output': True,
 'check_toxicity': True,
 'content_safety_enabled': True,
 'content_safety_severity_threshold': 3,
 'runtime_enabled': True,
 'sanitize_pii': False}


import os
import re
import logging
import asyncio
import time
from typing import Tuple, Any, Dict, Optional

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator, ValidationError, Field
from dotenv import load_dotenv

# Observability wrappers are injected automatically by the runtime.
# Do NOT add @trace_agent or import trace_step manually.
# Use trace_step/trace_step_sync as required.

# Load environment variables from .env if present
load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AdditionAssistantAgent")

# --- Configuration Management ---

class Config:
    """Configuration loader for environment variables."""
    @staticmethod
    def get_openai_api_key() -> Optional[str]:
        return os.getenv("OPENAI_API_KEY")

    @staticmethod
    @trace_agent(agent_name='General Addition Assistant')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def get_llm_model() -> str:
        return os.getenv("OPENAI_MODEL", "gpt-4.1")

    @staticmethod
    @trace_agent(agent_name='General Addition Assistant')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def get_llm_temperature() -> float:
        try:
            return float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        except Exception:
            return 0.7

    @staticmethod
    @trace_agent(agent_name='General Addition Assistant')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def get_llm_max_tokens() -> int:
        try:
            return int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
        except Exception:
            return 2000

    @staticmethod
    def validate():
        """Optional config validation, not called at import time."""
        if not Config.get_openai_api_key():
            raise ValueError("OPENAI_API_KEY is not set in environment variables.")

# --- Input Model and Validation ---

class AdditionRequest(BaseModel):
    input_text: str = Field(..., max_length=50000)

    @field_validator("input_text")
    @classmethod
    def validate_input_text(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Input must be a string.")
        v = v.strip()
        if not v:
            raise ValueError("Input text cannot be empty.")
        if len(v) > 50000:
            raise ValueError("Input text exceeds 50,000 characters.")
        return v

# --- Logger Component ---

class Logger:
    """Logging & Monitoring Layer Component."""
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def log(self, event_type: str, message: str, context: Dict[str, Any]):
        logger.info(f"[{event_type}] {message} | Context: {context}")

# --- Error Handler Component ---

class ErrorHandler:
    """Handles errors, applies retry logic, and provides fallback responses."""
    FALLBACK_RESPONSE = "Sorry, I couldn't process your input. Please provide two numbers, and I'll add them for you!"

    def __init__(self, logger: Logger):
        self.logger = logger

    def handle_error(self, error_code: str, context: Dict[str, Any]) -> str:
        """Handles errors and returns user-facing message."""
        self.logger.log("error", f"Error occurred: {error_code}", context)
        if error_code in ("INVALID_INPUT", "MISSING_INPUT"):
            return "Please provide two valid numbers you'd like to add together."
        elif error_code == "SYSTEM_ERROR":
            return self.FALLBACK_RESPONSE
        else:
            return self.FALLBACK_RESPONSE

# --- Input Validator Component ---

class InputValidator:
    """Validates that both inputs are present and numeric."""
    def __init__(self, logger: Logger):
        self.logger = logger

    def validate(self, input_number_1: Any, input_number_2: Any) -> Tuple[float, float]:
        """Checks if both inputs are valid numbers."""
        try:
            a = float(input_number_1)
            b = float(input_number_2)
            return a, b
        except Exception as e:
            self.logger.log("validation_error", "Input validation failed", {
                "input_number_1": input_number_1,
                "input_number_2": input_number_2,
                "error": str(e)
            })
            raise ValueError("INVALID_INPUT")

# --- Addition Engine Component ---

class AdditionEngine:
    """Performs the addition operation on validated numeric inputs."""
    def add(self, a: float, b: float) -> float:
        try:
            return a + b
        except Exception as e:
            logger.error(f"Addition failed: {e}")
            raise RuntimeError("SYSTEM_ERROR")

# --- Output Formatter Component ---

class OutputFormatter:
    """Formats the final response according to output templates and user context."""
    def format_response(self, sum_: float, input_number_1: float, input_number_2: float) -> str:
        try:
            return f"The sum of {input_number_1:g} and {input_number_2:g} is {sum_:g}."
        except Exception as e:
            logger.error(f"Output formatting failed: {e}")
            return ErrorHandler.FALLBACK_RESPONSE

# --- LLM Response Generator Component ---

class LLMResponseGenerator:
    """Generates conversational responses using the LLM based on operation results or errors."""
    SYSTEM_PROMPT = (
        "You are a helpful assistant whose role is to help users add two numbers. "
        "When a user provides two numbers, validate that both are numeric values. "
        "If both inputs are valid, perform the addition and return the result in a friendly, conversational tone. "
        "If the input is invalid or missing, politely prompt the user to provide two valid numbers. "
        "Never perform operations other than addition. If you cannot process the request, respond with a clear and supportive fallback message."
    )
    FALLBACK_RESPONSE = ErrorHandler.FALLBACK_RESPONSE

    def __init__(self):
        self.model = Config.get_llm_model()
        self.temperature = Config.get_llm_temperature()
        self.max_tokens = Config.get_llm_max_tokens()

    @trace_agent(agent_name='General Addition Assistant')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def get_llm_client(self):
        import openai
        api_key = Config.get_openai_api_key()
        if not api_key:
            raise ValueError("OPENAI_API_KEY not configured")
        return openai.AsyncOpenAI(api_key=api_key)

    @trace_agent(agent_name='General Addition Assistant')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def generate_response(self, context: Dict[str, Any]) -> str:
        """Generates a conversational response using the LLM."""
        import openai
        max_attempts = 3
        delay = 0.7
        last_exception = None
        prompt_user = context.get("user_prompt", "")
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt_user}
        ]
        for attempt in range(1, max_attempts + 1):
            async with trace_step(
                "generate_llm_response", step_type="llm_call",
                decision_summary=f"Call LLM to generate conversational response (attempt {attempt})",
                output_fn=lambda r: f"length={len(r) if r else 0}",
            ) as step:
                _t0 = time.time()
                try:
                    client = self.get_llm_client()
                    response = await client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    content = response.choices[0].message.content.strip()
                    try:
                        trace_model_call(
                            provider="openai",
                            model_name=self.model,
                            prompt_tokens=getattr(response.usage, "prompt_tokens", 0),
                            completion_tokens=getattr(response.usage, "completion_tokens", 0),
                            latency_ms=int((time.time() - _t0) * 1000),
                            response_summary=content[:200] if content else ""
                        )
                    except Exception:
                        pass
                    step.capture(content)
                    return content
                except Exception as e:
                    last_exception = e
                    logger.warning(f"LLM API call failed (attempt {attempt}): {e}")
                    await asyncio.sleep(delay * attempt)
        logger.error(f"LLM API failed after {max_attempts} attempts: {last_exception}")
        return self.FALLBACK_RESPONSE

# --- User Input Handler Component ---

class UserInputHandler:
    """Receives and parses user input, initiates processing pipeline."""
    def __init__(self, input_validator: InputValidator, logger: Logger):
        self.input_validator = input_validator
        self.logger = logger

    @with_content_safety(config=GUARDRAILS_CONFIG)
    def receive_input(self, input_text: str) -> Tuple[Any, Any]:
        """Extracts two numbers from user input text."""
        # Try to extract numbers using regex
        numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", input_text)
        if len(numbers) < 2:
            self.logger.log("parse_error", "Failed to extract two numbers", {"input_text": input_text})
            raise ValueError("MISSING_INPUT")
        try:
            a, b = numbers[0], numbers[1]
            return a, b
        except Exception as e:
            self.logger.log("parse_error", "Error extracting numbers", {"input_text": input_text, "error": str(e)})
            raise ValueError("INVALID_INPUT")

# --- Main Agent Class ---

class AdditionAssistantAgent:
    """Main orchestrator for the General Addition Assistant."""

    def __init__(self):
        self.logger = Logger()
        self.error_handler = ErrorHandler(self.logger)
        self.input_validator = InputValidator(self.logger)
        self.addition_engine = AdditionEngine()
        self.output_formatter = OutputFormatter()
        self.llm_response_generator = LLMResponseGenerator()
        self.user_input_handler = UserInputHandler(self.input_validator, self.logger)

    @trace_agent(agent_name='General Addition Assistant')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_user_input(self, input_text: str) -> str:
        """Main entry point: orchestrates input parsing, validation, addition, and response generation."""
        async with trace_step(
            "parse_numbers", step_type="parse",
            decision_summary="Extract two numbers from user input",
            output_fn=lambda r: f"numbers={r}" if isinstance(r, tuple) else str(r),
        ) as step:
            try:
                a_raw, b_raw = self.parse_numbers(input_text)
                step.capture((a_raw, b_raw))
            except Exception as e:
                step.capture(str(e))
                return await self.handle_error(str(e), {"input_text": input_text})

        async with trace_step(
            "validate_numbers", step_type="process",
            decision_summary="Validate both inputs are numbers",
            output_fn=lambda r: f"validated={r}" if isinstance(r, tuple) else str(r),
        ) as step:
            try:
                a, b = self.validate_numbers(a_raw, b_raw)
                step.capture((a, b))
            except Exception as e:
                step.capture(str(e))
                return await self.handle_error(str(e), {"input_number_1": a_raw, "input_number_2": b_raw})

        async with trace_step(
            "add_numbers", step_type="process",
            decision_summary="Perform addition",
            output_fn=lambda r: f"sum={r}",
        ) as step:
            try:
                sum_ = self.add_numbers(a, b)
                step.capture(sum_)
            except Exception as e:
                step.capture(str(e))
                return await self.handle_error(str(e), {"input_number_1": a, "input_number_2": b})

        async with trace_step(
            "format_output", step_type="format",
            decision_summary="Format output message",
            output_fn=lambda r: f"output={r}",
        ) as step:
            try:
                output = self.format_output(sum_, a, b)
                step.capture(output)
            except Exception as e:
                step.capture(str(e))
                return await self.handle_error(str(e), {"sum": sum_, "input_number_1": a, "input_number_2": b})

        async with trace_step(
            "generate_llm_response", step_type="llm_call",
            decision_summary="Generate conversational response using LLM",
            output_fn=lambda r: f"length={len(r) if r else 0}",
        ) as step:
            try:
                context = {
                    "user_prompt": output,
                    "input_number_1": a,
                    "input_number_2": b,
                    "sum": sum_
                }
                llm_response = await self.generate_llm_response(context)
                step.capture(llm_response)
                return llm_response
            except Exception as e:
                step.capture(str(e))
                return await self.handle_error(str(e), {"output": output, "context": context})

    @trace_agent(agent_name='General Addition Assistant')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def parse_numbers(self, input_text: str) -> Tuple[Any, Any]:
        """Extracts two numbers from user input text."""
        return self.user_input_handler.receive_input(input_text)

    def validate_numbers(self, input_number_1: Any, input_number_2: Any) -> Tuple[float, float]:
        """Checks if both inputs are valid numbers."""
        return self.input_validator.validate(input_number_1, input_number_2)

    def add_numbers(self, a: float, b: float) -> float:
        """Performs addition on two validated numbers."""
        return self.addition_engine.add(a, b)

    def format_output(self, sum_: float, input_number_1: float, input_number_2: float) -> str:
        """Formats the output message for the user."""
        return self.output_formatter.format_response(sum_, input_number_1, input_number_2)

    @trace_agent(agent_name='General Addition Assistant')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def generate_llm_response(self, context: Dict[str, Any]) -> str:
        """Generates a conversational response using the LLM."""
        return await self.llm_response_generator.generate_response(context)

    async def handle_error(self, error_code: str, context: Dict[str, Any]) -> str:
        """Handles errors, applies retry logic, and returns fallback or escalation responses."""
        # For LLM error, try to generate fallback LLM response, else static fallback
        if error_code == "SYSTEM_ERROR":
            try:
                context["user_prompt"] = ErrorHandler.FALLBACK_RESPONSE
                return await self.generate_llm_response(context)
            except Exception:
                pass
        return self.error_handler.handle_error(error_code, context)

# --- FastAPI App and Endpoints ---

app = FastAPI(
    title="General Addition Assistant",
    description="A friendly assistant that adds two numbers and responds conversationally.",
    version="1.0.0"
)

# CORS (allow all origins for demo/public endpoint)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = AdditionAssistantAgent()

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Pydantic validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error_type": "VALIDATION_ERROR",
            "message": "Invalid input. Please check your request format.",
            "tips": [
                "Ensure your JSON is well-formed.",
                "Check for missing or extra commas, brackets, or quotes.",
                "Input text must be a non-empty string up to 50,000 characters."
            ]
        }
    )

@app.exception_handler(HTTPException)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTPException: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_type": "HTTP_ERROR",
            "message": exc.detail,
            "tips": [
                "Check your request and try again."
            ]
        }
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error_type": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred. Please try again later.",
            "tips": [
                "If the problem persists, contact support.",
                "Ensure your input is valid and try again."
            ]
        }
    )

@app.post("/add", response_model=None)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def add_numbers_endpoint(request: Request):
    """
    Endpoint to add two numbers from user input.
    """
    try:
        data = await request.json()
    except Exception as e:
        logger.warning(f"Malformed JSON: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error_type": "MALFORMED_JSON",
                "message": "Malformed JSON in request body.",
                "tips": [
                    "Ensure your JSON is well-formed.",
                    "Check for missing or extra commas, brackets, or quotes."
                ]
            }
        )
    try:
        addition_request = AdditionRequest(**data)
    except ValidationError as ve:
        logger.warning(f"Input validation error: {ve}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "error_type": "VALIDATION_ERROR",
                "message": "Invalid input. Please check your request format.",
                "tips": [
                    "Input text must be a non-empty string up to 50,000 characters."
                ]
            }
        )
    try:
        response_text = await agent.process_user_input(addition_request.input_text)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "response": response_text
            }
        )
    except Exception as e:
        logger.error(f"Error in /add endpoint: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error_type": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred. Please try again later.",
                "tips": [
                    "If the problem persists, contact support.",
                    "Ensure your input is valid and try again."
                ]
            }
        )

@app.get("/")
async def root():
    return {
        "success": True,
        "message": "Welcome to the General Addition Assistant! POST to /add with {'input_text': 'Add 3 and 4'}."
    }

# --- Main Execution Block ---



async def _run_with_eval_service():
    """Entrypoint: initialises observability then runs the agent."""
    import logging as _obs_log
    _obs_logger = _obs_log.getLogger(__name__)
    # ── 1. Observability DB schema ─────────────────────────────────────
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401 – register ORM models
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
    except Exception as _e:
        _obs_logger.warning('Observability DB init skipped: %s', _e)
    # ── 2. OpenTelemetry tracer ────────────────────────────────────────
    try:
        from observability.instrumentation import initialize_tracer
        initialize_tracer()
    except Exception as _e:
        _obs_logger.warning('Tracer init skipped: %s', _e)
    # ── 3. Evaluation background worker ───────────────────────────────
    _stop_eval = None
    try:
        from observability.evaluation_background_service import (
            start_evaluation_worker as _start_eval,
            stop_evaluation_worker as _stop_eval_fn,
        )
        await _start_eval()
        _stop_eval = _stop_eval_fn
    except Exception as _e:
        _obs_logger.warning('Evaluation worker start skipped: %s', _e)
    # ── 4. Run the agent ───────────────────────────────────────────────
    try:
        import uvicorn
        logger.info("Starting General Addition Assistant on http://0.0.0.0:8000")
        uvicorn.run("agent:app", host="0.0.0.0", port=8000, reload=False)
        pass  # TODO: run your agent here
    finally:
        if _stop_eval is not None:
            try:
                await _stop_eval()
            except Exception:
                pass


if __name__ == "__main__":
    import asyncio as _asyncio
    _asyncio.run(_run_with_eval_service())