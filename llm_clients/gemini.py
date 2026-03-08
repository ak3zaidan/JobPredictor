from llm_clients.helper import build_content_with_images_gemini
from typing import Optional, Dict, List, Iterator, Any, Generator
from keys import GEMINI_API_KEY
import requests
import json


class GeminiClient:
    """Google Gemini API client for making LLM calls."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or GEMINI_API_KEY
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.headers = {
            "Content-Type": "application/json"
        }
    
    def simple_chat(
        self,
        model: str,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        message_history: Optional[List[Dict[str, str]]] = None,
        images: Optional[List[str]] = None
    ) -> str:
        """
        Make a non-streaming chat completion request to Gemini.
        """
        # Build contents array
        contents = []
        
        # Add message history if provided
        if message_history:
            for msg in message_history:
                role = msg.get("role", "user")
                content_text = msg.get("content", "")
                # Skip system messages in history - they'll be handled via systemInstruction
                if role == "system":
                    continue
                elif role == "user":
                    contents.append({
                        "role": "user",
                        "parts": [{"text": content_text}]
                    })
                elif role == "assistant":
                    contents.append({
                        "role": "model",
                        "parts": [{"text": content_text}]
                    })
        
        # Build current user message with images if provided
        if images:
            parts = build_content_with_images_gemini(prompt, images)
            contents.append({
                "role": "user",
                "parts": parts
            })
        else:
            contents.append({
                "role": "user",
                "parts": [{"text": prompt}]
            })
        
        # Build generation config
        generation_config = {
            "temperature": temperature
        }
        
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens
        
        # Build request payload
        payload = {
            "contents": contents,
            "generationConfig": generation_config
        }
        
        # Add system instruction if provided
        if system_message:
            payload["systemInstruction"] = {
                "parts": [{"text": system_message}]
            }
        
        # Make request
        url = f"{self.base_url}/models/{model}:generateContent"
        params = {"key": self.api_key}
        
        response = requests.post(
            url,
            headers=self.headers,
            json=payload,
            params=params,
            timeout=60
        )
        response.raise_for_status()
        response_data = response.json()
        
        if "error" in response_data:
            raise Exception(f"Error: {response_data['error']}")
        
        # Extract text from response
        if "candidates" in response_data and len(response_data["candidates"]) > 0:
            candidate = response_data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                text_parts = [part.get("text", "") for part in parts if "text" in part]
                return "".join(text_parts)
        
        raise Exception("No content in response")
    
    def stream_chat(
        self,
        model: str,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        message_history: Optional[List[Dict[str, str]]] = None,
        images: Optional[List[str]] = None
    ) -> Iterator[str]:
        """
        Stream chat completion responses from Gemini.
        Yields text chunks as they arrive.
        """
        # Build contents array (same as simple_chat)
        contents = []
        
        if message_history:
            for msg in message_history:
                role = msg.get("role", "user")
                content_text = msg.get("content", "")
                # Skip system messages in history - they'll be handled via systemInstruction
                if role == "system":
                    continue
                elif role == "user":
                    contents.append({
                        "role": "user",
                        "parts": [{"text": content_text}]
                    })
                elif role == "assistant":
                    contents.append({
                        "role": "model",
                        "parts": [{"text": content_text}]
                    })
        
        # Build current user message with images if provided
        if images:
            parts = build_content_with_images_gemini(prompt, images)
            contents.append({
                "role": "user",
                "parts": parts
            })
        else:
            contents.append({
                "role": "user",
                "parts": [{"text": prompt}]
            })
        
        # Build generation config
        generation_config = {
            "temperature": temperature
        }
        
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens
        
        # Build request payload
        payload = {
            "contents": contents,
            "generationConfig": generation_config
        }
        
        if system_message:
            payload["systemInstruction"] = {
                "parts": [{"text": system_message}]
            }
        
        # Make streaming request
        url = f"{self.base_url}/models/{model}:streamGenerateContent"
        params = {"key": self.api_key}
        
        response = requests.post(
            url,
            headers=self.headers,
            json=payload,
            params=params,
            stream=True,
            timeout=60
        )
        response.raise_for_status()
        
        # Parse streaming response
        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                # Gemini streaming responses are JSON objects, one per line
                try:
                    data = json.loads(line_text)
                    if "candidates" in data and len(data["candidates"]) > 0:
                        candidate = data["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            parts = candidate["content"]["parts"]
                            for part in parts:
                                if "text" in part:
                                    text = part["text"]
                                    if text:
                                        yield text
                except json.JSONDecodeError:
                    continue
    
    def chat_with_tools(
        self,
        model: str,
        prompt: str,
        tools: List[Dict],
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        message_history: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Chat with tool calling support for Gemini.
        
        Returns:
            Dict with either:
            - {"type": "message", "content": "response text"}
            - {"type": "tool_calls", "calls": [{"name": "...", "arguments": {...}, "id": "..."}], "text": "optional text"}
        """
        # Build contents array
        contents = []
        
        if message_history:
            for msg in message_history:
                role = msg.get("role", "user")
                content_text = msg.get("content", "")
                # Skip system messages in history - they'll be handled via systemInstruction
                if role == "system":
                    continue
                elif role == "user":
                    contents.append({
                        "role": "user",
                        "parts": [{"text": content_text}]
                    })
                elif role == "assistant":
                    contents.append({
                        "role": "model",
                        "parts": [{"text": content_text}]
                    })
        
        # Build current user message with images if provided
        if images:
            parts = build_content_with_images_gemini(prompt, images)
            contents.append({
                "role": "user",
                "parts": parts
            })
        else:
            contents.append({
                "role": "user",
                "parts": [{"text": prompt}]
            })
        
        # Convert tools to Gemini format
        # Gemini expects a single tools array with functionDeclarations
        function_declarations = []
        for tool in tools:
            if "function" in tool:
                func_def = tool["function"]
                function_declarations.append({
                    "name": func_def["name"],
                    "description": func_def["description"],
                    "parameters": func_def["parameters"]
                })
        
        gemini_tools = None
        if function_declarations:
            gemini_tools = [{
                "functionDeclarations": function_declarations
            }]
        
        # Build generation config
        generation_config = {
            "temperature": temperature
        }
        
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens
        
        # Build request payload
        payload = {
            "contents": contents,
            "generationConfig": generation_config
        }
        
        if system_message:
            payload["systemInstruction"] = {
                "parts": [{"text": system_message}]
            }
        
        if gemini_tools:
            payload["tools"] = gemini_tools
        
        # Make request
        url = f"{self.base_url}/models/{model}:generateContent"
        params = {"key": self.api_key}
        
        response = requests.post(
            url,
            headers=self.headers,
            json=payload,
            params=params,
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        
        if "error" in data:
            raise Exception(f"Error: {data['error']}")
        
        # Check for function calls in response
        tool_calls = []
        text_content = ""
        
        if "candidates" in data and len(data["candidates"]) > 0:
            candidate = data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                for part in parts:
                    if "text" in part:
                        text_content += part["text"]
                    elif "functionCall" in part:
                        fc = part["functionCall"]
                        tool_calls.append({
                            "id": fc.get("name", ""),  # Gemini doesn't provide separate IDs
                            "name": fc["name"],
                            "arguments": fc.get("args", {})
                        })
        
        if tool_calls:
            return {"type": "tool_calls", "calls": tool_calls, "text": text_content}
        
        return {"type": "message", "content": text_content}
    
    def stream_with_tools(
        self,
        model: str,
        prompt: str,
        tools: List[Dict],
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        message_history: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List[str]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream chat with tool calling support for Gemini.
        
        Yields:
            - {"type": "content", "chunk": "text chunk"} for regular content
            - {"type": "tool_calls", "calls": [...]} when tool calls are complete
            - {"type": "done"} when streaming is complete
        """
        # Build contents array
        contents = []
        
        if message_history:
            for msg in message_history:
                role = msg.get("role", "user")
                content_text = msg.get("content", "")
                # Skip system messages in history - they'll be handled via systemInstruction
                if role == "system":
                    continue
                elif role == "user":
                    contents.append({
                        "role": "user",
                        "parts": [{"text": content_text}]
                    })
                elif role == "assistant":
                    contents.append({
                        "role": "model",
                        "parts": [{"text": content_text}]
                    })
        
        # Build current user message with images if provided
        if images:
            parts = build_content_with_images_gemini(prompt, images)
            contents.append({
                "role": "user",
                "parts": parts
            })
        else:
            contents.append({
                "role": "user",
                "parts": [{"text": prompt}]
            })
        
        # Convert tools to Gemini format
        # Gemini expects a single tools array with functionDeclarations
        function_declarations = []
        for tool in tools:
            if "function" in tool:
                func_def = tool["function"]
                function_declarations.append({
                    "name": func_def["name"],
                    "description": func_def["description"],
                    "parameters": func_def["parameters"]
                })
        
        gemini_tools = None
        if function_declarations:
            gemini_tools = [{
                "functionDeclarations": function_declarations
            }]
        
        # Build generation config
        generation_config = {
            "temperature": temperature
        }
        
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens
        
        # Build request payload
        payload = {
            "contents": contents,
            "generationConfig": generation_config
        }
        
        if system_message:
            payload["systemInstruction"] = {
                "parts": [{"text": system_message}]
            }
        
        if gemini_tools:
            payload["tools"] = gemini_tools
        
        # Make streaming request
        url = f"{self.base_url}/models/{model}:streamGenerateContent"
        params = {"key": self.api_key}
        
        response = requests.post(
            url,
            headers=self.headers,
            json=payload,
            params=params,
            stream=True,
            timeout=120
        )
        response.raise_for_status()
        
        # Track tool calls and content
        tool_calls_accumulator: List[Dict[str, Any]] = []
        has_tool_calls = False
        
        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                try:
                    data = json.loads(line_text)
                    if "candidates" in data and len(data["candidates"]) > 0:
                        candidate = data["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            parts = candidate["content"]["parts"]
                            for part in parts:
                                if "text" in part:
                                    text = part["text"]
                                    if text:
                                        yield {"type": "content", "chunk": text}
                                elif "functionCall" in part:
                                    has_tool_calls = True
                                    fc = part["functionCall"]
                                    # Check if we already have this function call
                                    existing = None
                                    for tc in tool_calls_accumulator:
                                        if tc["name"] == fc["name"]:
                                            existing = tc
                                            break
                                    
                                    if existing:
                                        # Merge arguments (in case of streaming function calls)
                                        existing["arguments"].update(fc.get("args", {}))
                                    else:
                                        tool_calls_accumulator.append({
                                            "id": fc["name"],
                                            "name": fc["name"],
                                            "arguments": fc.get("args", {})
                                        })
                except json.JSONDecodeError:
                    continue
        
        # If we accumulated tool calls, yield them now
        if has_tool_calls and tool_calls_accumulator:
            yield {"type": "tool_calls", "calls": tool_calls_accumulator}
        
        yield {"type": "done"}
