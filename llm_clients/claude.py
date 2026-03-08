from llm_clients.helper import build_content_with_images_claude
from typing import Optional, Dict, List, Iterator, Any, Generator
from keys import CLAUDE_API_KEY
import requests
import json


class ClaudeClient:
    """Claude API client for making LLM calls."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or CLAUDE_API_KEY
        self.base_url = "https://api.anthropic.com/v1"
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
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
        messages = []
        
        # Add message history if provided
        if message_history:
            messages.extend(message_history)
        
        # Build user message with images if provided
        if images:
            content = build_content_with_images_claude(prompt, images)
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens or 8096,
            "temperature": temperature
        }
        
        if system_message:
            payload["system"] = system_message
        
        response = requests.post(
            f"{self.base_url}/messages",
            headers=self.headers,
            json=payload,
            timeout=60
        )

        response.raise_for_status()
        response_data = response.json()
        
        if "content" in response_data and response_data["content"]:
            text = response_data["content"][0]["text"]
            return text
        else:
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
        Stream chat completion responses from Claude.
        Yields text chunks as they arrive.
        """
        messages = []
        
        # Add message history if provided
        if message_history:
            messages.extend(message_history)
        
        # Build user message with images if provided
        if images:
            content = build_content_with_images_claude(prompt, images)
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens or 8096,
            "temperature": temperature,
            "stream": True
        }
        
        if system_message:
            payload["system"] = system_message
        
        # Make streaming request
        response = requests.post(
            f"{self.base_url}/messages",
            headers=self.headers,
            json=payload,
            stream=True,
            timeout=60
        )
        response.raise_for_status()
        
        # Parse SSE stream
        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                if line_text.startswith('data: '):
                    data_str = line_text[6:]  # Remove 'data: ' prefix
                    try:
                        data = json.loads(data_str)
                        if data.get('type') == 'content_block_delta':
                            delta = data.get('delta', {})
                            text = delta.get('text', '')
                            if text:
                                yield text
                        elif data.get('type') == 'message_stop':
                            break
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
        Chat with tool calling support for Claude.
        
        Returns:
            Dict with either:
            - {"type": "message", "content": "response text"}
            - {"type": "tool_calls", "calls": [{"name": "...", "arguments": {...}, "id": "..."}], "text": "optional text"}
        """
        messages = []
        
        if message_history:
            messages.extend(message_history)
        
        if images:
            content = build_content_with_images_claude(prompt, images)
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "max_tokens": max_tokens or 8096,
            "temperature": temperature
        }
        
        if system_message:
            payload["system"] = system_message
        
        response = requests.post(
            f"{self.base_url}/messages",
            headers=self.headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        
        # Check for tool use in response
        tool_calls = []
        text_content = ""
        
        for block in data.get("content", []):
            if block["type"] == "tool_use":
                tool_calls.append({
                    "id": block["id"],
                    "name": block["name"],
                    "arguments": block["input"]
                })
            elif block["type"] == "text":
                text_content += block["text"]
        
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
        Stream chat with tool calling support for Claude.
        
        Yields:
            - {"type": "content", "chunk": "text chunk"} for regular content
            - {"type": "tool_calls", "calls": [...]} when tool calls are complete
            - {"type": "done"} when streaming is complete
        """
        messages = []
        
        if message_history:
            messages.extend(message_history)
        
        if images:
            content = build_content_with_images_claude(prompt, images)
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "max_tokens": max_tokens or 8096,
            "temperature": temperature,
            "stream": True
        }
        
        if system_message:
            payload["system"] = system_message
        
        response = requests.post(
            f"{self.base_url}/messages",
            headers=self.headers,
            json=payload,
            stream=True,
            timeout=120
        )
        response.raise_for_status()
        
        # Track tool use blocks
        tool_calls: Dict[int, Dict[str, Any]] = {}
        current_block_index = -1
        current_block_type = None
        
        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                if line_text.startswith('data: '):
                    data_str = line_text[6:]
                    try:
                        data = json.loads(data_str)
                        event_type = data.get('type')
                        
                        if event_type == 'content_block_start':
                            current_block_index = data.get('index', 0)
                            block = data.get('content_block', {})
                            current_block_type = block.get('type')
                            
                            if current_block_type == 'tool_use':
                                tool_calls[current_block_index] = {
                                    "id": block.get('id', ''),
                                    "name": block.get('name', ''),
                                    "arguments": ""
                                }
                        
                        elif event_type == 'content_block_delta':
                            delta = data.get('delta', {})
                            delta_type = delta.get('type')
                            
                            if delta_type == 'text_delta':
                                text = delta.get('text', '')
                                if text:
                                    yield {"type": "content", "chunk": text}
                            
                            elif delta_type == 'input_json_delta':
                                # Accumulate tool input JSON
                                partial_json = delta.get('partial_json', '')
                                if current_block_index in tool_calls:
                                    tool_calls[current_block_index]['arguments'] += partial_json
                        
                        elif event_type == 'message_stop':
                            break
                            
                    except json.JSONDecodeError:
                        continue
        
        # If we have tool calls, yield them
        if tool_calls:
            calls = []
            for idx in sorted(tool_calls.keys()):
                tc = tool_calls[idx]
                try:
                    args = json.loads(tc['arguments']) if tc['arguments'] else {}
                except json.JSONDecodeError:
                    args = {}
                calls.append({
                    "id": tc['id'],
                    "name": tc['name'],
                    "arguments": args
                })
            yield {"type": "tool_calls", "calls": calls}
        
        yield {"type": "done"}
