from llm_clients.helper import build_content_with_images_openai_grok
from typing import Optional, Dict, List, Iterator, Any, Generator
from keys import GROK_API_KEY
import requests
import json


class GrokClient:
    """Grok API client for making LLM calls."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or GROK_API_KEY
        self.base_url = "https://api.x.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
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
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Add message history if provided
        if message_history:
            messages.extend(message_history)
        
        # Build user message with images if provided
        if images:
            content = build_content_with_images_openai_grok(prompt, images)
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        response = response.json()
        
        if "error" in response:
            raise Exception(f"Error: {response['error']}")
        
        try:
            text = response["choices"][0]["message"]["content"]
            return text
        except Exception as e:
            raise Exception(f"Error parsing response: {str(e)}")
    
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
        Stream chat completion responses from Grok.
        Yields text chunks as they arrive.
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Add message history if provided
        if message_history:
            messages.extend(message_history)
        
        # Build user message with images if provided
        if images:
            content = build_content_with_images_openai_grok(prompt, images)
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        # Make streaming request
        response = requests.post(
            f"{self.base_url}/chat/completions",
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
                    if data_str.strip() == '[DONE]':
                        break
                    try:
                        data = json.loads(data_str)
                        if 'choices' in data and len(data['choices']) > 0:
                            delta = data['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                yield content
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
        Chat with tool calling support for Grok (OpenAI-compatible API).
        
        Returns:
            Dict with either:
            - {"type": "message", "content": "response text"}
            - {"type": "tool_calls", "calls": [{"name": "...", "arguments": {...}, "id": "..."}]}
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        if message_history:
            messages.extend(message_history)
        
        if images:
            content = build_content_with_images_openai_grok(prompt, images)
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": temperature
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        
        if "error" in data:
            raise Exception(f"Error: {data['error']}")
        
        choice = data["choices"][0]
        message = choice["message"]
        
        # Check if the model wants to call tools
        if message.get("tool_calls"):
            tool_calls = []
            for tc in message["tool_calls"]:
                tool_calls.append({
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": json.loads(tc["function"]["arguments"])
                })
            return {"type": "tool_calls", "calls": tool_calls, "raw_message": message}
        
        return {"type": "message", "content": message.get("content", "")}
    
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
        Stream chat with tool calling support for Grok.
        
        Yields:
            - {"type": "content", "chunk": "text chunk"} for regular content
            - {"type": "tool_calls", "calls": [...]} when tool calls are complete
            - {"type": "done"} when streaming is complete
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        if message_history:
            messages.extend(message_history)
        
        if images:
            content = build_content_with_images_openai_grok(prompt, images)
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": temperature,
            "stream": True
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            stream=True,
            timeout=120
        )
        response.raise_for_status()
        
        # Accumulate tool calls as they stream
        tool_calls_accumulator: Dict[int, Dict[str, Any]] = {}
        has_tool_calls = False
        
        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                if line_text.startswith('data: '):
                    data_str = line_text[6:]
                    if data_str.strip() == '[DONE]':
                        break
                    try:
                        data = json.loads(data_str)
                        if 'choices' in data and len(data['choices']) > 0:
                            delta = data['choices'][0].get('delta', {})
                            
                            # Handle regular content
                            content = delta.get('content', '')
                            if content:
                                yield {"type": "content", "chunk": content}
                            
                            # Handle tool calls (accumulate them)
                            if delta.get('tool_calls'):
                                has_tool_calls = True
                                for tc in delta['tool_calls']:
                                    idx = tc.get('index', 0)
                                    if idx not in tool_calls_accumulator:
                                        tool_calls_accumulator[idx] = {
                                            "id": tc.get('id', ''),
                                            "name": "",
                                            "arguments": ""
                                        }
                                    
                                    if tc.get('id'):
                                        tool_calls_accumulator[idx]['id'] = tc['id']
                                    if tc.get('function', {}).get('name'):
                                        tool_calls_accumulator[idx]['name'] = tc['function']['name']
                                    if tc.get('function', {}).get('arguments'):
                                        tool_calls_accumulator[idx]['arguments'] += tc['function']['arguments']
                    except json.JSONDecodeError:
                        continue
        
        # If we accumulated tool calls, yield them now
        if has_tool_calls and tool_calls_accumulator:
            calls = []
            for idx in sorted(tool_calls_accumulator.keys()):
                tc = tool_calls_accumulator[idx]
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
