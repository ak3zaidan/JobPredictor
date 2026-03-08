from typing import Optional, List, Dict, Any
import base64

def _detect_image_format_mime(base64_string: str) -> str:
    """Detect image format from base64 string and return MIME type (for Claude)."""
    # Remove data URL prefix if present
    if base64_string.startswith('data:image/'):
        # Extract format from data URL
        mime_type = base64_string.split(';')[0].split(':')[-1]
        return mime_type
    
    # Try to detect from base64 content (check first few bytes)
    try:
        # Remove any data URL prefix
        clean_base64 = base64_string.split(',')[-1] if ',' in base64_string else base64_string
        decoded = base64.b64decode(clean_base64[:20])
        
        # Check magic bytes
        if decoded.startswith(b'\xff\xd8\xff'):
            return 'image/jpeg'
        elif decoded.startswith(b'\x89PNG'):
            return 'image/png'
        elif decoded.startswith(b'GIF'):
            return 'image/gif'
        elif decoded.startswith(b'WEBP', 8):
            return 'image/webp'
        else:
            return 'image/png'  # Default fallback
    except:
        return 'image/png'  # Default fallback

def _detect_image_format_extension(base64_string: str) -> str:
    """Detect image format from base64 string and return file extension (for OpenAI/Grok)."""
    # Remove data URL prefix if present
    if base64_string.startswith('data:image/'):
        # Extract format from data URL
        format_part = base64_string.split(';')[0].split('/')[-1]
        return format_part
    
    # Try to detect from base64 content (check first few bytes)
    try:
        # Remove any data URL prefix
        clean_base64 = base64_string.split(',')[-1] if ',' in base64_string else base64_string
        decoded = base64.b64decode(clean_base64[:20])
        
        # Check magic bytes
        if decoded.startswith(b'\xff\xd8\xff'):
            return 'jpeg'
        elif decoded.startswith(b'\x89PNG'):
            return 'png'
        elif decoded.startswith(b'GIF'):
            return 'gif'
        elif decoded.startswith(b'WEBP', 8):
            return 'webp'
        else:
            return 'png'  # Default fallback
    except:
        return 'png'  # Default fallback
    
def build_content_with_images_claude(prompt: str, images: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Build content array with text and images for Claude format."""
    content = []
    
    # Add images if provided (Claude requires images first)
    if images:
        for base64_image in images:
            # Remove data URL prefix if present
            clean_base64 = base64_image.split(',')[-1] if ',' in base64_image else base64_image
            media_type = _detect_image_format_mime(base64_image)
            
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": clean_base64
                }
            })
    
    # Add text if prompt is provided (text should come after images)
    if prompt:
        content.append({"type": "text", "text": prompt})
    
    return content

def build_content_with_images_openai_grok(prompt: str, images: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Build content array with text and images for OpenAI/Grok format."""
    content = []
    
    # Add text if prompt is provided (OpenAI/Grok require text first)
    if prompt:
        content.append({"type": "text", "text": prompt})
    
    # Add images if provided
    if images:
        for base64_image in images:
            # Remove data URL prefix if present
            clean_base64 = base64_image.split(',')[-1] if ',' in base64_image else base64_image
            image_format = _detect_image_format_extension(base64_image)
            
            # Build data URL
            data_url = f"data:image/{image_format};base64,{clean_base64}"
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": data_url
                }
            })
    
    return content

def build_content_with_images_gemini(prompt: str, images: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Build content array with text and images for Gemini format."""
    content = []
    
    # Gemini supports mixed content, we can add text and images in any order
    # But typically we add text first, then images
    
    # Add text if prompt is provided
    if prompt:
        content.append({"text": prompt})
    
    # Add images if provided
    if images:
        for base64_image in images:
            # Remove data URL prefix if present
            clean_base64 = base64_image.split(',')[-1] if ',' in base64_image else base64_image
            media_type = _detect_image_format_mime(base64_image)
            
            content.append({
                "inline_data": {
                    "mime_type": media_type,
                    "data": clean_base64
                }
            })
    
    return content
