import json
import re

def parse_event_string(event_str):
    """
    Robustly parse event strings with possible multiple JSON, key-value, or label parts.
    Flattens all JSON and key-value pairs into a single dict.
    """
    if not isinstance(event_str, str):
        return {}

    # Split at top-level colons (not inside braces)
    def smart_split(s):
        parts = []
        depth = 0
        last = 0
        for i, c in enumerate(s):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
            elif c == ':' and depth == 0:
                parts.append(s[last:i])
                last = i+1
        parts.append(s[last:])
        return parts

    parts = smart_split(event_str)
    result = {}

    for p in parts:
        p = p.strip()
        # Try JSON
        try:
            if p.startswith('{') and p.endswith('}'):
                result.update(json.loads(p))
                continue
        except Exception:
            pass
        # Try key=value (comma or semicolon separated)
        if '=' in p:
            for kv in re.split(r'[;,]', p):
                if '=' in kv:
                    k, v = kv.split('=', 1)
                    result[k.strip()] = v.strip()
            continue
        # Try key:value (but not if it's the whole string)
        if ':' in p and p.count(':') == 1:
            k, v = p.split(':', 1)
            result[k.strip()] = v.strip()
            continue
        # Fallback: label
        if p:
            label_key = 'label' if 'label' not in result else f"label_{len([k for k in result if k.startswith('label')])}"
            result[label_key] = p

    return result