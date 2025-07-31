import json

def parse_event_string(event_string):
    """
    Robustly parse event string or object.
    """
    result = {}
    # If already a dict, just return it (with label if present)
    if isinstance(event_string, dict):
        result.update(event_string)
        if 'label' not in result and 'event_type' in result:
            result['label'] = result['event_type']
        return result
    # If tuple, try to flatten
    if isinstance(event_string, tuple):
        # Try to flatten tuple like ('TRIAL_END', {...})
        if len(event_string) == 2 and isinstance(event_string[1], dict):
            result['label'] = event_string[0]
            result.update(event_string[1])
            return result
        # Fallback: string conversion
        event_string = str(event_string)
    # Now handle as string
    if not isinstance(event_string, str):
        event_string = str(event_string)
    if ':' in event_string:
        prefix, rest = event_string.split(':', 1)
        result['label'] = prefix
        # Try JSON
        try:
            data = json.loads(rest)
            if isinstance(data, dict):
                result.update(data)
                return result
        except Exception:
            pass
        # Fallback to old colon format
        parts = rest.split(':')
        idx = 0
        if parts and '=' not in parts[0]:
            if prefix == "TRIAL_END":
                result["trial_nr"] = parts[0]
            else:
                result[f"value_{idx}"] = parts[0]
            idx += 1
        for part in parts[idx:]:
            if '=' in part:
                k, v = part.split('=', 1)
                result[k] = v
            elif part:
                result[f"value_{idx}"] = part
                idx += 1
    else:
        result['label'] = event_string
    return result



import json
import re

def old_parse_event_string(event_str):
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