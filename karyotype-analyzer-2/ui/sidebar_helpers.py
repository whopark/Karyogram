"""Sidebar helper: localStorage JavaScript injection for API key persistence."""

import json as json_module


def inject_local_storage_script(action: str, key_value: str = "") -> str:
    """Inject JavaScript to interact with browser localStorage."""
    if action == "save":
        safe_key = json_module.dumps(key_value)
        return f"""
        <script>
        localStorage.setItem('karyotype_openai_api_key', {safe_key});
        console.log('API key saved to localStorage');
        </script>
        """
    elif action == "clear":
        return """
        <script>
        localStorage.removeItem('karyotype_openai_api_key');
        console.log('API key cleared from localStorage');
        </script>
        """
    elif action == "load":
        return """
        <div id="ls-loader" style="display:none;"></div>
        <script>
        (function() {
            const savedKey = localStorage.getItem('karyotype_openai_api_key');
            if (savedKey) {
                document.getElementById('ls-loader').setAttribute('data-key', savedKey);
                const container = document.querySelector('[data-testid="stSidebar"]');
                if (container) {
                    let indicator = document.getElementById('saved-key-indicator');
                    if (!indicator) {
                        indicator = document.createElement('div');
                        indicator.id = 'saved-key-indicator';
                        indicator.style.display = 'none';
                        indicator.textContent = savedKey;
                        container.appendChild(indicator);
                    }
                }
            }
        })();
        </script>
        """
    return ""
