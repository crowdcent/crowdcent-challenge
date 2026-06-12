// Centaur "Ask AI" widget loader (replaces the retired RunLLM widget).
// The widget app is self-hosted at centaur.crowdcent.com and talks to the
// public /ask chat API there; this file only injects the loader script
// with the docs-site configuration. Keyboard shortcut and theme color
// match the old RunLLM setup.
document.addEventListener("DOMContentLoaded", function () {
    var script = document.createElement("script");
    script.id = "centaur-widget-script";

    script.src = "https://centaur.crowdcent.com/static/widget.js";
    script.setAttribute("data-endpoint", "https://centaur.crowdcent.com/ask");
    script.setAttribute("data-name", "CrowdCent Challenge Docs");
    script.setAttribute("data-keyboard-shortcut", "Mod+k");
    script.setAttribute("data-theme-color", "#62e4fb");
    script.setAttribute("data-support-email", "info@crowdcent.com");
    script.setAttribute("data-community-url", "https://discord.gg/v6ZSGuTbQS");

    script.async = true;
    document.head.appendChild(script);
});
