document.addEventListener("DOMContentLoaded", function () {
    var script = document.createElement("script");
    script.type = "module";
    script.id = "runllm-widget-script"
  
    script.src = "https://widget.runllm.com";
  
    script.setAttribute("version", "stable");
    script.setAttribute("crossorigin", "true");
    script.setAttribute("runllm-keyboard-shortcut", "Mod+k");
    script.setAttribute("runllm-name", "CrowdCent Challenge Docs");
    script.setAttribute("runllm-position", "BOTTOM_RIGHT");
    script.setAttribute("runllm-assistant-id", "990");
    script.setAttribute("runllm-theme-color", "#62e4fb");
    script.setAttribute("runllm-support-email", "info@crowdcent.com");
    script.setAttribute("runllm-join-community-text", "Join our Discord");
    script.setAttribute("runllm-community-url", "https://discord.gg/v6ZSGuTbQS");

    script.async = true;
    document.head.appendChild(script);
  });