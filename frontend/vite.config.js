import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

const normalizeBackendUrl = (rawUrl) => {
  if (!rawUrl) {
    return "http://127.0.0.1:8000";
  }

  try {
    return new URL(rawUrl).origin;
  } catch {
    return rawUrl;
  }
};

/** Suppress noisy ECONNREFUSED / proxy error logs while backend is starting */
function silentProxyErrorHandler(proxy) {
  proxy.on("error", (err, _req, res) => {
    const code = err.code || "";
    // Only suppress connection-refused / timeout — let real errors through
    if (code === "ECONNREFUSED" || code === "ECONNRESET" || code === "ETIMEDOUT") {
      if (res && typeof res.writeHead === "function" && !res.headersSent) {
        res.writeHead(503, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "Backend unavailable", code }));
      }
      return; // suppress console spam
    }
    console.error("[vite proxy]", err.message);
  });
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const backendUrl = normalizeBackendUrl(
    env.VITE_API_BASE_URL || env.VITE_BACKEND_URL || "http://127.0.0.1:8000",
  );
  const websocketUrl = backendUrl.replace(/^http/i, "ws");

  return {
    plugins: [react()],
    server: {
      host: "0.0.0.0",
      port: 5173,
      allowedHosts: ["short-schools-taste.loca.lt", ".loca.lt", ".trycloudflare.com"],
      proxy: {
        "/api": {
          target: backendUrl,
          changeOrigin: true,
          secure: false,
          proxyTimeout: 60000,
          timeout: 60000,
          configure: silentProxyErrorHandler,
        },
        "/ws": {
          target: websocketUrl,
          changeOrigin: true,
          ws: true,
          secure: false,
          proxyTimeout: 60000,
          timeout: 60000,
          configure: silentProxyErrorHandler,
        },
      },
    },
  };
});
