import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  server: {
    // The page never talks to a database or a model; it reads the published state the API serves.
    proxy: { "/api": "http://localhost:8000" },
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./src/test-setup.ts"],
    // Off by default, which makes a `?raw` import of the stylesheet come back as an empty string.
    // `restyle.test.tsx` reads the real stylesheet to check selector specificity, and a guard
    // handed an empty string is a guard that passes because it saw nothing.
    css: true,
  },
});
