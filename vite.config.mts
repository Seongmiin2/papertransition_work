import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  // Electron loads the production UI from a file:// URL, so assets must be relative.
  base: "./",
  root: "apps/desktop",
  plugins: [react()],
  build: { outDir: "../../dist-renderer", emptyOutDir: true }
});
