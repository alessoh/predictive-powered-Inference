import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    // In local dev the Python inference API (api/step.py on Vercel) is
    // served by scripts/dev_api.py; Vercel routes /api/step natively.
    if (process.env.NODE_ENV === "development") {
      return [{ source: "/api/step", destination: "http://127.0.0.1:8765/api/step" }];
    }
    return [];
  },
};

export default nextConfig;
