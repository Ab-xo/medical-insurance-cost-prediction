import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow next/image to optimise images served from the FastAPI backend.
  // Raw <img> tags are used for dynamic matplotlib PNGs from localhost:8000
  // because Next.js <Image> requires statically-known hosts at build time.
  images: {
    remotePatterns: [
      {
        protocol: "http",
        hostname: "localhost",
        port: "8000",
        pathname: "/outputs/**",
      },
      {
        protocol: "http",
        hostname: "127.0.0.1",
        port: "8000",
        pathname: "/outputs/**",
      },
    ],
  },

  // Remove the X-Powered-By: Next.js response header
  poweredByHeader: false,

  // Catch side-effect bugs early in development
  reactStrictMode: true,
};

export default nextConfig;
