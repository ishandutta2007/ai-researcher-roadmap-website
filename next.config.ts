import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  basePath: "/ai-researcher-roadmap-website",
  assetPrefix: "/ai-researcher-roadmap-website/",
  output: 'export',
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
