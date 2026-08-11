/**
 * Next.js configuration.
 *
 * The frontend is a static-friendly App Router application that talks to the
 * FastAPI backend over CORS, so there are no rewrites or proxies here: the
 * browser calls NEXT_PUBLIC_API_URL directly and the backend allows the origin.
 *
 * @type {import('next').NextConfig}
 */
const nextConfig = {
  reactStrictMode: true,
};

module.exports = nextConfig;
