import { defineConfig } from 'vite'

// Same-origin proxy for the WebRTC relay pages: the browser only ever talks to
// this server over plain HTTP, so the relays' self-signed HTTPS certs never
// need to be accepted (Safari included). WebRTC media itself flows directly
// browser <-> relay over ICE/DTLS and does not pass through this proxy.
// Robot running the WebRTC relays; set CAM_HOST when the GUI runs on another PC.
const camHost = process.env.CAM_HOST ?? '127.0.0.1'

const camProxy = Object.fromEntries(
  [60001, 60002, 60003, 60004, 60005, 60006].map((port) => [
    `/cam/${port}`,
    {
      target: `https://${camHost}:${port}`,
      changeOrigin: true,
      secure: false, // relay cert is self-signed, loopback only
      rewrite: (p: string) => p.replace(`/cam/${port}`, ''),
    },
  ])
)

export default defineConfig({
  server: { proxy: camProxy },
})
